use mdns_sd::{ServiceDaemon, ServiceEvent};
use rust_cast::channels::media::{IdleReason, Media, MediaResponse, PlayerState, StreamType};
use rust_cast::channels::heartbeat::HeartbeatResponse;
use rust_cast::channels::receiver::CastDeviceApp;
use rust_cast::{CastDevice, ChannelMessage};
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, TcpListener, TcpStream, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const DISCOVERY_TIME: Duration = Duration::from_secs(4);
const RECEIVER: &str = "receiver-0";

pub struct Speaker {
    pub name: String,
    pub ip: IpAddr,
    pub port: u16,
}

/// Finds Cast devices on the local network.
pub fn discover() -> Result<Vec<Speaker>, Box<dyn Error>> {
    let mdns = ServiceDaemon::new()?;
    let events = mdns.browse("_googlecast._tcp.local.")?;
    let deadline = Instant::now() + DISCOVERY_TIME;
    let mut speakers: Vec<Speaker> = Vec::new();

    while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
        let Ok(event) = events.recv_timeout(remaining) else {
            break;
        };
        let ServiceEvent::ServiceResolved(info) = event else {
            continue;
        };
        let name = info
            .txt_properties
            .get_property_val_str("fn")
            .unwrap_or(&info.fullname)
            .to_string();
        let ip = info
            .addresses
            .iter()
            .map(|address| address.to_ip_addr())
            .find(IpAddr::is_ipv4);
        if let Some(ip) = ip
            && !speakers.iter().any(|speaker| speaker.name == name)
        {
            speakers.push(Speaker {
                name,
                ip,
                port: info.port,
            });
        }
    }
    let _ = mdns.shutdown();
    speakers.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(speakers)
}

/// Picks the one speaker whose name contains `query`, ignoring case.
pub fn find(query: &str) -> Result<Speaker, Box<dyn Error>> {
    let speakers = discover()?;
    let names = || {
        speakers
            .iter()
            .map(|speaker| speaker.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let query_lower = query.to_lowercase();
    let matches = speakers
        .iter()
        .filter(|speaker| speaker.name.to_lowercase().contains(&query_lower))
        .count();
    if matches != 1 {
        return Err(format!(
            "{matches} speakers match \"{query}\" (found: {})",
            names()
        )
        .into());
    }
    speakers
        .into_iter()
        .find(|speaker| speaker.name.to_lowercase().contains(&query_lower))
        .ok_or_else(|| "speaker disappeared".into())
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("mp3") => "audio/mpeg",
        Some("m4a" | "m4b") => "audio/mp4",
        Some("flac") => "audio/flac",
        _ => "audio/wav",
    }
}

fn wav_duration(path: &Path) -> Option<f32> {
    let bytes = std::fs::metadata(path).ok()?.len();
    // Good enough for a playback timeout: mono 16-bit at the model's rate.
    (content_type(path) == "audio/wav").then(|| bytes as f32 / 48_000.0)
}

/// Answers each `GET /<index>` with that file, so nothing else can be fetched.
fn handle_request(mut stream: TcpStream, files: &[PathBuf]) -> std::io::Result<()> {
    let mut line = String::new();
    BufReader::new(stream.try_clone()?).read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let file = parts
        .next()
        .and_then(|path| path.strip_prefix('/'))
        .and_then(|index| index.parse::<usize>().ok())
        .and_then(|index| files.get(index));

    let Some(file) = file else {
        return stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    };
    let body = std::fs::read(file)?;
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        content_type(file),
        body.len()
    )?;
    if method == "GET" {
        stream.write_all(&body)?;
    }
    Ok(())
}

/// Serves `files` on a temporary port for as long as the process runs.
fn serve(speaker: &Speaker, files: Vec<PathBuf>) -> Result<String, Box<dyn Error>> {
    // The address the speaker can reach us on is whichever routes towards it.
    let probe = UdpSocket::bind("0.0.0.0:0")?;
    probe.connect((speaker.ip, speaker.port))?;
    let listener = TcpListener::bind((probe.local_addr()?.ip(), 0))?;
    let base = format!("http://{}", listener.local_addr()?);

    let files = Arc::new(files);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let files = Arc::clone(&files);
            std::thread::spawn(move || {
                if let Err(error) = handle_request(stream, &files) {
                    eprintln!("serving audio: {error}");
                }
            });
        }
    });
    Ok(base)
}

/// Plays `files` in order on `speaker`, returning once the last one finishes.
pub fn play(speaker: &Speaker, files: Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    // Dependencies enable two TLS backends, so rustls needs telling which to use.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let base = serve(speaker, files.clone())?;
    let device =
        CastDevice::connect_without_host_verification(speaker.ip.to_string(), speaker.port)?;
    device.connection.connect(RECEIVER)?;
    device.heartbeat.ping()?;
    let app = device.receiver.launch_app(&CastDeviceApp::DefaultMediaReceiver)?;
    device.connection.connect(app.transport_id.as_str())?;

    let outcome = files.iter().enumerate().try_for_each(|(index, file)| {
        println!("Playing {} on {}", file.display(), speaker.name);
        let media = Media {
            content_id: format!("{base}/{index}"),
            stream_type: StreamType::Buffered,
            content_type: content_type(file).to_string(),
            metadata: None,
            duration: None,
        };
        device
            .media
            .load(app.transport_id.as_str(), app.session_id.as_str(), &media)?;
        wait_until_finished(&device, wav_duration(file))
    });

    let _ = device.receiver.stop_app(app.session_id.as_str());
    outcome
}

fn wait_until_finished(
    device: &CastDevice<'_>,
    duration: Option<f32>,
) -> Result<(), Box<dyn Error>> {
    let limit = Duration::from_secs_f32(duration.unwrap_or(4.0 * 3600.0) + 60.0);
    let started = Instant::now();
    while started.elapsed() < limit {
        match device.receive()? {
            ChannelMessage::Heartbeat(HeartbeatResponse::Ping) => device.heartbeat.pong()?,
            ChannelMessage::Media(MediaResponse::Status(status)) => {
                for entry in status.entries {
                    if entry.player_state != PlayerState::Idle {
                        continue;
                    }
                    match entry.idle_reason {
                        Some(IdleReason::Finished) => return Ok(()),
                        Some(IdleReason::Error) => return Err("the speaker couldn't play the file".into()),
                        Some(IdleReason::Cancelled | IdleReason::Interrupted) => {
                            return Err("playback was stopped on the speaker".into());
                        }
                        None => {}
                    }
                }
            }
            ChannelMessage::Media(MediaResponse::LoadFailed(_)) => {
                return Err("the speaker failed to load the file".into());
            }
            _ => {}
        }
    }
    Err("timed out waiting for playback to finish".into())
}
