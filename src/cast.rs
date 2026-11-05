use mdns_sd::{ServiceDaemon, ServiceEvent};
use rust_cast::channels::media::{IdleReason, Media, MediaResponse, PlayerState, StreamType};
use rust_cast::channels::connection::ConnectionResponse;
use rust_cast::channels::heartbeat::HeartbeatResponse;
use rust_cast::channels::receiver::{CastDeviceApp, ReceiverResponse};
use rust_cast::{CastDevice, ChannelMessage};
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, TcpListener, TcpStream, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const DISCOVERY_TIME: Duration = Duration::from_secs(4);
const RECEIVER: &str = "receiver-0";
/// How long past the file's length a speaker can go before playback is given up on.
const PLAYBACK_MARGIN: Duration = Duration::from_secs(60);

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

/// Reads `bytes=start-end` into the inclusive range to send, if it's satisfiable.
fn parse_range(header: &str, size: usize) -> Option<(usize, usize)> {
    let (start, end) = header.trim().strip_prefix("bytes=")?.split_once('-')?;
    let last = size.checked_sub(1)?;
    if start.is_empty() {
        let suffix = end.parse::<usize>().ok().filter(|&suffix| suffix > 0)?;
        return Some((size.saturating_sub(suffix), last));
    }
    let start = start.parse::<usize>().ok().filter(|&start| start <= last)?;
    let end = if end.is_empty() { last } else { end.parse::<usize>().ok()?.min(last) };
    (start <= end).then_some((start, end))
}

/// Answers each `GET /<index>` with that file, so nothing else can be fetched.
/// Honours byte ranges, which speakers use to seek within a file.
fn handle_request(mut stream: TcpStream, files: &[PathBuf]) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut range = None;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("range") {
                range = Some(value.trim().to_string());
            }
        }
    }
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
    let size = body.len();
    let (status, extra, sent) = match range {
        Some(range) => match parse_range(&range, size) {
            Some((start, end)) => (
                "206 Partial Content",
                format!("Content-Range: bytes {start}-{end}/{size}\r\n"),
                &body[start..=end],
            ),
            None => {
                return write!(
                    stream,
                    "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{size}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        },
        None => ("200 OK", String::new(), &body[..]),
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n{extra}Connection: close\r\n\r\n",
        content_type(file),
        sent.len()
    )?;
    if method == "GET" {
        stream.write_all(sent)?;
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
        wait_until_finished(&device, app.session_id.as_str())
    });

    let _ = device.receiver.stop_app(app.session_id.as_str());
    outcome
}

/// Decides when a speaker has gone quiet for too long. Only time spent not paused
/// counts, and there's no limit until the speaker reports the file's length.
struct PlaybackClock {
    counted: Duration,
    since: Instant,
    paused: bool,
    length: Option<Duration>,
}

impl PlaybackClock {
    fn new(now: Instant) -> Self {
        Self { counted: Duration::ZERO, since: now, paused: false, length: None }
    }

    fn counted_at(&self, now: Instant) -> Duration {
        if self.paused {
            self.counted
        } else {
            self.counted + now.saturating_duration_since(self.since)
        }
    }

    /// Records a status from the speaker.
    fn observe(&mut self, now: Instant, paused: bool, length: Option<f32>) {
        self.counted = self.counted_at(now);
        self.since = now;
        self.paused = paused;
        if let Some(length) = length.and_then(|seconds| Duration::try_from_secs_f32(seconds).ok()) {
            self.length = Some(length);
        }
    }

    fn timed_out(&self, now: Instant) -> bool {
        self.length.is_some_and(|length| self.counted_at(now) > length + PLAYBACK_MARGIN)
    }
}

fn wait_until_finished(
    device: &CastDevice<'_>,
    session_id: &str,
) -> Result<(), Box<dyn Error>> {
    let mut clock = PlaybackClock::new(Instant::now());
    let mut session_seen = false;
    while !clock.timed_out(Instant::now()) {
        match device.receive()? {
            ChannelMessage::Heartbeat(HeartbeatResponse::Ping) => device.heartbeat.pong()?,
            ChannelMessage::Media(MediaResponse::Status(status)) => {
                for entry in status.entries {
                    clock.observe(
                        Instant::now(),
                        entry.player_state == PlayerState::Paused,
                        entry.media.as_ref().and_then(|media| media.duration),
                    );
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
            // Stopping from the speaker or an app closes the session, often
            // without a final media status.
            ChannelMessage::Connection(ConnectionResponse::Close) => {
                return Err("playback was stopped on the speaker".into());
            }
            // A display can report no apps just after a launch, so a session
            // can only have stopped once it has been seen running.
            ChannelMessage::Receiver(ReceiverResponse::Status(status)) => {
                let running = status.applications.iter().any(|app| app.session_id == session_id);
                if session_seen && !running {
                    return Err("playback was stopped on the speaker".into());
                }
                session_seen |= running;
            }
            ChannelMessage::Media(MediaResponse::LoadFailed(_)) => {
                return Err("the speaker failed to load the file".into());
            }
            _ => {}
        }
    }
    Err("timed out waiting for playback to finish".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn after(start: Instant, minutes: u64) -> Instant {
        start + Duration::from_secs(minutes * 60)
    }

    #[test]
    fn an_unknown_length_never_times_out() {
        let start = Instant::now();
        let clock = PlaybackClock::new(start);
        assert!(!clock.timed_out(after(start, 100 * 60)));
    }

    #[test]
    fn a_known_length_times_out_a_minute_after_it_should_have_ended() {
        let start = Instant::now();
        let mut clock = PlaybackClock::new(start);
        clock.observe(start, false, Some(600.0));
        assert!(!clock.timed_out(after(start, 10)));
        assert!(clock.timed_out(after(start, 12)));
    }

    #[test]
    fn a_book_longer_than_four_hours_plays_to_its_end() {
        let start = Instant::now();
        let mut clock = PlaybackClock::new(start);
        clock.observe(start, false, Some(6.0 * 3600.0));
        assert!(!clock.timed_out(after(start, 6 * 60)));
    }

    #[test]
    fn time_spent_paused_does_not_count() {
        let start = Instant::now();
        let mut clock = PlaybackClock::new(start);
        clock.observe(start, false, Some(600.0));
        clock.observe(after(start, 5), true, None);
        assert!(!clock.timed_out(after(start, 5 * 60)));
        clock.observe(after(start, 5 * 60), false, None);
        assert!(!clock.timed_out(after(start, 5 * 60 + 5)));
        assert!(clock.timed_out(after(start, 5 * 60 + 7)));
    }

    #[test]
    fn ranges_are_clamped_to_the_file() {
        assert_eq!(parse_range("bytes=0-", 10), Some((0, 9)));
        assert_eq!(parse_range("bytes=2-4", 10), Some((2, 4)));
        assert_eq!(parse_range("bytes=8-99", 10), Some((8, 9)));
        assert_eq!(parse_range("bytes=-3", 10), Some((7, 9)));
        assert_eq!(parse_range("bytes=10-", 10), None);
        assert_eq!(parse_range("bytes=5-2", 10), None);
        assert_eq!(parse_range("bytes=0-", 0), None);
    }
}
