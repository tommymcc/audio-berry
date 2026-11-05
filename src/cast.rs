use mdns_sd::{ServiceDaemon, ServiceEvent};
use crate::epub::image_type;
use rust_cast::channels::media::{
    IdleReason, Image, Media, MediaResponse, Metadata, MusicTrackMediaMetadata, PlayerState, StreamType,
};
use rust_cast::channels::connection::ConnectionResponse;
use rust_cast::channels::heartbeat::HeartbeatResponse;
use rust_cast::channels::receiver::{CastDeviceApp, ReceiverResponse};
use rust_cast::{CastDevice, ChannelMessage};
use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::{IpAddr, TcpListener, TcpStream, UdpSocket};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::process::Command;
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

/// What a display shows while a file plays.
#[derive(Debug, Default, PartialEq)]
struct Track {
    title: String,
    artist: Option<String>,
    album: Option<String>,
    number: Option<u32>,
    cover: Option<Vec<u8>>,
}

/// Turns a name like `03-The_Pool` into track 3, "The Pool".
fn track_from_name(path: &Path) -> Track {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let (number, rest) = match stem.split_once('-') {
        Some((digits, rest)) if !rest.is_empty() => match digits.parse::<u32>() {
            Ok(number) => (Some(number), rest),
            Err(_) => (None, stem.as_ref()),
        },
        _ => (None, stem.as_ref()),
    };
    let title = rest.replace('_', " ").trim().to_string();
    Track { title, number, ..Track::default() }
}

/// Reads a file's title, artist and album tags, if ffprobe can.
fn read_tags(path: &Path) -> Option<serde_json::Map<String, serde_json::Value>> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "format_tags", "-of", "json"])
        .arg(path)
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    json.get("format")?.get("tags")?.as_object().cloned()
}

/// A picture inside the file, such as an M4B's cover, or else a `cover.jpg` or
/// `cover.png` beside it or one folder up, where `book` saves one.
fn find_cover(path: &Path) -> Option<Vec<u8>> {
    let embedded = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", "0:v:0", "-c", "copy", "-frames:v", "1", "-f", "image2pipe", "-"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| output.stdout);
    let folder = path.parent()?;
    let beside = [folder, folder.parent().unwrap_or(folder)]
        .into_iter()
        .flat_map(|dir| ["cover.jpg", "cover.png"].map(|name| dir.join(name)))
        .filter_map(|cover| std::fs::read(cover).ok());
    embedded.into_iter().chain(beside).find(|bytes| image_type(bytes).is_some())
}

/// Describes a file from its tags, falling back on its name. Missing details only
/// leave the display emptier, so they never stop playback.
fn describe(path: &Path) -> Track {
    let mut track = track_from_name(path);
    let tags = read_tags(path).unwrap_or_default();
    let tag = |name: &str| {
        tags.iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .and_then(|(_, value)| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(String::from)
    };
    if let Some(title) = tag("title") {
        track.title = title;
    }
    track.artist = tag("artist");
    track.album = tag("album");
    track.cover = find_cover(path);
    track
}

/// Something the speaker can fetch: an audio file, or a cover held in memory.
enum Resource {
    File(PathBuf),
    Image(Vec<u8>),
}

impl Resource {
    fn content_type(&self) -> &'static str {
        match self {
            Resource::File(path) => content_type(path),
            Resource::Image(bytes) => image_type(bytes).map_or("application/octet-stream", |(mime, _)| mime),
        }
    }

    fn body(&self) -> std::io::Result<Cow<'_, [u8]>> {
        match self {
            Resource::File(path) => std::fs::read(path).map(Cow::Owned),
            Resource::Image(bytes) => Ok(Cow::Borrowed(bytes)),
        }
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

/// Answers each `GET /<index>` with that resource, so nothing else can be fetched.
/// Honours byte ranges, which speakers use to seek within a file.
fn handle_request(mut stream: TcpStream, files: &[Resource]) -> std::io::Result<()> {
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
    let body = file.body()?;
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
        file.content_type(),
        sent.len()
    )?;
    if method == "GET" {
        stream.write_all(sent)?;
    }
    Ok(())
}

/// Serves `files` on a temporary port for as long as the process runs.
fn serve(speaker: &Speaker, files: Vec<Resource>) -> Result<String, Box<dyn Error>> {
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
    // Covers are served after the audio, each picture once.
    let tracks: Vec<Track> = files.iter().map(|file| describe(file)).collect();
    let mut covers: Vec<&[u8]> = Vec::new();
    let cover_index: Vec<Option<usize>> = tracks
        .iter()
        .map(|track| {
            let cover = track.cover.as_deref()?;
            let position = covers.iter().position(|seen| *seen == cover).unwrap_or_else(|| {
                covers.push(cover);
                covers.len() - 1
            });
            Some(files.len() + position)
        })
        .collect();
    let resources = files
        .iter()
        .cloned()
        .map(Resource::File)
        .chain(covers.iter().map(|cover| Resource::Image(cover.to_vec())))
        .collect();
    let base = serve(speaker, resources)?;
    let device =
        CastDevice::connect_without_host_verification(speaker.ip.to_string(), speaker.port)?;
    device.connection.connect(RECEIVER)?;
    device.heartbeat.ping()?;
    let app = device.receiver.launch_app(&CastDeviceApp::DefaultMediaReceiver)?;
    device.connection.connect(app.transport_id.as_str())?;

    let outcome = files.iter().enumerate().try_for_each(|(index, file)| {
        let track = &tracks[index];
        println!("Playing {} ({}) on {}", track.title, file.display(), speaker.name);
        let media = Media {
            content_id: format!("{base}/{index}"),
            stream_type: StreamType::Buffered,
            content_type: content_type(file).to_string(),
            metadata: Some(Metadata::MusicTrack(MusicTrackMediaMetadata {
                title: Some(track.title.clone()),
                artist: track.artist.clone(),
                album_name: track.album.clone(),
                track_number: track.number,
                images: cover_index[index]
                    .map(|cover| Image::new(format!("{base}/{cover}")))
                    .into_iter()
                    .collect(),
                ..MusicTrackMediaMetadata::default()
            })),
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
    fn a_chapter_file_name_gives_its_number_and_title() {
        let track = track_from_name(Path::new("out/chapters/03-The_Pool_of_Tears.wav"));
        assert_eq!(track.number, Some(3));
        assert_eq!(track.title, "The Pool of Tears");
        let track = track_from_name(Path::new("Alice-in-Wonderland.m4b"));
        assert_eq!(track.number, None);
        assert_eq!(track.title, "Alice-in-Wonderland");
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
