//! OSC (Open Sound Control) protocol handler for plugin <-> Python communication.
//!
//! Messages:
//!   /lossy/param/{name} <f32>      — set a single parameter
//!   /lossy/preset/load <string>    — load a named preset
//!   /lossy/request_snapshot        — trigger audio snapshot capture
//!   /lossy/metrics <f32> <f32>     — outgoing: RMS, peak

use std::net::UdpSocket;
use std::sync::mpsc;
use std::thread;

/// Incoming command from OSC client (Python GUI).
#[derive(Debug, Clone)]
pub enum OscCommand {
    SetParam { name: String, value: f32 },
    LoadPreset { name: String },
    RequestSnapshot,
}

/// OSC server that listens for commands on a UDP port.
pub struct OscServer {
    rx: mpsc::Receiver<OscCommand>,
    _thread: thread::JoinHandle<()>,
}

impl OscServer {
    /// Start listening on `port`. Commands are received via `recv()`.
    pub fn start(port: u16) -> std::io::Result<Self> {
        let socket = UdpSocket::bind(format!("127.0.0.1:{port}"))?;
        socket.set_nonblocking(false)?;
        let (tx, rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            let mut buf = [0u8; 2048];
            loop {
                let size = match socket.recv(&mut buf) {
                    Ok(n) => n,
                    Err(_) => continue,
                };
                let (_, packet) = match rosc::decoder::decode_udp(&buf[..size]) {
                    Ok(result) => result,
                    Err(_) => continue,
                };
                if let Some(cmd) = parse_packet(&packet) {
                    if tx.send(cmd).is_err() {
                        break; // receiver dropped
                    }
                }
            }
        });

        Ok(Self {
            rx,
            _thread: handle,
        })
    }

    /// Try to receive a command (non-blocking).
    pub fn try_recv(&self) -> Option<OscCommand> {
        self.rx.try_recv().ok()
    }

    /// Drain all pending commands.
    pub fn drain(&self) -> Vec<OscCommand> {
        let mut cmds = Vec::new();
        while let Ok(cmd) = self.rx.try_recv() {
            cmds.push(cmd);
        }
        cmds
    }
}

/// OSC client for sending messages to Python GUI.
pub struct OscClient {
    socket: UdpSocket,
    target: String,
}

impl OscClient {
    /// Create client that sends to `target_addr` (e.g. "127.0.0.1:9001").
    pub fn new(target_addr: &str) -> std::io::Result<Self> {
        let socket = UdpSocket::bind("127.0.0.1:0")?;
        Ok(Self {
            socket,
            target: target_addr.to_string(),
        })
    }

    /// Send metrics snapshot.
    pub fn send_metrics(&self, rms: f32, peak: f32) -> std::io::Result<()> {
        let msg = rosc::OscMessage {
            addr: "/lossy/metrics".to_string(),
            args: vec![rosc::OscType::Float(rms), rosc::OscType::Float(peak)],
        };
        let bytes = rosc::encoder::encode(&rosc::OscPacket::Message(msg))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        self.socket.send_to(&bytes, &self.target)?;
        Ok(())
    }

    /// Send snapshot-ready notification with file path.
    pub fn send_snapshot_ready(&self, path: &str) -> std::io::Result<()> {
        let msg = rosc::OscMessage {
            addr: "/lossy/snapshot_ready".to_string(),
            args: vec![rosc::OscType::String(path.to_string())],
        };
        let bytes = rosc::encoder::encode(&rosc::OscPacket::Message(msg))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        self.socket.send_to(&bytes, &self.target)?;
        Ok(())
    }
}

fn parse_packet(packet: &rosc::OscPacket) -> Option<OscCommand> {
    match packet {
        rosc::OscPacket::Message(msg) => parse_message(msg),
        rosc::OscPacket::Bundle(bundle) => {
            // Return first parseable message from bundle
            bundle.content.iter().find_map(parse_packet)
        }
    }
}

fn parse_message(msg: &rosc::OscMessage) -> Option<OscCommand> {
    if let Some(name) = msg.addr.strip_prefix("/lossy/param/") {
        let value = match msg.args.first()? {
            rosc::OscType::Float(f) => *f,
            rosc::OscType::Double(d) => *d as f32,
            rosc::OscType::Int(i) => *i as f32,
            _ => return None,
        };
        Some(OscCommand::SetParam {
            name: name.to_string(),
            value,
        })
    } else if msg.addr == "/lossy/preset/load" {
        let name = match msg.args.first()? {
            rosc::OscType::String(s) => s.clone(),
            _ => return None,
        };
        Some(OscCommand::LoadPreset { name })
    } else if msg.addr == "/lossy/request_snapshot" {
        Some(OscCommand::RequestSnapshot)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_set_param() {
        let msg = rosc::OscMessage {
            addr: "/lossy/param/loss".to_string(),
            args: vec![rosc::OscType::Float(0.7)],
        };
        let cmd = parse_message(&msg).unwrap();
        match cmd {
            OscCommand::SetParam { name, value } => {
                assert_eq!(name, "loss");
                assert!((value - 0.7).abs() < 1e-6);
            }
            _ => panic!("expected SetParam"),
        }
    }

    #[test]
    fn parse_load_preset() {
        let msg = rosc::OscMessage {
            addr: "/lossy/preset/load".to_string(),
            args: vec![rosc::OscType::String("frozen_pad".to_string())],
        };
        let cmd = parse_message(&msg).unwrap();
        match cmd {
            OscCommand::LoadPreset { name } => assert_eq!(name, "frozen_pad"),
            _ => panic!("expected LoadPreset"),
        }
    }

    #[test]
    fn parse_snapshot_request() {
        let msg = rosc::OscMessage {
            addr: "/lossy/request_snapshot".to_string(),
            args: vec![],
        };
        let cmd = parse_message(&msg).unwrap();
        assert!(matches!(cmd, OscCommand::RequestSnapshot));
    }

    #[test]
    fn roundtrip_encode_decode() {
        let msg = rosc::OscMessage {
            addr: "/lossy/param/verb".to_string(),
            args: vec![rosc::OscType::Float(0.5)],
        };
        let bytes =
            rosc::encoder::encode(&rosc::OscPacket::Message(msg)).expect("encode failed");
        let (_, packet) = rosc::decoder::decode_udp(&bytes).expect("decode failed");
        let cmd = parse_packet(&packet).unwrap();
        match cmd {
            OscCommand::SetParam { name, value } => {
                assert_eq!(name, "verb");
                assert!((value - 0.5).abs() < 1e-6);
            }
            _ => panic!("expected SetParam"),
        }
    }
}
