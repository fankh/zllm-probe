//! Behavioral guarantees of the proxy that a naive implementation silently
//! breaks: (1) upstream sees the client's auth header, (2) streaming bodies
//! relay incrementally instead of buffering until the upstream finishes,
//! (3) the detect path injects logprobs, strips our extension flag, and
//! attaches the report.
//!
//! Each test runs a scripted TCP upstream and the real probe-proxy binary.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Kills the proxy binary even when an assertion panics.
struct ProxyGuard(Child);
impl Drop for ProxyGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Spawn probe-proxy against `upstream`, wait until it accepts connections.
fn spawn_proxy(upstream: &str) -> (ProxyGuard, u16) {
    // Reserve an ephemeral port, free it, hand it to the proxy. (Tiny race,
    // fine for tests.)
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let child = Command::new(env!("CARGO_BIN_EXE_probe-proxy"))
        .env("PROBE_UPSTREAM", upstream)
        .env("PROBE_PORT", port.to_string())
        .spawn()
        .expect("spawn probe-proxy");
    let guard = ProxyGuard(child);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return (guard, port);
        }
        assert!(Instant::now() < deadline, "proxy did not come up");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Read one HTTP request (headers + content-length body) off a stream.
fn read_request(stream: &mut TcpStream) -> String {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut head = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" || line.is_empty() {
            break;
        }
        head.push_str(&line);
    }
    let clen = head
        .lines()
        .find_map(|l| {
            let l = l.to_ascii_lowercase();
            l.strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    let mut body = vec![0u8; clen];
    reader.read_exact(&mut body).unwrap();
    head.push_str(&String::from_utf8_lossy(&body));
    head
}

/// Non-detect streaming request: the upstream emits two SSE chunks ~1.2 s
/// apart (connection-close framing); the client must see the first chunk
/// while the upstream is still holding the second — i.e., the proxy did not
/// buffer — and the upstream must have received the Authorization header.
#[test]
fn streaming_relays_incrementally_and_forwards_auth() {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let up_addr = format!("http://{}", upstream.local_addr().unwrap());
    let (req_tx, req_rx) = mpsc::channel::<String>();

    std::thread::spawn(move || {
        let (mut sock, _) = upstream.accept().unwrap();
        let req = read_request(&mut sock);
        req_tx.send(req).unwrap();
        sock.write_all(
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
        )
        .unwrap();
        sock.write_all(b"data: {\"chunk\":1}\n\n").unwrap();
        sock.flush().unwrap();
        std::thread::sleep(Duration::from_millis(1200));
        sock.write_all(b"data: [DONE]\n\n").unwrap();
        sock.flush().unwrap();
    });

    let (_proxy, port) = spawn_proxy(&up_addr);
    let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    write!(
        client,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer test-token\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    )
    .unwrap();

    client.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let start = Instant::now();
    let mut buf = [0u8; 4096];
    let mut received = String::new();
    let mut first_chunk_at = None;
    loop {
        match client.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                received.push_str(&String::from_utf8_lossy(&buf[..n]));
                if first_chunk_at.is_none() && received.contains("{\"chunk\":1}") {
                    first_chunk_at = Some(start.elapsed());
                }
            }
            Err(e) => panic!("read: {e}"),
        }
    }
    let total = start.elapsed();

    let upstream_req = req_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(
        upstream_req.to_ascii_lowercase().contains("authorization: bearer test-token"),
        "upstream must receive the client's auth header; got:\n{upstream_req}"
    );
    assert!(received.contains("data: [DONE]"), "full stream must arrive:\n{received}");
    let first = first_chunk_at.expect("first chunk never seen");
    assert!(
        total - first > Duration::from_millis(700),
        "first chunk must arrive well before the stream ends (streaming, not buffered): \
         first at {first:?}, total {total:?}"
    );
}

/// Detect path: the proxy strips `detect_hallucination`, injects
/// `logprobs`/`top_logprobs`, forwards auth, and attaches the report.
#[test]
fn detect_injects_logprobs_and_attaches_report() {
    let upstream = TcpListener::bind("127.0.0.1:0").unwrap();
    let up_addr = format!("http://{}", upstream.local_addr().unwrap());
    let (req_tx, req_rx) = mpsc::channel::<String>();

    std::thread::spawn(move || {
        let (mut sock, _) = upstream.accept().unwrap();
        let req = read_request(&mut sock);
        req_tx.send(req).unwrap();
        let body = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"logprobs":{"content":[{"token":"hi","logprob":-0.05,"top_logprobs":[{"token":"hi","logprob":-0.05},{"token":"yo","logprob":-3.2}]}]}}]}"#;
        write!(
            sock,
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .unwrap();
    });

    let (_proxy, port) = spawn_proxy(&up_addr);
    let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"detect_hallucination":true}"#;
    write!(
        client,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nauthorization: Bearer tok2\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    )
    .unwrap();

    client.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut resp = String::new();
    client.read_to_string(&mut resp).unwrap();

    let upstream_req = req_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(
        !upstream_req.contains("detect_hallucination"),
        "extension flag must not leak upstream:\n{upstream_req}"
    );
    assert!(
        upstream_req.contains("\"logprobs\":true") && upstream_req.contains("\"top_logprobs\""),
        "proxy must inject the logprobs request:\n{upstream_req}"
    );
    assert!(
        upstream_req.to_ascii_lowercase().contains("authorization: bearer tok2"),
        "detect path must forward auth too:\n{upstream_req}"
    );
    assert!(
        resp.contains("\"hallucination\"") && resp.contains("\"risk_score\""),
        "response must carry the report:\n{resp}"
    );
}
