//! Integration tests for pedal-chat (sync wrapper).
//!
//! Run: `cargo test -p pedal-chat --features integration -- --nocapture`
#![cfg(feature = "integration")]

use pedal_chat::{ChatBackend, ChatMsg};
use std::time::{Duration, Instant};

fn poll_until_done(backend: &ChatBackend, timeout: Duration) -> Vec<ChatMsg> {
    let start = Instant::now();
    let mut all = Vec::new();
    loop {
        let msgs = backend.poll();
        for msg in &msgs {
            if matches!(msg, ChatMsg::Done) {
                all.extend(msgs);
                return all;
            }
        }
        all.extend(msgs);
        if start.elapsed() > timeout {
            panic!("Timed out after {timeout:?}. Got {} messages.", all.len());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn simple_chat() {
    let backend = ChatBackend::new("Reply with one word.");
    backend.send("Say hello");

    let msgs = poll_until_done(&backend, Duration::from_secs(30));

    let texts: Vec<_> = msgs.iter().filter(|m| matches!(m, ChatMsg::Text(_))).collect();
    let errors: Vec<_> = msgs.iter().filter(|m| matches!(m, ChatMsg::Error(_))).collect();

    assert!(errors.is_empty(), "Got errors: {errors:?}");
    assert!(!texts.is_empty(), "Should have text");

    if let Some(ChatMsg::Text(t)) = texts.last() {
        eprintln!("Response: {t}");
        assert!(!t.trim().is_empty());
    }
}

#[test]
fn multi_turn() {
    let backend = ChatBackend::new("Reply in one short sentence.");

    backend.send("What is 2+2?");
    let msgs1 = poll_until_done(&backend, Duration::from_secs(30));
    let t1 = msgs1.iter().filter_map(|m| match m { ChatMsg::Text(t) => Some(t.clone()), _ => None }).last().unwrap();
    eprintln!("Turn 1: {t1}");

    backend.send("Add 3");
    let msgs2 = poll_until_done(&backend, Duration::from_secs(30));
    let t2 = msgs2.iter().filter_map(|m| match m { ChatMsg::Text(t) => Some(t.clone()), _ => None }).last().unwrap();
    eprintln!("Turn 2: {t2}");
}
