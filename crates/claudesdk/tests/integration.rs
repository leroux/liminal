//! Integration tests for claudesdk.
//!
//! Run: `cargo test -p claudesdk --features integration -- --nocapture`
#![cfg(feature = "integration")]

use claudesdk::{Client, ClientConfig, Event};

fn config() -> ClientConfig {
    ClientConfig {
        model: "sonnet".into(),
        system_prompt: Some("Reply in one short sentence.".into()),
        permission_mode: "plan".into(),
        auto_approve: true,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn start_and_stop() {
    let mut client = Client::start(config()).await.expect("start");
    eprintln!("Client started");
    client.stop().await;
    eprintln!("Client stopped");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simple_message() {
    let mut client = Client::start(config()).await.expect("start");

    let turn = client.message("What is 2+2?").await.expect("message");
    eprintln!("Response: {}", turn.text);
    eprintln!("Cost: {:?}, Duration: {:?}ms", turn.cost_usd, turn.duration_ms);

    assert!(!turn.text.is_empty(), "should have text");
    assert!(!turn.is_error, "should not be error");
    assert!(!turn.session_id.is_empty(), "should have session_id");

    client.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_events() {
    let mut client = Client::start(config()).await.expect("start");

    client.send("Say hello").await.expect("send");

    let mut got_delta = false;
    let mut got_message = false;
    let mut got_done = false;

    while let Some(event) = client.recv().await {
        match event {
            Event::TextDelta(s) => {
                eprint!("{s}");
                got_delta = true;
            }
            Event::Message(s) => {
                eprintln!("\nFull message: {s}");
                got_message = true;
            }
            Event::TurnDone { cost_usd, .. } => {
                eprintln!("Done. Cost: {cost_usd:?}");
                got_done = true;
                break;
            }
            Event::Error(e) => panic!("Error: {e}"),
            _ => {}
        }
    }

    // We should get at least the assembled message and done
    assert!(got_message || got_delta, "should have text");
    assert!(got_done, "should have TurnDone");

    client.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_turn() {
    let mut client = Client::start(config()).await.expect("start");

    let t1 = client.message("What is 2+2?").await.expect("turn 1");
    eprintln!("Turn 1: {}", t1.text);
    assert!(!t1.text.is_empty());

    let t2 = client.message("Add 3 to that").await.expect("turn 2");
    eprintln!("Turn 2: {}", t2.text);
    assert!(!t2.text.is_empty());

    client.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn json_in_response() {
    let config = ClientConfig {
        model: "sonnet".into(),
        system_prompt: Some(
            "You tune audio effects. When asked to set params, respond with a ```json block \
             containing key-value pairs. Available: loss (0-1), crush (0-1)."
                .into(),
        ),
        permission_mode: "plan".into(),
        auto_approve: true,
    };

    let mut client = Client::start(config).await.expect("start");

    let turn = client
        .message("Set loss to 0.7 and crush to 0.3")
        .await
        .expect("message");

    eprintln!("Response: {}", turn.text);
    assert!(
        turn.text.contains("json") || turn.text.contains("loss"),
        "should contain JSON params"
    );

    client.stop().await;
}
