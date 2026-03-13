//! Debug: verify Client.start + message works end-to-end.
#![cfg(feature = "integration")]

use claudesdk::{Client, ClientConfig};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simple_message() {
    let mut client = Client::start(ClientConfig {
        model: "sonnet".into(),
        system_prompt: Some("Reply with one word.".into()),
        permission_mode: "plan".into(),
        auto_approve: true,
    })
    .await
    .expect("start");

    eprintln!("Client started. Sending message...");

    let turn = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        client.message("Say hello"),
    )
    .await
    .expect("timeout")
    .expect("message");

    eprintln!("Response: {:?}", turn.text);
    assert!(!turn.text.is_empty());
    assert!(!turn.is_error);

    client.stop().await;
}
