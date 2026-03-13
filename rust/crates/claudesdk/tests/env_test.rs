//! Quick test: verify env_remove works with tokio Command.
#![cfg(feature = "integration")]

use tokio::process::Command;

#[tokio::test]
async fn env_remove_works() {
    // Check parent has CLAUDECODE
    let parent_val = std::env::var("CLAUDECODE").unwrap_or_default();
    eprintln!("Parent CLAUDECODE={parent_val:?}");

    // Spawn a child that prints CLAUDECODE
    let output = Command::new("env")
        .env_remove("CLAUDECODE")
        .env_remove("CLAUDE_CODE_SSE_PORT")
        .output()
        .await
        .expect("env");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let has_claudecode = stdout.lines().any(|l| l.starts_with("CLAUDECODE="));
    let has_sse = stdout.lines().any(|l| l.starts_with("CLAUDE_CODE_SSE_PORT="));
    eprintln!("Child has CLAUDECODE={has_claudecode}, CLAUDE_CODE_SSE_PORT={has_sse}");

    assert!(!has_claudecode, "CLAUDECODE should be removed");
    assert!(!has_sse, "CLAUDE_CODE_SSE_PORT should be removed");
}

#[tokio::test]
async fn claude_without_nesting() {
    // Spawn claude with env removed, just check if it starts
    let output = Command::new("claude")
        .args(["--print", "--output-format", "stream-json", "--model", "sonnet", "-p", "say ok"])
        .env_remove("CLAUDECODE")
        .env_remove("CLAUDE_CODE_SSE_PORT")
        .output()
        .await
        .expect("claude");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("exit={}", output.status);
    eprintln!("stdout lines={}", stdout.lines().count());
    eprintln!("stderr: {}", &stderr[..stderr.len().min(200)]);

    assert!(output.status.success(), "claude should exit 0");
    assert!(!stdout.is_empty(), "should have output");

    // First line should be system.init
    let first = stdout.lines().next().unwrap_or("");
    assert!(first.contains("system"), "first line should be system.init, got: {}", &first[..first.len().min(100)]);
}
