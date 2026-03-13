fn main() -> Result<(), anyhow::Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // On macOS, automatically upgrade `bundle` to `bundle-universal` unless
    // the user explicitly passed --target or already said `bundle-universal`.
    if cfg!(target_os = "macos") {
        let command = args.first().map(|s| s.as_str());
        let has_target = args.iter().any(|a| a == "--target" || a.starts_with("--target="));

        if command == Some("bundle") && !has_target {
            let new_args: Vec<String> = std::iter::once("bundle-universal".to_string())
                .chain(args.into_iter().skip(1))
                .collect();
            return nih_plug_xtask::main_with_args("cargo xtask", new_args);
        }
    }

    nih_plug_xtask::main_with_args("cargo xtask", args)
}
