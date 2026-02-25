use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let preset_dir = Path::new(&manifest_dir).join("../../../reverb/gui/presets");
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("embedded_presets.rs");

    let mut code = String::from("pub static EMBEDDED_PRESETS: &[(&str, &str)] = &[\n");

    if let Ok(entries) = fs::read_dir(&preset_dir) {
        let mut files: Vec<_> = entries
            .flatten()
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
            .collect();
        files.sort_by_key(|e| e.file_name());

        for entry in files {
            let path = entry.path().canonicalize().unwrap();
            let name = entry.path().file_stem().unwrap().to_str().unwrap().to_string();
            // Use forward slashes for include_str! — backslashes from Windows
            // canonicalize() (e.g. \\?\D:\...) are treated as escape sequences.
            let path_str = path.to_str().unwrap().replace('\\', "/");
            code.push_str(&format!(
                "    (\"{}\", include_str!(\"{}\")),\n",
                name, path_str
            ));
        }
    }

    code.push_str("];\n");
    fs::write(&dest, code).unwrap();

    println!("cargo:rerun-if-changed={}", preset_dir.display());
}
