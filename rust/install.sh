#!/usr/bin/env bash
# Build and install all plugins (VST3 + CLAP) to system plugin directories.
# Usage: ./install.sh [plugin-name]
#   ./install.sh              # all plugins
#   ./install.sh lossy        # just lossy-plugin
set -euo pipefail

VST3_DIR="$HOME/Library/Audio/Plug-Ins/VST3"
CLAP_DIR="$HOME/Library/Audio/Plug-Ins/CLAP"
BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)/target/bundled"

PLUGINS=(lossy-plugin fractal-plugin reverb-plugin)

if [ "${1:-}" != "" ]; then
    # Accept "lossy" or "lossy-plugin"
    name="${1%-plugin}-plugin"
    PLUGINS=("$name")
fi

cd "$(dirname "$0")"

for plugin in "${PLUGINS[@]}"; do
    echo "==> Building $plugin..."
    cargo xtask bundle "$plugin" --release

    echo "==> Installing $plugin..."
    for fmt in vst3 clap; do
        src="$BUNDLE_DIR/$plugin.$fmt"
        case "$fmt" in
            vst3) dst="$VST3_DIR/$plugin.$fmt" ;;
            clap) dst="$CLAP_DIR/$plugin.$fmt" ;;
        esac
        if [ -d "$src" ]; then
            rm -rf "$dst"
            /bin/cp -R "$src" "$dst"
            echo "    $dst"
        fi
    done
done

echo "Done. Restart your DAW to load the new builds."
