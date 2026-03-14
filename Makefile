# VST Plugins — build helpers
# Usage:
#   make              — release build + bundle lossy (VST3 + CLAP)
#   make install      — copy lossy plugins to system plugin dirs (macOS)
#   make run          — launch lossy standalone
#   make fractal      — release build + bundle fractal
#   make install-fractal — copy fractal plugins to system plugin dirs
#   make run-fractal  — launch fractal standalone
#   make all-plugins  — build both lossy and fractal
#   make install-all  — install both plugins
#   make debug        — debug build + bundle lossy
#   make export-presets — export JSON presets to .vstpreset for Ableton
#   make clean        — cargo clean

BUNDLE  := target/bundled
LOSSY   := lossy-plugin
FRACTAL := fractal-plugin
REVERB  := reverb-plugin

# macOS plugin install dirs
VST3_DIR := $(HOME)/Library/Audio/Plug-Ins/VST3
CLAP_DIR := $(HOME)/Library/Audio/Plug-Ins/CLAP

.PHONY: all release debug install uninstall run clean \
        fractal install-fractal uninstall-fractal run-fractal \
        reverb install-reverb uninstall-reverb run-reverb \
        all-plugins install-all export-presets

all: release

# --- Lossy ---
release:
	cargo xtask bundle $(LOSSY) --release

debug:
	cargo xtask bundle $(LOSSY)

install: release export-presets
	@mkdir -p "$(VST3_DIR)" "$(CLAP_DIR)"
	@echo "Installing VST3 → $(VST3_DIR)"
	@rm -rf "$(VST3_DIR)/$(LOSSY).vst3"
	@cp -r "$(BUNDLE)/$(LOSSY).vst3" "$(VST3_DIR)/"
	@echo "Installing CLAP → $(CLAP_DIR)"
	@rm -rf "$(CLAP_DIR)/$(LOSSY).clap"
	@cp -r "$(BUNDLE)/$(LOSSY).clap" "$(CLAP_DIR)/"
	@echo "Done. Restart your DAW to pick up changes."

uninstall:
	rm -rf "$(VST3_DIR)/$(LOSSY).vst3"
	rm -rf "$(CLAP_DIR)/$(LOSSY).clap"
	@echo "Removed lossy plugins. Restart your DAW."

run: release
	cargo run --release --bin lossy-standalone

# --- Fractal ---
fractal:
	cargo xtask bundle $(FRACTAL) --release

fractal-debug:
	cargo xtask bundle $(FRACTAL)

install-fractal: fractal export-presets
	@mkdir -p "$(VST3_DIR)" "$(CLAP_DIR)"
	@echo "Installing VST3 → $(VST3_DIR)"
	@rm -rf "$(VST3_DIR)/$(FRACTAL).vst3"
	@cp -r "$(BUNDLE)/$(FRACTAL).vst3" "$(VST3_DIR)/"
	@echo "Installing CLAP → $(CLAP_DIR)"
	@rm -rf "$(CLAP_DIR)/$(FRACTAL).clap"
	@cp -r "$(BUNDLE)/$(FRACTAL).clap" "$(CLAP_DIR)/"
	@echo "Done. Restart your DAW to pick up changes."

uninstall-fractal:
	rm -rf "$(VST3_DIR)/$(FRACTAL).vst3"
	rm -rf "$(CLAP_DIR)/$(FRACTAL).clap"
	@echo "Removed fractal plugins. Restart your DAW."

run-fractal: fractal
	cargo run --release --bin fractal-standalone

# --- Reverb ---
reverb:
	cargo xtask bundle $(REVERB) --release

reverb-debug:
	cargo xtask bundle $(REVERB)

install-reverb: reverb export-presets
	@mkdir -p "$(VST3_DIR)" "$(CLAP_DIR)"
	@echo "Installing VST3 → $(VST3_DIR)"
	@rm -rf "$(VST3_DIR)/$(REVERB).vst3"
	@cp -r "$(BUNDLE)/$(REVERB).vst3" "$(VST3_DIR)/"
	@echo "Installing CLAP → $(CLAP_DIR)"
	@rm -rf "$(CLAP_DIR)/$(REVERB).clap"
	@cp -r "$(BUNDLE)/$(REVERB).clap" "$(CLAP_DIR)/"
	@echo "Done. Restart your DAW to pick up changes."

uninstall-reverb:
	rm -rf "$(VST3_DIR)/$(REVERB).vst3"
	rm -rf "$(CLAP_DIR)/$(REVERB).clap"
	@echo "Removed reverb plugins. Restart your DAW."

run-reverb: reverb
	cargo run --release --bin reverb-standalone

# --- All ---
all-plugins: release fractal reverb

install-all: install install-fractal install-reverb export-presets

# --- Universal (fat binary) ---
universal:
	cargo build --release -p lossy-plugin -p fractal-plugin -p reverb-plugin \
		--target aarch64-apple-darwin --target x86_64-apple-darwin
	cargo xtask bundle $(LOSSY) --release --target aarch64-apple-darwin --target x86_64-apple-darwin
	cargo xtask bundle $(FRACTAL) --release --target aarch64-apple-darwin --target x86_64-apple-darwin
	cargo xtask bundle $(REVERB) --release --target aarch64-apple-darwin --target x86_64-apple-darwin
	@for plugin in $(LOSSY) $(FRACTAL) $(REVERB); do \
		lib="lib$$(echo $$plugin | tr '-' '_').dylib"; \
		arm="target/aarch64-apple-darwin/release/$$lib"; \
		x86="target/x86_64-apple-darwin/release/$$lib"; \
		lipo -create "$$arm" "$$x86" -output "$(BUNDLE)/$$plugin.vst3/Contents/MacOS/$$plugin"; \
		lipo -create "$$arm" "$$x86" -output "$(BUNDLE)/$$plugin.clap/Contents/MacOS/$$plugin"; \
		echo "Universal: $$plugin"; \
	done

install-universal: universal export-presets
	@mkdir -p "$(VST3_DIR)" "$(CLAP_DIR)"
	@for plugin in $(LOSSY) $(FRACTAL) $(REVERB); do \
		rm -rf "$(VST3_DIR)/$$plugin.vst3"; \
		cp -r "$(BUNDLE)/$$plugin.vst3" "$(VST3_DIR)/"; \
		rm -rf "$(CLAP_DIR)/$$plugin.clap"; \
		cp -r "$(BUNDLE)/$$plugin.clap" "$(CLAP_DIR)/"; \
		echo "Installed $$plugin"; \
	done
	@echo "Done. Restart your DAW to pick up changes."

# --- Presets ---
export-presets:
	cargo run -p preset-export --release

clean:
	cargo clean
