# Lossy VST Plugin — build helpers
# Usage:
#   make          — release build + bundle (VST3 + CLAP)
#   make install  — copy plugins to system plugin dirs (macOS)
#   make run      — launch standalone
#   make debug    — debug build + bundle
#   make clean    — cargo clean

BUNDLE  := target/bundled
PLUGIN  := lossy-plugin

# macOS plugin install dirs
VST3_DIR := $(HOME)/Library/Audio/Plug-Ins/VST3
CLAP_DIR := $(HOME)/Library/Audio/Plug-Ins/CLAP

.PHONY: all release debug install uninstall run clean

all: release

release:
	cargo xtask bundle $(PLUGIN) --release

debug:
	cargo xtask bundle $(PLUGIN)

install: release
	@mkdir -p "$(VST3_DIR)" "$(CLAP_DIR)"
	@echo "Installing VST3 → $(VST3_DIR)"
	@rm -rf "$(VST3_DIR)/$(PLUGIN).vst3"
	@cp -r "$(BUNDLE)/$(PLUGIN).vst3" "$(VST3_DIR)/"
	@echo "Installing CLAP → $(CLAP_DIR)"
	@rm -rf "$(CLAP_DIR)/$(PLUGIN).clap"
	@cp -r "$(BUNDLE)/$(PLUGIN).clap" "$(CLAP_DIR)/"
	@echo "Done. Restart your DAW to pick up changes."

uninstall:
	rm -rf "$(VST3_DIR)/$(PLUGIN).vst3"
	rm -rf "$(CLAP_DIR)/$(PLUGIN).clap"
	@echo "Removed plugins. Restart your DAW."

run: debug
	cargo run --bin lossy-standalone

clean:
	cargo clean
