/// Preset browser — loads JSON presets, search, favorites, categories.
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use vizia::prelude::*;

/// A single preset entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PresetEntry {
    pub name: String,
    pub category: String,
    pub description: String,
    pub file_path: PathBuf,
    #[serde(skip)]
    pub params_json: String,
}

impl Data for PresetEntry {
    fn same(&self, other: &Self) -> bool {
        self == other
    }
}

/// Events for the preset browser.
pub enum PresetBrowserEvent {
    LoadPreset(usize),
    NextPreset,
    PrevPreset,
    ToggleFavorite(usize),
    SetSearch(String),
    RefreshPresets,
}

/// Preset browser data model.
#[derive(Debug, Clone, Lens)]
pub struct PresetBrowserData {
    pub presets: Vec<PresetEntry>,
    pub selected: usize,
    pub search_filter: String,
    pub favorites: HashSet<String>,
    pub favorites_path: PathBuf,
    pub preset_dir: PathBuf,
}

impl PresetBrowserData {
    pub fn new(preset_dir: PathBuf) -> Self {
        let favorites_path = preset_dir.join("favorites.json");
        let favorites = load_favorites(&favorites_path);
        let presets = load_presets_from_dir(&preset_dir);

        Self {
            presets,
            selected: 0,
            search_filter: String::new(),
            favorites,
            favorites_path,
            preset_dir,
        }
    }

    pub fn current_preset(&self) -> Option<&PresetEntry> {
        if self.selected > 0 && self.selected <= self.presets.len() {
            Some(&self.presets[self.selected - 1])
        } else {
            None
        }
    }

    fn save_favorites(&self) {
        let json = serde_json::to_string_pretty(
            &self.favorites.iter().collect::<Vec<_>>(),
        )
        .unwrap_or_default();
        let _ = std::fs::write(&self.favorites_path, json);
    }
}

impl Model for PresetBrowserData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|e, _| match e {
            PresetBrowserEvent::LoadPreset(idx) => {
                self.selected = *idx;
                if let Some(preset) = self.current_preset() {
                    let json = preset.params_json.clone();
                    cx.emit(crate::gui::app_shell::AppEvent::PresetLoaded(json));
                }
            }
            PresetBrowserEvent::NextPreset => {
                if self.selected < self.presets.len() {
                    self.selected += 1;
                    if let Some(preset) = self.current_preset() {
                        let json = preset.params_json.clone();
                        cx.emit(crate::gui::app_shell::AppEvent::PresetLoaded(json));
                    }
                }
            }
            PresetBrowserEvent::PrevPreset => {
                if self.selected > 1 {
                    self.selected -= 1;
                    if let Some(preset) = self.current_preset() {
                        let json = preset.params_json.clone();
                        cx.emit(crate::gui::app_shell::AppEvent::PresetLoaded(json));
                    }
                }
            }
            PresetBrowserEvent::ToggleFavorite(idx) => {
                if let Some(preset) = self.presets.get(*idx) {
                    let name = preset.name.clone();
                    if self.favorites.contains(&name) {
                        self.favorites.remove(&name);
                    } else {
                        self.favorites.insert(name);
                    }
                    self.save_favorites();
                }
            }
            PresetBrowserEvent::SetSearch(query) => {
                self.search_filter = query.clone();
            }
            PresetBrowserEvent::RefreshPresets => {
                self.presets = load_presets_from_dir(&self.preset_dir);
            }
        });
    }
}

/// Load all .json presets from a directory, sorted by category then name.
pub fn load_presets_from_dir(dir: &Path) -> Vec<PresetEntry> {
    let mut presets = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return presets,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            if let Ok(content) = std::fs::read_to_string(&path) {
                let name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                let (category, description) = if let Ok(val) =
                    serde_json::from_str::<serde_json::Value>(&content)
                {
                    let meta = val.get("_meta");
                    let cat = meta
                        .and_then(|m| m.get("category"))
                        .and_then(|c| c.as_str())
                        .unwrap_or("Uncategorized")
                        .to_string();
                    let desc = meta
                        .and_then(|m| m.get("description"))
                        .and_then(|d| d.as_str())
                        .unwrap_or("")
                        .to_string();
                    (cat, desc)
                } else {
                    ("Uncategorized".to_string(), String::new())
                };

                presets.push(PresetEntry {
                    name,
                    category,
                    description,
                    file_path: path,
                    params_json: content,
                });
            }
        }
    }

    presets.sort_by(|a, b| a.category.cmp(&b.category).then(a.name.cmp(&b.name)));
    presets
}

fn load_favorites(path: &Path) -> HashSet<String> {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str::<Vec<String>>(&content)
            .unwrap_or_default()
            .into_iter()
            .collect(),
        Err(_) => HashSet::new(),
    }
}

/// Build the preset browser view.
pub fn preset_browser_view(cx: &mut Context) {
    VStack::new(cx, |cx| {
        // Search bar
        HStack::new(cx, |cx| {
            Label::new(cx, "Search:");
            Textbox::new(cx, PresetBrowserData::search_filter)
                .on_edit(|cx, text| cx.emit(PresetBrowserEvent::SetSearch(text)));
        })
        .height(Auto)
        .horizontal_gap(Pixels(8.0));

        // Nav buttons
        HStack::new(cx, |cx| {
            Button::new(cx, |cx| Label::new(cx, "<"))
                .on_press(|cx| cx.emit(PresetBrowserEvent::PrevPreset));
            Label::new(
                cx,
                PresetBrowserData::selected.map(|s| {
                    if *s == 0 {
                        "(no preset)".to_string()
                    } else {
                        format!("Preset {s}")
                    }
                }),
            );
            Button::new(cx, |cx| Label::new(cx, ">"))
                .on_press(|cx| cx.emit(PresetBrowserEvent::NextPreset));
        })
        .height(Auto)
        .horizontal_gap(Pixels(4.0));

        // Preset list
        ScrollView::new(cx, |cx| {
            List::new(cx, PresetBrowserData::presets, |cx, idx, item| {
                let i = idx;
                HStack::new(cx, move |cx| {
                    Label::new(cx, item.map(|p| format!("[{}]", p.category)))
                        .class("preset-category")
                        .width(Pixels(120.0));
                    Label::new(cx, item.map(|p| p.name.clone()))
                        .class("preset-name");
                })
                .height(Pixels(24.0))
                .on_press(move |cx| cx.emit(PresetBrowserEvent::LoadPreset(i + 1)));
            });
        });
    })
    .vertical_gap(Pixels(4.0));
}
