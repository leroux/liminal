//! egui-based GUI for the Lossy plugin.

use crate::params::LossyPluginParams;
use crate::presets::{self, Preset};
use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, egui, widgets};
use std::sync::Arc;

/// State persisted across GUI frames.
struct GuiState {
    presets: Vec<Preset>,
    selected: usize, // 0 = "(no preset)", 1.. = preset index + 1
    loaded: bool,
}

pub fn create(params: Arc<LossyPluginParams>) -> Option<Box<dyn Editor>> {
    create_egui_editor(
        params.editor_state.clone(),
        GuiState {
            presets: Vec::new(),
            selected: 0,
            loaded: false,
        },
        |_, _| {},
        move |egui_ctx, setter, state| {
            // Lazy-load presets on first frame
            if !state.loaded {
                // Try filesystem first, fall back to compile-time embedded presets
                if let Some(dir) = presets::find_preset_dir() {
                    state.presets = presets::load_presets(&dir);
                    nih_plug::nih_log!("Loaded {} presets from {}", state.presets.len(), dir.display());
                }
                if state.presets.is_empty() {
                    state.presets = presets::load_embedded_presets();
                    nih_plug::nih_log!("Loaded {} embedded presets", state.presets.len());
                }
                state.loaded = true;
            }

            egui::CentralPanel::default().show(egui_ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Lossy");
                    ui.separator();

                    // --- Preset Browser ---
                    if !state.presets.is_empty() {
                        ui.horizontal(|ui| {
                            ui.label("Preset:");
                            let current_label = if state.selected == 0 {
                                "(no preset)".to_string()
                            } else {
                                let p = &state.presets[state.selected - 1];
                                if p.category.is_empty() || p.category == "Uncategorized" {
                                    p.name.clone()
                                } else {
                                    format!("[{}] {}", p.category, p.name)
                                }
                            };

                            egui::ComboBox::from_id_salt("preset_selector")
                                .selected_text(&current_label)
                                .width(350.0)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(state.selected == 0, "(no preset)").clicked() {
                                        state.selected = 0;
                                    }
                                    let mut last_cat = String::new();
                                    for (i, preset) in state.presets.iter().enumerate() {
                                        if preset.category != last_cat {
                                            ui.separator();
                                            ui.label(
                                                egui::RichText::new(&preset.category).strong(),
                                            );
                                            last_cat = preset.category.clone();
                                        }
                                        let label = &preset.name;
                                        if ui
                                            .selectable_label(state.selected == i + 1, label)
                                            .clicked()
                                        {
                                            state.selected = i + 1;
                                            presets::apply_preset(preset, &params, setter);
                                        }
                                    }
                                });
                        });

                        // Show description if a preset is selected
                        if state.selected > 0 {
                            let desc = &state.presets[state.selected - 1].description;
                            if !desc.is_empty() {
                                ui.label(
                                    egui::RichText::new(desc).italics().weak(),
                                );
                            }
                        }
                        ui.separator();
                    }

                    // --- Spectral Loss ---
                    ui.collapsing("Spectral Loss", |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.loss, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.mode, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.jitter, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.window_size, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.hop_divisor, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.n_bands, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.global_amount, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.phase_loss, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.quantizer, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.pre_echo, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.noise_shape, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.weighting, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.hf_threshold, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.transient_ratio, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.slushy_rate, setter));
                    });

                    // --- Crush ---
                    ui.collapsing("Crush", |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.crush, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.decimate, setter));
                    });

                    // --- Packets ---
                    ui.collapsing("Packets", |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.packets, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.packet_rate, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.packet_size, setter));
                    });

                    // --- Filter ---
                    ui.collapsing("Filter", |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.filter_type, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.filter_freq, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.filter_width, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.filter_slope, setter));
                    });

                    // --- Reverb ---
                    ui.collapsing("Reverb", |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.verb, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.decay, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.verb_position, setter));
                    });

                    // --- Freeze ---
                    ui.collapsing("Freeze", |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.freeze, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.freeze_mode, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.freezer, setter));
                    });

                    // --- Gate & Limiter ---
                    ui.collapsing("Gate / Limiter", |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.gate, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.threshold, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.auto_gain, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.loss_gain, setter));
                    });

                    // --- Output ---
                    ui.separator();
                    ui.add(widgets::ParamSlider::for_param(&params.wet_dry, setter));
                });
            });
        },
    )
}
