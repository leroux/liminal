//! egui-based GUI for the Lossy plugin — CRT phosphor theme.

use crate::params::LossyPluginParams;
use crate::presets::{self, Preset};
use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, egui, widgets};
use std::sync::Arc;

// --- CRT phosphor color palette ---
const BG_DARK: egui::Color32 = egui::Color32::from_rgb(10, 15, 10);
const BG_SECTION: egui::Color32 = egui::Color32::from_rgb(22, 30, 22);
const BG_SLIDER: egui::Color32 = egui::Color32::from_rgb(26, 36, 26);
const GREEN_PRIMARY: egui::Color32 = egui::Color32::from_rgb(0, 221, 85);
const GREEN_DIM: egui::Color32 = egui::Color32::from_rgb(0, 102, 48);
const GREEN_ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 255, 102);
const GREEN_FAINT: egui::Color32 = egui::Color32::from_rgb(0, 40, 16);
const SCANLINE: egui::Color32 = egui::Color32::from_rgba_premultiplied(0, 0, 0, 18);

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
        // --- Build closure: theme setup (runs once per editor open) ---
        |egui_ctx, _state| {
            let mut style = (*egui_ctx.style()).clone();

            // Monospace everywhere
            let mono = |size: f32| egui::FontId::new(size, egui::FontFamily::Monospace);
            style.text_styles.insert(egui::TextStyle::Body, mono(13.0));
            style
                .text_styles
                .insert(egui::TextStyle::Button, mono(13.0));
            style
                .text_styles
                .insert(egui::TextStyle::Heading, mono(18.0));
            style
                .text_styles
                .insert(egui::TextStyle::Small, mono(11.0));
            style
                .text_styles
                .insert(egui::TextStyle::Monospace, mono(13.0));

            // Visuals overrides
            let v = &mut style.visuals;
            v.dark_mode = true;
            v.override_text_color = Some(GREEN_PRIMARY);
            v.panel_fill = BG_DARK;
            v.collapsing_header_frame = true;
            v.selection.bg_fill = egui::Color32::from_rgba_premultiplied(0, 28, 11, 40);

            // Widget states — sharp corners, CRT fills
            let zero = egui::CornerRadius::ZERO;

            v.widgets.inactive.bg_fill = BG_SLIDER;
            v.widgets.inactive.weak_bg_fill = BG_SECTION;
            v.widgets.inactive.corner_radius = zero;

            v.widgets.active.bg_fill = egui::Color32::from_rgb(0, 60, 25);
            v.widgets.active.corner_radius = zero;

            v.widgets.hovered.bg_fill = egui::Color32::from_rgb(30, 45, 30);
            v.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(25, 38, 25);
            v.widgets.hovered.corner_radius = zero;
            v.widgets.hovered.expansion = 1.0;

            v.widgets.noninteractive.bg_fill = BG_SECTION;
            v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, GREEN_FAINT);
            v.widgets.noninteractive.corner_radius = zero;

            v.widgets.open.bg_fill = BG_SECTION;
            v.widgets.open.corner_radius = zero;

            v.window_corner_radius = zero;
            v.menu_corner_radius = zero;

            // Spacing
            style.spacing.item_spacing = egui::vec2(8.0, 4.0);
            style.spacing.slider_width = 200.0;

            egui_ctx.set_style(style);
        },
        // --- Update closure: runs every frame ---
        move |egui_ctx, setter, state| {
            // Lazy-load presets on first frame
            if !state.loaded {
                if let Some(dir) = presets::find_preset_dir() {
                    state.presets = presets::load_presets(&dir);
                    nih_plug::nih_log!(
                        "Loaded {} presets from {}",
                        state.presets.len(),
                        dir.display()
                    );
                }
                if state.presets.is_empty() {
                    state.presets = presets::load_embedded_presets();
                    nih_plug::nih_log!("Loaded {} embedded presets", state.presets.len());
                }
                state.loaded = true;
            }

            egui::CentralPanel::default().show(egui_ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // ═══ Header ═══
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("L O S S Y")
                                .heading()
                                .color(GREEN_ACCENT)
                                .strong(),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.label(
                                    egui::RichText::new("codec emulator")
                                        .small()
                                        .color(GREEN_DIM),
                                );
                            },
                        );
                    });
                    ui.separator();

                    // ═══ Preset Browser ═══
                    if !state.presets.is_empty() {
                        ui.horizontal(|ui| {
                            // Prev button
                            if ui
                                .add_enabled(state.selected > 1, egui::Button::new("\u{25C0}"))
                                .clicked()
                            {
                                state.selected -= 1;
                                presets::apply_preset(
                                    &state.presets[state.selected - 1],
                                    &params,
                                    setter,
                                );
                            }

                            // Preset ComboBox
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
                                .width(ui.available_width() - 32.0)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(state.selected == 0, "(no preset)")
                                        .clicked()
                                    {
                                        state.selected = 0;
                                    }
                                    let mut last_cat = String::new();
                                    for (i, preset) in state.presets.iter().enumerate() {
                                        if preset.category != last_cat {
                                            ui.separator();
                                            ui.label(
                                                egui::RichText::new(&preset.category)
                                                    .color(GREEN_ACCENT)
                                                    .strong(),
                                            );
                                            last_cat = preset.category.clone();
                                        }
                                        if ui
                                            .selectable_label(
                                                state.selected == i + 1,
                                                &preset.name,
                                            )
                                            .clicked()
                                        {
                                            state.selected = i + 1;
                                            presets::apply_preset(preset, &params, setter);
                                        }
                                    }
                                });

                            // Next button
                            if ui
                                .add_enabled(
                                    state.selected < state.presets.len(),
                                    egui::Button::new("\u{25B6}"),
                                )
                                .clicked()
                            {
                                state.selected += 1;
                                presets::apply_preset(
                                    &state.presets[state.selected - 1],
                                    &params,
                                    setter,
                                );
                            }
                        });

                        // Preset description
                        if state.selected > 0 {
                            let desc = &state.presets[state.selected - 1].description;
                            if !desc.is_empty() {
                                ui.label(
                                    egui::RichText::new(desc)
                                        .italics()
                                        .small()
                                        .color(GREEN_DIM),
                                );
                            }
                        }
                        ui.separator();
                    }

                    // ═══ Top-level params (always visible) ═══
                    ui.add(widgets::ParamSlider::for_param(&params.loss, setter));
                    ui.add(widgets::ParamSlider::for_param(&params.wet_dry, setter));
                    ui.separator();

                    // ═══ SPECTRAL (full width, 14 params) ═══
                    egui::CollapsingHeader::new(
                        egui::RichText::new("SPECTRAL").color(GREEN_ACCENT).strong(),
                    )
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.mode, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.jitter, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.window_size, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.hop_divisor, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.n_bands, setter));
                        ui.add(widgets::ParamSlider::for_param(
                            &params.global_amount,
                            setter,
                        ));
                        ui.add(widgets::ParamSlider::for_param(&params.phase_loss, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.quantizer, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.pre_echo, setter));
                        ui.add(widgets::ParamSlider::for_param(
                            &params.noise_shape,
                            setter,
                        ));
                        ui.add(widgets::ParamSlider::for_param(&params.weighting, setter));
                        ui.add(widgets::ParamSlider::for_param(
                            &params.hf_threshold,
                            setter,
                        ));
                        ui.add(widgets::ParamSlider::for_param(
                            &params.transient_ratio,
                            setter,
                        ));
                        ui.add(widgets::ParamSlider::for_param(
                            &params.slushy_rate,
                            setter,
                        ));
                    });

                    // ═══ Two-column grid for smaller sections ═══
                    ui.columns(2, |columns| {
                        // --- Left column ---
                        egui::CollapsingHeader::new(
                            egui::RichText::new("CRUSH").color(GREEN_ACCENT).strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[0], |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.crush, setter));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.decimate,
                                setter,
                            ));
                        });

                        egui::CollapsingHeader::new(
                            egui::RichText::new("FILTER").color(GREEN_ACCENT).strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[0], |ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.filter_type,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.filter_freq,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.filter_width,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.filter_slope,
                                setter,
                            ));
                        });

                        egui::CollapsingHeader::new(
                            egui::RichText::new("FREEZE").color(GREEN_ACCENT).strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[0], |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.freeze, setter));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.freeze_mode,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(&params.freezer, setter));
                        });

                        // --- Right column ---
                        egui::CollapsingHeader::new(
                            egui::RichText::new("PACKETS").color(GREEN_ACCENT).strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[1], |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.packets, setter));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.packet_rate,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.packet_size,
                                setter,
                            ));
                        });

                        egui::CollapsingHeader::new(
                            egui::RichText::new("REVERB").color(GREEN_ACCENT).strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[1], |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.verb, setter));
                            ui.add(widgets::ParamSlider::for_param(&params.decay, setter));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.verb_position,
                                setter,
                            ));
                        });

                        egui::CollapsingHeader::new(
                            egui::RichText::new("GATE / LIMITER")
                                .color(GREEN_ACCENT)
                                .strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[1], |ui| {
                            ui.add(widgets::ParamSlider::for_param(&params.gate, setter));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.threshold,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.auto_gain,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.loss_gain,
                                setter,
                            ));
                        });
                    });
                });
            });

            // ═══ Scan-line overlay ═══
            let painter = egui_ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("scanlines"),
            ));
            let rect = egui_ctx.screen_rect();
            let mut y = rect.top();
            while y < rect.bottom() {
                painter.hline(rect.x_range(), y, egui::Stroke::new(1.0, SCANLINE));
                y += 3.0;
            }
        },
    )
}
