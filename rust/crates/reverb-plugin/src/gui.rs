//! egui-based GUI for the Reverb plugin — deep blue/indigo theme.

use crate::params::ReverbPluginParams;
use crate::presets::{self, Preset};
use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, egui, widgets};
use std::sync::Arc;

// --- Deep blue color palette ---
const BG_DARK: egui::Color32 = egui::Color32::from_rgb(8, 10, 20);
const BG_SECTION: egui::Color32 = egui::Color32::from_rgb(16, 20, 36);
const BG_SLIDER: egui::Color32 = egui::Color32::from_rgb(20, 24, 42);
const BLUE_PRIMARY: egui::Color32 = egui::Color32::from_rgb(70, 150, 255);
const BLUE_DIM: egui::Color32 = egui::Color32::from_rgb(30, 65, 120);
const BLUE_ACCENT: egui::Color32 = egui::Color32::from_rgb(100, 180, 255);
const BLUE_FAINT: egui::Color32 = egui::Color32::from_rgb(8, 14, 30);
const SCANLINE: egui::Color32 = egui::Color32::from_rgba_premultiplied(0, 0, 0, 18);

/// State persisted across GUI frames.
struct GuiState {
    presets: Vec<Preset>,
    selected: usize,
    loaded: bool,
}

pub fn create(params: Arc<ReverbPluginParams>) -> Option<Box<dyn Editor>> {
    create_egui_editor(
        params.editor_state.clone(),
        GuiState {
            presets: Vec::new(),
            selected: 0,
            loaded: false,
        },
        // --- Build closure: theme setup ---
        |egui_ctx, _state| {
            let mut style = (*egui_ctx.style()).clone();

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

            let v = &mut style.visuals;
            v.dark_mode = true;
            v.override_text_color = Some(BLUE_PRIMARY);
            v.panel_fill = BG_DARK;
            v.collapsing_header_frame = true;
            v.selection.bg_fill = egui::Color32::from_rgba_premultiplied(8, 16, 40, 40);

            let zero = egui::CornerRadius::ZERO;

            v.widgets.inactive.bg_fill = BG_SLIDER;
            v.widgets.inactive.weak_bg_fill = BG_SECTION;
            v.widgets.inactive.corner_radius = zero;

            v.widgets.active.bg_fill = egui::Color32::from_rgb(15, 30, 70);
            v.widgets.active.corner_radius = zero;

            v.widgets.hovered.bg_fill = egui::Color32::from_rgb(25, 35, 55);
            v.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(20, 28, 48);
            v.widgets.hovered.corner_radius = zero;
            v.widgets.hovered.expansion = 1.0;

            v.widgets.noninteractive.bg_fill = BG_SECTION;
            v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BLUE_FAINT);
            v.widgets.noninteractive.corner_radius = zero;

            v.widgets.open.bg_fill = BG_SECTION;
            v.widgets.open.corner_radius = zero;

            v.window_corner_radius = zero;
            v.menu_corner_radius = zero;

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
                    // Header
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("R E V E R B")
                                .heading()
                                .color(BLUE_ACCENT)
                                .strong(),
                        );
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.label(
                                    egui::RichText::new("FDN algorithmic reverb")
                                        .small()
                                        .color(BLUE_DIM),
                                );
                            },
                        );
                    });
                    ui.separator();

                    // Preset Browser
                    if !state.presets.is_empty() {
                        ui.horizontal(|ui| {
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
                                                    .color(BLUE_ACCENT)
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

                        if state.selected > 0 {
                            let desc = &state.presets[state.selected - 1].description;
                            if !desc.is_empty() {
                                ui.label(
                                    egui::RichText::new(desc)
                                        .italics()
                                        .small()
                                        .color(BLUE_DIM),
                                );
                            }
                        }
                        ui.separator();
                    }

                    // Top-level params (always visible)
                    ui.add(widgets::ParamSlider::for_param(&params.feedback_gain, setter));
                    ui.add(widgets::ParamSlider::for_param(&params.wet_dry, setter));
                    ui.separator();

                    // GLOBAL section
                    egui::CollapsingHeader::new(
                        egui::RichText::new("GLOBAL").color(BLUE_ACCENT).strong(),
                    )
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.diffusion, setter));
                        ui.add(widgets::ParamSlider::for_param(
                            &params.diffusion_stages,
                            setter,
                        ));
                        ui.add(widgets::ParamSlider::for_param(&params.saturation, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.pre_delay, setter));
                        ui.add(widgets::ParamSlider::for_param(
                            &params.stereo_width,
                            setter,
                        ));
                    });

                    // DELAY TIMES section
                    egui::CollapsingHeader::new(
                        egui::RichText::new("DELAY TIMES").color(BLUE_ACCENT).strong(),
                    )
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.delay_time_1, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.delay_time_2, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.delay_time_3, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.delay_time_4, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.delay_time_5, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.delay_time_6, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.delay_time_7, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.delay_time_8, setter));
                    });

                    // DAMPING section
                    egui::CollapsingHeader::new(
                        egui::RichText::new("DAMPING").color(BLUE_ACCENT).strong(),
                    )
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.add(widgets::ParamSlider::for_param(&params.damping_1, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.damping_2, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.damping_3, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.damping_4, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.damping_5, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.damping_6, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.damping_7, setter));
                        ui.add(widgets::ParamSlider::for_param(&params.damping_8, setter));
                    });

                    // Two-column grid for smaller sections
                    ui.columns(2, |columns| {
                        // --- Left column ---
                        egui::CollapsingHeader::new(
                            egui::RichText::new("MATRIX").color(BLUE_ACCENT).strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[0], |ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.matrix_type,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.matrix_seed,
                                setter,
                            ));
                        });

                        egui::CollapsingHeader::new(
                            egui::RichText::new("INPUT GAINS")
                                .color(BLUE_ACCENT)
                                .strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[0], |ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.input_gain_1,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.input_gain_2,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.input_gain_3,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.input_gain_4,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.input_gain_5,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.input_gain_6,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.input_gain_7,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.input_gain_8,
                                setter,
                            ));
                        });

                        egui::CollapsingHeader::new(
                            egui::RichText::new("NODE PANS")
                                .color(BLUE_ACCENT)
                                .strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[0], |ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.node_pan_1,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.node_pan_2,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.node_pan_3,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.node_pan_4,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.node_pan_5,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.node_pan_6,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.node_pan_7,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.node_pan_8,
                                setter,
                            ));
                        });

                        // --- Right column ---
                        egui::CollapsingHeader::new(
                            egui::RichText::new("OUTPUT GAINS")
                                .color(BLUE_ACCENT)
                                .strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[1], |ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.output_gain_1,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.output_gain_2,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.output_gain_3,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.output_gain_4,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.output_gain_5,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.output_gain_6,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.output_gain_7,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.output_gain_8,
                                setter,
                            ));
                        });

                        egui::CollapsingHeader::new(
                            egui::RichText::new("MODULATION")
                                .color(BLUE_ACCENT)
                                .strong(),
                        )
                        .default_open(false)
                        .show(&mut columns[1], |ui| {
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_master_rate,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_waveform,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_correlation,
                                setter,
                            ));
                            ui.separator();
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_depth_delay,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_depth_damping,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_depth_output,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_depth_matrix,
                                setter,
                            ));
                            ui.separator();
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_rate_scale_delay,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_rate_scale_damping,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_rate_scale_output,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_rate_matrix,
                                setter,
                            ));
                            ui.separator();
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_matrix2_type,
                                setter,
                            ));
                            ui.add(widgets::ParamSlider::for_param(
                                &params.mod_matrix2_seed,
                                setter,
                            ));
                        });
                    });
                });
            });

            // Scan-line overlay
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
