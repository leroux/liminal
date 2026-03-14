//! Vizia GUI for the Fractal plugin.

use crate::params::FractalPluginParams;
use crate::presets::{self, Preset};
use pedal_chat::{ChatBackend, ChatMsg};
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaTheming};
use std::sync::Arc;

#[derive(Clone, Lens)]
struct GuiData {
    params: Arc<FractalPluginParams>,
    presets: Vec<Preset>,
    selected: usize,
    chat_input: String,
    chat_messages: Vec<(String, String)>,
    chat_busy: bool,
}

impl nih_plug_vizia::vizia::binding::Data for Preset {
    fn same(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Model for GuiData {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|e, _| match e {
            GuiEvent::SelectPreset(idx) => {
                self.selected = *idx;
                if *idx > 0 {
                    if let Some(preset) = self.presets.get(*idx - 1) {
                        apply_preset_vizia(cx, &self.params, &preset.params);
                    }
                }
            }
            GuiEvent::SetChatInput(text) => {
                self.chat_input = text.clone();
            }
            GuiEvent::SendChat => {
                if self.chat_input.trim().is_empty() || self.chat_busy {
                    return;
                }
                let text = self.chat_input.clone();
                self.chat_messages.push(("user".into(), text.clone()));
                self.chat_input.clear();
                self.chat_busy = true;
                CHAT_BACKEND.with(|cell| {
                    let mut backend = cell.borrow_mut();
                    if backend.is_none() {
                        *backend = Some(ChatBackend::new(SYSTEM_PROMPT));
                    }
                    if let Some(b) = backend.as_ref() {
                        b.send(&text);
                    }
                });
            }
            GuiEvent::PollChat => {
                CHAT_BACKEND.with(|cell| {
                    if let Some(b) = cell.borrow().as_ref() {
                        for msg in b.poll() {
                            match msg {
                                ChatMsg::Text(text) => {
                                    if let Some(last) = self.chat_messages.last_mut() {
                                        if last.0 == "assistant" {
                                            last.1 = text;
                                            return;
                                        }
                                    }
                                    self.chat_messages.push(("assistant".into(), text));
                                }
                                ChatMsg::Error(e) => {
                                    self.chat_messages.push(("error".into(), e));
                                    self.chat_busy = false;
                                }
                                ChatMsg::Done => {
                                    self.chat_busy = false;
                                }
                            }
                        }
                    }
                });
            }
        });
    }
}

#[derive(Debug, Clone)]
enum GuiEvent {
    SelectPreset(usize),
    SetChatInput(String),
    SendChat,
    PollChat,
}

thread_local! {
    static CHAT_BACKEND: std::cell::RefCell<Option<ChatBackend>> = const { std::cell::RefCell::new(None) };
}

const SYSTEM_PROMPT: &str = "You are an audio effects tuning assistant for a fractal audio processor plugin. \
    Help the user achieve their desired fractal/spectral sound. Give concise advice about parameter adjustments. \
    Keep responses short and focused on audio production.";

// Amber theme — overrides nih-plug's light default
const THEME_CSS: &str = r#"
:root {
    background-color: rgb(18, 14, 6);
    color: rgb(220, 195, 140);
    font-size: 13;
}
label {
    color: rgb(220, 195, 140);
}
textbox {
    background-color: rgb(34, 28, 14);
    color: rgb(220, 195, 140);
    border-color: rgb(120, 80, 20);
    border-radius: 2px;
}
button {
    background-color: rgb(32, 26, 12);
    color: rgb(220, 195, 140);
    border-color: rgb(120, 80, 20);
    border-radius: 2px;
}
button:hover {
    background-color: rgb(45, 36, 18);
}
param-slider {
    border-color: rgb(120, 80, 20);
}
param-slider .fill {
    background-color: rgb(200, 140, 30);
}
param-slider .value-entry {
    color: rgb(220, 195, 140);
}
param-slider .value-entry .caret {
    background-color: rgb(255, 200, 80);
}
param-button {
    border-color: rgb(120, 80, 20);
    color: rgb(220, 195, 140);
}
param-button:checked {
    background-color: rgb(200, 140, 30);
}
scrollview scrollbar {
    background-color: rgb(18, 14, 6);
}
scrollview scrollbar .thumb {
    background-color: rgb(120, 80, 20);
}
dropdown popup {
    background-color: rgb(28, 22, 10);
    border-color: rgb(120, 80, 20);
}
"#;

pub fn create(params: Arc<FractalPluginParams>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(
        params.editor_state.clone(),
        ViziaTheming::Custom,
        move |cx, _| {
            assets::register_noto_sans_light(cx);
            cx.add_stylesheet(THEME_CSS).ok();

            let mut all_presets = Vec::new();
            if let Some(dir) = presets::find_preset_dir() {
                all_presets = presets::load_presets(&dir);
            }
            if all_presets.is_empty() {
                all_presets = presets::load_embedded_presets();
            }

            GuiData {
                params: params.clone(),
                presets: all_presets,
                selected: 0,
                chat_input: String::new(),
                chat_messages: Vec::new(),
                chat_busy: false,
            }
            .build(cx);

            cx.spawn(|proxy| loop {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if proxy.emit(GuiEvent::PollChat).is_err() {
                    break;
                }
            });

            HStack::new(cx, |cx| {
                // ── Left column: title, preset, param sections ──
                VStack::new(cx, |cx| {
                    // Title + preset row
                    HStack::new(cx, |cx| {
                        Label::new(cx, "Fractal")
                            .font_family(vec![FamilyOwned::Name(String::from(
                                assets::NOTO_SANS,
                            ))])
                            .font_weight(FontWeightKeyword::Thin)
                            .font_size(30.0)
                            .width(Auto);

                        Dropdown::new(
                            cx,
                            |cx| {
                                Label::new(
                                    cx,
                                    GuiData::root.map(|d: &GuiData| {
                                        if d.selected == 0 {
                                            "(init)".to_string()
                                        } else if d.selected <= d.presets.len() {
                                            d.presets[d.selected - 1].name.clone()
                                        } else {
                                            String::new()
                                        }
                                    }),
                                )
                            },
                            |cx| {
                                ScrollView::new(cx, 0.0, 0.0, false, true, |cx| {
                                    Label::new(cx, "(init)")
                                        .width(Stretch(1.0))
                                        .cursor(CursorIcon::Hand)
                                        .on_press(|cx| {
                                            cx.emit(GuiEvent::SelectPreset(0));
                                            cx.emit(PopupEvent::Close);
                                        });
                                    Binding::new(
                                        cx,
                                        GuiData::presets,
                                        |cx, presets_lens| {
                                            let presets = presets_lens.get(cx);
                                            for (i, preset) in presets.iter().enumerate()
                                            {
                                                let name = preset.name.clone();
                                                let idx = i + 1;
                                                Label::new(cx, &name)
                                                    .width(Stretch(1.0))
                                                    .cursor(CursorIcon::Hand)
                                                    .on_press(move |cx| {
                                                        cx.emit(
                                                            GuiEvent::SelectPreset(idx),
                                                        );
                                                        cx.emit(PopupEvent::Close);
                                                    });
                                            }
                                        },
                                    );
                                })
                                .height(Pixels(200.0));
                            },
                        )
                        .width(Stretch(1.0));
                    })
                    .col_between(Pixels(8.0))
                    .height(Auto)
                    .child_top(Pixels(4.0))
                    .child_bottom(Pixels(4.0))
                    .left(Pixels(8.0))
                    .right(Pixels(8.0));

                    // Scrollable param sections
                    ScrollView::new(cx, 0.0, 0.0, false, true, |cx| {
                        VStack::new(cx, |cx| {
                            // ── Core Fractal ──
                            section(cx, "Core Fractal", |cx| {
                                param_row(cx, "Scales", |p| &p.num_scales);
                                param_row(cx, "Ratio", |p| &p.scale_ratio);
                                param_row(cx, "Decay", |p| &p.amplitude_decay);
                                param_row_enum(cx, "Interp", |p| &p.interp);
                                param_row_bool(cx, "Reverse", |p| &p.reverse_scales);
                                param_row(cx, "Offset", |p| &p.scale_offset);
                            });

                            // ── Iteration / Feedback ──
                            section(cx, "Iteration", |cx| {
                                param_row(cx, "Iterations", |p| &p.iterations);
                                param_row(cx, "Iter Decay", |p| &p.iter_decay);
                                param_row(cx, "Saturation", |p| &p.saturation);
                                param_row(cx, "Feedback", |p| &p.feedback);
                            });

                            // ── Spectral ──
                            section(cx, "Spectral", |cx| {
                                param_row(cx, "Spectral", |p| &p.spectral);
                                param_row(cx, "Window Size", |p| &p.window_size);
                            });

                            // ── Pre-filter ──
                            section(cx, "Pre-Filter", |cx| {
                                param_row_enum(cx, "Type", |p| &p.filter_type);
                                param_row(cx, "Freq", |p| &p.filter_freq);
                                param_row(cx, "Q", |p| &p.filter_q);
                            });

                            // ── Post-filter ──
                            section(cx, "Post-Filter", |cx| {
                                param_row_enum(cx, "Type", |p| &p.post_filter_type);
                                param_row(cx, "Freq", |p| &p.post_filter_freq);
                            });

                            // ── Effects ──
                            section(cx, "Effects", |cx| {
                                param_row(cx, "Gate", |p| &p.gate);
                                param_row(cx, "Crush", |p| &p.crush);
                                param_row(cx, "Decimate", |p| &p.decimate);
                            });

                            // ── Layers ──
                            section(cx, "Layers", |cx| {
                                param_row(cx, "Layer 1", |p| &p.layer_gain_1);
                                param_row(cx, "Layer 2", |p| &p.layer_gain_2);
                                param_row(cx, "Layer 3", |p| &p.layer_gain_3);
                                param_row(cx, "Layer 4", |p| &p.layer_gain_4);
                                param_row(cx, "Layer 5", |p| &p.layer_gain_5);
                                param_row(cx, "Layer 6", |p| &p.layer_gain_6);
                                param_row(cx, "Layer 7", |p| &p.layer_gain_7);
                                param_row_bool(cx, "Only Wet", |p| &p.fractal_only_wet);
                                param_row(cx, "Spread", |p| &p.layer_spread);
                                param_row(cx, "Detune", |p| &p.layer_detune);
                                param_row(cx, "Delay", |p| &p.layer_delay);
                                param_row(cx, "Tilt", |p| &p.layer_tilt);
                            });

                            // ── Output ──
                            section(cx, "Output", |cx| {
                                param_row(cx, "Wet/Dry", |p| &p.wet_dry);
                                param_row(cx, "Output Gain", |p| &p.output_gain);
                                param_row(cx, "Threshold", |p| &p.threshold);
                            });
                        })
                        .row_between(Pixels(16.0))
                        .height(Auto)
                        .width(Stretch(1.0))
                        .child_left(Pixels(8.0))
                        .child_right(Pixels(8.0))
                        .child_top(Pixels(4.0))
                        .child_bottom(Pixels(8.0));
                    })
                    .height(Stretch(1.0));
                })
                .width(Percentage(60.0))
                .height(Stretch(1.0));

                // ── Column separator ──
                Element::new(cx)
                    .width(Pixels(1.0))
                    .height(Stretch(1.0))
                    .background_color(Color::rgb(140, 90, 20));

                // ── Right column: chat panel ──
                VStack::new(cx, |cx| {
                    Label::new(cx, "Chat")
                        .font_size(16.0)
                        .font_weight(FontWeightKeyword::Bold)
                        .height(Auto)
                        .top(Pixels(8.0))
                        .bottom(Pixels(4.0))
                        .left(Pixels(8.0));

                    ScrollView::new(cx, 0.0, 0.0, false, true, |cx| {
                        Binding::new(cx, GuiData::chat_messages, |cx, msgs_lens| {
                            let msgs = msgs_lens.get(cx);
                            for (role, text) in msgs.iter() {
                                let prefix = match role.as_str() {
                                    "user" => "You: ",
                                    "assistant" => "Claude: ",
                                    "error" => "Error: ",
                                    _ => "",
                                };
                                Label::new(cx, &format!("{prefix}{text}"))
                                    .width(Stretch(1.0))
                                    .font_size(12.0)
                                    .left(Pixels(8.0))
                                    .right(Pixels(8.0))
                                    .bottom(Pixels(2.0));
                            }
                        });
                    })
                    .height(Stretch(1.0));

                    HStack::new(cx, |cx| {
                        Textbox::new(cx, GuiData::chat_input)
                            .on_edit(|cx, text| {
                                cx.emit(GuiEvent::SetChatInput(text));
                            })
                            .on_submit(|cx, _, _| {
                                cx.emit(GuiEvent::SendChat);
                            })
                            .width(Stretch(1.0))
                            .height(Pixels(28.0));

                        Button::new(
                            cx,
                            |cx| cx.emit(GuiEvent::SendChat),
                            |cx| Label::new(cx, "Send"),
                        )
                        .width(Pixels(50.0));
                    })
                    .col_between(Pixels(4.0))
                    .height(Auto)
                    .left(Pixels(8.0))
                    .right(Pixels(8.0))
                    .bottom(Pixels(8.0));
                })
                .width(Percentage(40.0))
                .height(Stretch(1.0));
            })
            .width(Stretch(1.0))
            .height(Stretch(1.0))
            .background_color(Color::rgb(18, 14, 6));

            ResizeHandle::new(cx);
        },
    )
}

/// Label + ParamSlider row for continuous parameters.
fn param_row<P, FMap>(cx: &mut Context, label: &str, params_to_param: FMap)
where
    P: Param + 'static,
    FMap: Fn(&Arc<FractalPluginParams>) -> &P + Copy + 'static,
{
    HStack::new(cx, |cx| {
        Label::new(cx, label)
            .width(Pixels(100.0))
            .child_top(Stretch(1.0))
            .child_bottom(Stretch(1.0));
        ParamSlider::new(cx, GuiData::params, params_to_param).width(Stretch(1.0));
    })
    .col_between(Pixels(6.0))
    .height(Auto)
    .width(Stretch(1.0));
}

/// Label + ParamSlider row for enum/stepped parameters.
fn param_row_enum<P, FMap>(cx: &mut Context, label: &str, params_to_param: FMap)
where
    P: Param + 'static,
    FMap: Fn(&Arc<FractalPluginParams>) -> &P + Copy + 'static,
{
    HStack::new(cx, |cx| {
        Label::new(cx, label)
            .width(Pixels(100.0))
            .child_top(Stretch(1.0))
            .child_bottom(Stretch(1.0));
        ParamSlider::new(cx, GuiData::params, params_to_param)
            .set_style(ParamSliderStyle::CurrentStepLabeled { even: true })
            .width(Stretch(1.0));
    })
    .col_between(Pixels(6.0))
    .height(Auto)
    .width(Stretch(1.0));
}

/// Label + ParamButton row for boolean parameters.
fn param_row_bool(cx: &mut Context, label: &str, params_to_param: fn(&Arc<FractalPluginParams>) -> &BoolParam) {
    HStack::new(cx, |cx| {
        Label::new(cx, label)
            .width(Pixels(100.0))
            .child_top(Stretch(1.0))
            .child_bottom(Stretch(1.0));
        ParamButton::new(cx, GuiData::params, params_to_param)
            .width(Stretch(1.0));
    })
    .col_between(Pixels(6.0))
    .height(Auto)
    .width(Stretch(1.0));
}

fn section(cx: &mut Context, title: &str, content: impl FnOnce(&mut Context)) {
    VStack::new(cx, |cx| {
        Label::new(cx, title)
            .font_size(13.0)
            .font_weight(FontWeightKeyword::Bold)
            .color(Color::rgb(255, 210, 100))
            .width(Stretch(1.0))
            .height(Auto)
            .top(Pixels(2.0))
            .bottom(Pixels(4.0));
        Element::new(cx)
            .height(Pixels(1.0))
            .width(Stretch(1.0))
            .background_color(Color::rgb(140, 90, 20))
            .bottom(Pixels(4.0));
        content(cx);
    })
    .row_between(Pixels(2.0))
    .height(Auto)
    .width(Stretch(1.0));
}

fn set_param_f32(cx: &mut EventContext, param: &FloatParam, value: f32) {
    let ptr = param.as_ptr();
    let normalized = param.preview_normalized(value);
    cx.emit(RawParamEvent::BeginSetParameter(ptr));
    cx.emit(RawParamEvent::SetParameterNormalized(ptr, normalized));
    cx.emit(RawParamEvent::EndSetParameter(ptr));
}

fn set_param_i32(cx: &mut EventContext, param: &IntParam, value: i32) {
    let ptr = param.as_ptr();
    let normalized = param.preview_normalized(value);
    cx.emit(RawParamEvent::BeginSetParameter(ptr));
    cx.emit(RawParamEvent::SetParameterNormalized(ptr, normalized));
    cx.emit(RawParamEvent::EndSetParameter(ptr));
}

fn set_param_bool(cx: &mut EventContext, param: &BoolParam, value: bool) {
    let ptr = param.as_ptr();
    let normalized = param.preview_normalized(value);
    cx.emit(RawParamEvent::BeginSetParameter(ptr));
    cx.emit(RawParamEvent::SetParameterNormalized(ptr, normalized));
    cx.emit(RawParamEvent::EndSetParameter(ptr));
}

fn set_param_enum<T: Enum + PartialEq>(cx: &mut EventContext, param: &EnumParam<T>, index: i32) {
    let val = T::from_index(index as usize);
    let ptr = param.as_ptr();
    let normalized = param.preview_normalized(val);
    cx.emit(RawParamEvent::BeginSetParameter(ptr));
    cx.emit(RawParamEvent::SetParameterNormalized(ptr, normalized));
    cx.emit(RawParamEvent::EndSetParameter(ptr));
}

fn apply_preset_vizia(
    cx: &mut EventContext,
    pp: &FractalPluginParams,
    p: &fractal_dsp::FractalParams,
) {
    set_param_i32(cx, &pp.num_scales, p.num_scales);
    set_param_f32(cx, &pp.scale_ratio, p.scale_ratio as f32);
    set_param_f32(cx, &pp.amplitude_decay, p.amplitude_decay as f32);
    set_param_enum(cx, &pp.interp, p.interp);
    set_param_bool(cx, &pp.reverse_scales, p.reverse_scales != 0);
    set_param_f32(cx, &pp.scale_offset, p.scale_offset as f32);
    set_param_i32(cx, &pp.iterations, p.iterations);
    set_param_f32(cx, &pp.iter_decay, p.iter_decay as f32);
    set_param_f32(cx, &pp.saturation, p.saturation as f32);
    set_param_f32(cx, &pp.spectral, p.spectral as f32);
    set_param_i32(cx, &pp.window_size, p.window_size);
    set_param_enum(cx, &pp.filter_type, p.filter_type);
    set_param_f32(cx, &pp.filter_freq, p.filter_freq as f32);
    set_param_f32(cx, &pp.filter_q, p.filter_q as f32);
    set_param_enum(cx, &pp.post_filter_type, p.post_filter_type);
    set_param_f32(cx, &pp.post_filter_freq, p.post_filter_freq as f32);
    set_param_f32(cx, &pp.gate, p.gate as f32);
    set_param_f32(cx, &pp.crush, p.crush as f32);
    set_param_f32(cx, &pp.decimate, p.decimate as f32);
    set_param_f32(cx, &pp.layer_gain_1, p.layer_gain_1 as f32);
    set_param_f32(cx, &pp.layer_gain_2, p.layer_gain_2 as f32);
    set_param_f32(cx, &pp.layer_gain_3, p.layer_gain_3 as f32);
    set_param_f32(cx, &pp.layer_gain_4, p.layer_gain_4 as f32);
    set_param_f32(cx, &pp.layer_gain_5, p.layer_gain_5 as f32);
    set_param_f32(cx, &pp.layer_gain_6, p.layer_gain_6 as f32);
    set_param_f32(cx, &pp.layer_gain_7, p.layer_gain_7 as f32);
    set_param_bool(cx, &pp.fractal_only_wet, p.fractal_only_wet != 0);
    set_param_f32(cx, &pp.layer_spread, p.layer_spread as f32);
    set_param_f32(cx, &pp.layer_detune, p.layer_detune as f32);
    set_param_f32(cx, &pp.layer_delay, p.layer_delay as f32);
    set_param_f32(cx, &pp.layer_tilt, p.layer_tilt as f32);
    set_param_f32(cx, &pp.feedback, p.feedback as f32);
    set_param_f32(cx, &pp.wet_dry, p.wet_dry as f32);
    set_param_f32(cx, &pp.output_gain, p.output_gain as f32);
    set_param_f32(cx, &pp.threshold, p.threshold as f32);
}
