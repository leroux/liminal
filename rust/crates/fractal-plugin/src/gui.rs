//! Vizia GUI for the Fractal plugin.

use crate::params::FractalPluginParams;
use crate::presets::{self, Preset};
use claudewire::chat::{ChatBackend, ChatMsg};
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
                                ChatMsg::AssistantText(text) => {
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

pub fn create(params: Arc<FractalPluginParams>) -> Option<Box<dyn Editor>> {
    create_vizia_editor(
        params.editor_state.clone(),
        ViziaTheming::Custom,
        move |cx, _| {
            assets::register_noto_sans_light(cx);

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

            VStack::new(cx, |cx| {
                Label::new(cx, "Fractal")
                    .font_family(vec![FamilyOwned::Name(String::from(assets::NOTO_SANS))])
                    .font_weight(FontWeightKeyword::Thin)
                    .font_size(30.0)
                    .height(Pixels(42.0))
                    .child_top(Stretch(1.0))
                    .child_bottom(Pixels(0.0));

                HStack::new(cx, |cx| {
                    Label::new(cx, "Preset:").width(Auto);
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
                                Binding::new(cx, GuiData::presets, |cx, presets_lens| {
                                    let presets = presets_lens.get(cx);
                                    for (i, preset) in presets.iter().enumerate() {
                                        let name = preset.name.clone();
                                        let idx = i + 1;
                                        Label::new(cx, &name)
                                            .width(Stretch(1.0))
                                            .cursor(CursorIcon::Hand)
                                            .on_press(move |cx| {
                                                cx.emit(GuiEvent::SelectPreset(idx));
                                                cx.emit(PopupEvent::Close);
                                            });
                                    }
                                });
                            })
                            .height(Pixels(200.0));
                        },
                    )
                    .width(Stretch(1.0));
                })
                .col_between(Pixels(4.0))
                .height(Auto);

                ScrollView::new(cx, 0.0, 0.0, false, true, |cx| {
                    GenericUi::new(cx, GuiData::params);
                })
                .width(Percentage(100.0));

                VStack::new(cx, |cx| {
                    Label::new(cx, "Chat")
                        .font_size(14.0)
                        .font_weight(FontWeightKeyword::Bold);

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
                                    .font_size(12.0);
                            }
                        });
                    })
                    .height(Pixels(120.0));

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
                    .height(Auto);
                })
                .height(Auto);
            })
            .row_between(Pixels(0.0))
            .child_left(Stretch(1.0))
            .child_right(Stretch(1.0));

            ResizeHandle::new(cx);
        },
    )
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
