//! Vizia GUI for the Reverb plugin.

use crate::params::ReverbPluginParams;
use crate::presets::{self, Preset};
use claudewire::chat::{ChatBackend, ChatMsg};
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaTheming};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// GUI data model
// ---------------------------------------------------------------------------

#[derive(Clone, Lens)]
struct GuiData {
    params: Arc<ReverbPluginParams>,
    presets: Vec<Preset>,
    selected: usize,
    chat_input: String,
    chat_messages: Vec<(String, String)>, // (role, text)
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

const SYSTEM_PROMPT: &str = "You are an audio effects tuning assistant for a reverb plugin. \
    Help the user achieve their desired reverb sound. Give concise advice about parameter adjustments. \
    Keep responses short and focused on audio production.";

pub fn create(params: Arc<ReverbPluginParams>) -> Option<Box<dyn Editor>> {
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

            // Poll chat every 200ms (only does work if backend exists)
            cx.spawn(|proxy| loop {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if proxy.emit(GuiEvent::PollChat).is_err() {
                    break;
                }
            });

            VStack::new(cx, |cx| {
                Label::new(cx, "Reverb")
                    .font_family(vec![FamilyOwned::Name(String::from(assets::NOTO_SANS))])
                    .font_weight(FontWeightKeyword::Thin)
                    .font_size(30.0)
                    .height(Pixels(42.0))
                    .child_top(Stretch(1.0))
                    .child_bottom(Pixels(0.0));

                // Preset dropdown
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

                // Params
                ScrollView::new(cx, 0.0, 0.0, false, true, |cx| {
                    GenericUi::new(cx, GuiData::params);
                })
                .width(Percentage(100.0));

                // Chat section
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

fn apply_preset_vizia(
    cx: &mut EventContext,
    pp: &ReverbPluginParams,
    p: &reverb_dsp::FdnParams,
) {
    use crate::params::MatrixType;

    let f = |arr: &[f64], i: usize, def: f64| -> f32 {
        arr.get(i).copied().unwrap_or(def) as f32
    };
    let fi = |arr: &[i32], i: usize, def: i32| -> i32 {
        arr.get(i).copied().unwrap_or(def)
    };

    set_param_f32(cx, &pp.feedback_gain, p.feedback_gain as f32);
    set_param_f32(cx, &pp.wet_dry, p.wet_dry as f32);
    set_param_f32(cx, &pp.diffusion, p.diffusion as f32);
    set_param_i32(cx, &pp.diffusion_stages, p.diffusion_stages);
    set_param_f32(cx, &pp.saturation, p.saturation as f32);
    set_param_i32(cx, &pp.pre_delay, p.pre_delay);
    set_param_f32(cx, &pp.stereo_width, p.stereo_width as f32);

    let mt = match p.matrix_type.as_str() {
        "householder" => MatrixType::Householder,
        "hadamard" => MatrixType::Hadamard,
        "diagonal" => MatrixType::Diagonal,
        "random_orthogonal" => MatrixType::RandomOrthogonal,
        "circulant" => MatrixType::Circulant,
        "stautner_puckette" => MatrixType::StautnerPuckette,
        _ => MatrixType::Householder,
    };
    let ptr = pp.matrix_type.as_ptr();
    let norm = pp.matrix_type.preview_normalized(mt);
    cx.emit(RawParamEvent::BeginSetParameter(ptr));
    cx.emit(RawParamEvent::SetParameterNormalized(ptr, norm));
    cx.emit(RawParamEvent::EndSetParameter(ptr));
    set_param_i32(cx, &pp.matrix_seed, p.matrix_seed);

    set_param_i32(cx, &pp.delay_time_1, fi(&p.delay_times, 0, 1310));
    set_param_i32(cx, &pp.delay_time_2, fi(&p.delay_times, 1, 1637));
    set_param_i32(cx, &pp.delay_time_3, fi(&p.delay_times, 2, 1821));
    set_param_i32(cx, &pp.delay_time_4, fi(&p.delay_times, 3, 2112));
    set_param_i32(cx, &pp.delay_time_5, fi(&p.delay_times, 4, 2342));
    set_param_i32(cx, &pp.delay_time_6, fi(&p.delay_times, 5, 2615));
    set_param_i32(cx, &pp.delay_time_7, fi(&p.delay_times, 6, 2986));
    set_param_i32(cx, &pp.delay_time_8, fi(&p.delay_times, 7, 3223));

    set_param_f32(cx, &pp.damping_1, f(&p.damping_coeffs, 0, 0.3));
    set_param_f32(cx, &pp.damping_2, f(&p.damping_coeffs, 1, 0.3));
    set_param_f32(cx, &pp.damping_3, f(&p.damping_coeffs, 2, 0.3));
    set_param_f32(cx, &pp.damping_4, f(&p.damping_coeffs, 3, 0.3));
    set_param_f32(cx, &pp.damping_5, f(&p.damping_coeffs, 4, 0.3));
    set_param_f32(cx, &pp.damping_6, f(&p.damping_coeffs, 5, 0.3));
    set_param_f32(cx, &pp.damping_7, f(&p.damping_coeffs, 6, 0.3));
    set_param_f32(cx, &pp.damping_8, f(&p.damping_coeffs, 7, 0.3));

    set_param_f32(cx, &pp.output_gain_1, f(&p.output_gains, 0, 1.0));
    set_param_f32(cx, &pp.output_gain_2, f(&p.output_gains, 1, 1.0));
    set_param_f32(cx, &pp.output_gain_3, f(&p.output_gains, 2, 1.0));
    set_param_f32(cx, &pp.output_gain_4, f(&p.output_gains, 3, 1.0));
    set_param_f32(cx, &pp.output_gain_5, f(&p.output_gains, 4, 1.0));
    set_param_f32(cx, &pp.output_gain_6, f(&p.output_gains, 5, 1.0));
    set_param_f32(cx, &pp.output_gain_7, f(&p.output_gains, 7, 1.0));
    set_param_f32(cx, &pp.output_gain_8, f(&p.output_gains, 7, 1.0));

    set_param_f32(cx, &pp.node_pan_1, f(&p.node_pans, 0, -1.0));
    set_param_f32(cx, &pp.node_pan_2, f(&p.node_pans, 1, -0.714));
    set_param_f32(cx, &pp.node_pan_3, f(&p.node_pans, 2, -0.429));
    set_param_f32(cx, &pp.node_pan_4, f(&p.node_pans, 3, -0.143));
    set_param_f32(cx, &pp.node_pan_5, f(&p.node_pans, 4, 0.143));
    set_param_f32(cx, &pp.node_pan_6, f(&p.node_pans, 5, 0.429));
    set_param_f32(cx, &pp.node_pan_7, f(&p.node_pans, 6, 0.714));
    set_param_f32(cx, &pp.node_pan_8, f(&p.node_pans, 7, 1.0));

    set_param_f32(cx, &pp.mod_master_rate, p.mod_master_rate as f32);
    set_param_f32(cx, &pp.mod_correlation, p.mod_correlation as f32);
    let avg = |arr: &[f64]| -> f32 {
        if arr.is_empty() { 0.0 } else { (arr.iter().sum::<f64>() / arr.len() as f64) as f32 }
    };
    set_param_f32(cx, &pp.mod_depth_delay, avg(&p.mod_depth_delay));
    set_param_f32(cx, &pp.mod_depth_damping, avg(&p.mod_depth_damping));
    set_param_f32(cx, &pp.mod_depth_output, avg(&p.mod_depth_output));
    set_param_f32(cx, &pp.mod_depth_matrix, p.mod_depth_matrix as f32);
}
