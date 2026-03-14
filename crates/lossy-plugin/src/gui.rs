//! Vizia GUI for the Lossy plugin.

use crate::capture::AudioCapture;
use crate::chat_logic;
use crate::params::LossyPluginParams;
use crate::presets::{self, Preset};
use pedal_chat::{ChatBackend, ChatMsg};
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaTheming};
use std::sync::Arc;

/// Simple file logger for debugging chat in plugin context.
macro_rules! chat_log {
    ($($arg:tt)*) => {{
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/lossy-chat.log")
        {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(f, "[{now}] {}", format_args!($($arg)*));
        }
    }};
}

#[derive(Clone, Lens)]
struct GuiData {
    params: Arc<LossyPluginParams>,
    presets: Vec<Preset>,
    selected: usize,
    chat_input: String,
    chat_messages: Vec<(String, String)>,
    chat_busy: bool,
    /// Raw accumulated assistant text for JSON extraction on Done.
    pending_response: String,
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
                chat_log!("SetChatInput: {:?}", text);
                self.chat_input = text.clone();
            }
            GuiEvent::SendChat => {
                if self.chat_input.trim().is_empty() || self.chat_busy {
                    chat_log!("SendChat: skipped (empty={}, busy={})", self.chat_input.trim().is_empty(), self.chat_busy);
                    return;
                }
                let user_text = self.chat_input.clone();
                chat_log!("SendChat: user_text={:?}", user_text);
                self.chat_messages.push(("user".into(), user_text.clone()));
                self.chat_input.clear();
                self.chat_busy = true;
                self.pending_response.clear();

                // Build enriched prompt with current params + audio metrics
                let dsp_params = self.params.to_dsp_params();
                let metrics = AUDIO_CAPTURE.with(|cell| {
                    cell.borrow()
                        .as_ref()
                        .and_then(|c| c.compute_metrics())
                });
                chat_log!("SendChat: has_metrics={}", metrics.is_some());
                let full_prompt = chat_logic::build_enriched_prompt(
                    &user_text,
                    &dsp_params,
                    metrics.as_ref(),
                );
                chat_log!("SendChat: prompt_len={}", full_prompt.len());

                // Show context sent to AI
                let mut context_summary = String::new();
                if let Some(m) = &metrics {
                    context_summary = format!(
                        "[context: RMS {:.0}dB, peak {:.0}dB, centroid {:.0}Hz]",
                        m.rms_db, m.peak_db, m.spectral_centroid_hz,
                    );
                } else {
                    context_summary = "[context: params sent, no audio]".into();
                }
                self.chat_messages.push(("context".into(), context_summary));

                CHAT_BACKEND.with(|cell| {
                    let mut backend = cell.borrow_mut();
                    if backend.is_none() {
                        chat_log!("SendChat: creating new ChatBackend");
                        *backend = Some(ChatBackend::new(SYSTEM_PROMPT));
                    }
                    if let Some(b) = backend.as_ref() {
                        chat_log!("SendChat: sending to backend");
                        b.send(&full_prompt);
                    }
                });
            }
            GuiEvent::PollChat => {
                CHAT_BACKEND.with(|cell| {
                    if let Some(b) = cell.borrow().as_ref() {
                        let msgs = b.poll();
                        if !msgs.is_empty() {
                            chat_log!("PollChat: got {} messages", msgs.len());
                        }
                        for msg in msgs {
                            match msg {
                                ChatMsg::Text(text) => {
                                    chat_log!("PollChat: Text len={}", text.len());
                                    // Track raw response for JSON extraction
                                    self.pending_response = text.clone();
                                    // Update display (strip JSON block for readability)
                                    let display = chat_logic::strip_json_block(&text);
                                    if let Some(last) = self.chat_messages.last_mut() {
                                        if last.0 == "assistant" {
                                            last.1 = display;
                                            return;
                                        }
                                    }
                                    self.chat_messages
                                        .push(("assistant".into(), display));
                                }
                                ChatMsg::Error(e) => {
                                    chat_log!("PollChat: Error: {}", e);
                                    self.chat_messages.push(("error".into(), e));
                                    self.chat_busy = false;
                                }
                                ChatMsg::Done => {
                                    chat_log!("PollChat: Done. pending_response len={}", self.pending_response.len());
                                    chat_log!("PollChat: pending_response: {:?}", &self.pending_response[..self.pending_response.len().min(500)]);
                                    // Extract and apply params from JSON block
                                    let extracted = chat_logic::extract_json_block(&self.pending_response);
                                    chat_log!("PollChat: extracted json={}", extracted.is_some());
                                    if let Some(extracted) = extracted {
                                        chat_log!("PollChat: extracted keys: {:?}", extracted.as_object().map(|o| o.keys().collect::<Vec<_>>()));
                                        let current = self.params.to_dsp_params();
                                        match chat_logic::merge_params(&current, &extracted) {
                                            Ok(new_params) => {
                                                chat_log!("PollChat: merge OK, applying params");
                                                apply_preset_vizia(
                                                    cx, &self.params, &new_params,
                                                );
                                                // Show the applied JSON
                                                let keys: Vec<String> = extracted
                                                    .as_object()
                                                    .map(|o| o.iter().map(|(k, v)| format!("{k}={v}")).collect())
                                                    .unwrap_or_default();
                                                self.chat_messages.push((
                                                    "applied".into(),
                                                    format!("[applied: {}]", keys.join(", ")),
                                                ));
                                            }
                                            Err(e) => {
                                                chat_log!("PollChat: merge FAILED: {}", e);
                                                self.chat_messages.push((
                                                    "error".into(),
                                                    format!("Param error: {e}"),
                                                ));
                                            }
                                        }
                                    }
                                    self.pending_response.clear();
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
    static AUDIO_CAPTURE: std::cell::RefCell<Option<AudioCapture>> = const { std::cell::RefCell::new(None) };
}

const SYSTEM_PROMPT: &str = "\
You are an expert audio engineer tuning a codec artifact emulator (lossy audio effect).

SIGNAL CHAIN: Input -> Spectral Loss (STFT) -> Crush/Decimate -> Packets -> Filter -> Verb -> Gate -> Limiter -> Wet/Dry Mix -> Output

PARAMETERS AND RANGES:

Spectral Loss:
- inverse: 0=Standard (hear processed), 1=Inverse (hear residual)
- jitter (0.0-1.0): Random phase perturbation. 0=off, 1=max.
- loss (0.0-1.0): Destruction amount. 0=clean, 1.0=destroyed.
- window_size (64-16384): FFT window size. Large=smooth/dark, small=glitchy.
- hop_divisor (1-8): Overlap ratio. 4=75% overlap (default).
- n_bands (2-64): Bark-like bands for psychoacoustic gating.
- phase_loss (0.0-1.0): Phase quantization. 0=off, higher=more phasey.
- quantizer: 0=uniform, 1=compand (MP3-style).
- weighting (0.0-1.0): 0=equal freq weighting, 1=psychoacoustic ATH.
- hf_threshold (0.0-1.0): HF rolloff threshold. Default 0.3.

Crush: crush (0.0-1.0) bitcrusher, decimate (0.0-1.0) sample rate reduction.
Packets: packets 0=Clean/1=Loss/2=Repeat, packet_rate (0.0-1.0), packet_size (5-200ms).
Filter: filter_type 0=Bypass/1=Bandpass/2=Notch, filter_freq (20-20000Hz), filter_width (0-1), filter_slope 6=6dB/24=24dB/96=96dB per octave.
Effects: verb (0-1) lo-fi reverb, decay (0-1), freeze 0/1, freezer (0-1), gate (0-1).
Output: wet_dry (0-1), auto_gain (0-1), loss_gain (0-1).

RECIPES:
- Underwater/streaming: loss 0.7-0.9, window_size 4096, no crush
- Glitchy digital: loss 0.5, window_size 256-512, packet loss, crush 0.3
- Lo-fi radio: loss 0.4, bandpass 800-2000Hz, verb 0.2, decimate 0.3
- Frozen texture: freeze on, slushy mode, loss 0.5, verb 0.3
- Extreme destruction: loss 1.0, crush 0.6, decimate 0.5, packet repeat

RULES:
- You can chat normally without changing params. Only include a ```json block when ready to apply changes.
- The ```json block must be a flat JSON object with parameter key-value pairs.
- Only include params you want to change — missing keys keep current values.
- Stay within documented ranges. Use integer values for integer params.
- Keep text explanations concise. Focus on what you changed and why.
";

// Green theme — overrides nih-plug's light default
const THEME_CSS: &str = r#"
:root {
    background-color: rgb(8, 16, 10);
    color: rgb(160, 220, 170);
    font-size: 13;
}
label {
    color: rgb(160, 220, 170);
}
textbox {
    background-color: rgb(18, 30, 20);
    color: rgb(160, 220, 170);
    border-color: rgb(35, 90, 45);
    border-radius: 2px;
}
button {
    background-color: rgb(18, 32, 22);
    color: rgb(160, 220, 170);
    border-color: rgb(35, 90, 45);
    border-radius: 2px;
}
button:hover {
    background-color: rgb(25, 45, 30);
}
param-slider {
    border-color: rgb(35, 90, 45);
}
param-slider .fill {
    background-color: rgb(45, 160, 70);
}
param-slider .value-entry {
    color: rgb(160, 220, 170);
}
param-slider .value-entry .caret {
    background-color: rgb(70, 220, 100);
}
param-button {
    border-color: rgb(35, 90, 45);
    color: rgb(160, 220, 170);
}
param-button:checked {
    background-color: rgb(45, 160, 70);
}
scrollview scrollbar {
    background-color: rgb(8, 16, 10);
}
scrollview scrollbar .thumb {
    background-color: rgb(35, 90, 45);
}
dropdown popup {
    background-color: rgb(14, 24, 16);
    border-color: rgb(35, 90, 45);
}
"#;

pub fn create(
    params: Arc<LossyPluginParams>,
    audio_capture: AudioCapture,
) -> Option<Box<dyn Editor>> {
    create_vizia_editor(
        params.editor_state.clone(),
        ViziaTheming::Custom,
        move |cx, _| {
            assets::register_noto_sans_light(cx);
            cx.add_stylesheet(THEME_CSS).ok();

            chat_log!("GUI create: editor opened");

            // Store audio capture in thread_local for access in event handlers
            AUDIO_CAPTURE.with(|cell| {
                *cell.borrow_mut() = Some(audio_capture.clone());
            });

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
                pending_response: String::new(),
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
                        Label::new(cx, "Lossy")
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
                            // ── Spectral Loss ──
                            section(cx, "Spectral Loss", |cx| {
                                param_row_enum(cx, "Mode", |p| &p.mode);
                                param_row(cx, "Loss", |p| &p.loss);
                                param_row(cx, "Phase Loss", |p| &p.phase_loss);
                                param_row(cx, "Window Size", |p| &p.window_size);
                                param_row(cx, "Hop Divisor", |p| &p.hop_divisor);
                                param_row(cx, "Bands", |p| &p.n_bands);
                                param_row_enum(cx, "Quantizer", |p| &p.quantizer);
                                param_row(cx, "Weighting", |p| &p.weighting);
                                param_row(cx, "HF Threshold", |p| &p.hf_threshold);
                                param_row(cx, "Jitter", |p| &p.jitter);
                            });

                            // ── Crush ──
                            section(cx, "Crush", |cx| {
                                param_row(cx, "Crush", |p| &p.crush);
                                param_row(cx, "Decimate", |p| &p.decimate);
                            });

                            // ── Packets ──
                            section(cx, "Packets", |cx| {
                                param_row_enum(cx, "Packets", |p| &p.packets);
                                param_row(cx, "Pkt Rate", |p| &p.packet_rate);
                                param_row(cx, "Pkt Size", |p| &p.packet_size);
                            });

                            // ── Filter ──
                            section(cx, "Filter", |cx| {
                                param_row_enum(cx, "Type", |p| &p.filter_type);
                                param_row(cx, "Freq", |p| &p.filter_freq);
                                param_row(cx, "Width", |p| &p.filter_width);
                                param_row_enum(cx, "Slope", |p| &p.filter_slope);
                            });

                            // ── Reverb ──
                            section(cx, "Reverb", |cx| {
                                param_row(cx, "Verb", |p| &p.verb);
                                param_row(cx, "Decay", |p| &p.decay);
                            });

                            // ── Freeze ──
                            section(cx, "Freeze", |cx| {
                                HStack::new(cx, |cx| {
                                    Label::new(cx, "Freeze")
                                        .width(Pixels(100.0))
                                        .child_top(Stretch(1.0))
                                        .child_bottom(Stretch(1.0));
                                    ParamButton::new(cx, GuiData::params, |p| &p.freeze)
                                        .width(Stretch(1.0));
                                })
                                .col_between(Pixels(6.0))
                                .height(Auto)
                                .width(Stretch(1.0));
                                param_row(cx, "Freezer", |p| &p.freezer);
                            });

                            // ── Gate / Output ──
                            section(cx, "Gate / Output", |cx| {
                                param_row(cx, "Gate", |p| &p.gate);
                                param_row(cx, "Wet/Dry", |p| &p.wet_dry);
                                param_row(cx, "Auto Gain", |p| &p.auto_gain);
                                param_row(cx, "Loss Gain", |p| &p.loss_gain);
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
                    .background_color(Color::rgb(30, 100, 50));

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
                                    "context" => "",
                                    "applied" => "",
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
            .background_color(Color::rgb(8, 16, 10));

            ResizeHandle::new(cx);
        },
    )
}

/// Label + ParamSlider row for continuous parameters.
fn param_row<P, FMap>(cx: &mut Context, label: &str, params_to_param: FMap)
where
    P: Param + 'static,
    FMap: Fn(&Arc<LossyPluginParams>) -> &P + Copy + 'static,
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
    FMap: Fn(&Arc<LossyPluginParams>) -> &P + Copy + 'static,
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

fn section(cx: &mut Context, title: &str, content: impl FnOnce(&mut Context)) {
    VStack::new(cx, |cx| {
        Label::new(cx, title)
            .font_size(13.0)
            .font_weight(FontWeightKeyword::Bold)
            .color(Color::rgb(100, 255, 130))
            .width(Stretch(1.0))
            .height(Auto)
            .top(Pixels(2.0))
            .bottom(Pixels(4.0));
        Element::new(cx)
            .height(Pixels(1.0))
            .width(Stretch(1.0))
            .background_color(Color::rgb(30, 100, 50))
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
    pp: &LossyPluginParams,
    p: &lossy_dsp::LossyParams,
) {
    use crate::params::FilterSlope;

    set_param_enum(cx, &pp.mode, p.inverse);
    set_param_f32(cx, &pp.jitter, p.jitter as f32);
    set_param_f32(cx, &pp.loss, p.loss as f32);
    set_param_i32(cx, &pp.window_size, p.window_size);
    set_param_i32(cx, &pp.hop_divisor, p.hop_divisor);
    set_param_i32(cx, &pp.n_bands, p.n_bands);
    set_param_f32(cx, &pp.phase_loss, p.phase_loss as f32);
    set_param_enum(cx, &pp.quantizer, p.quantizer);
    set_param_f32(cx, &pp.weighting, p.weighting as f32);
    set_param_f32(cx, &pp.hf_threshold, p.hf_threshold as f32);
    set_param_f32(cx, &pp.crush, p.crush as f32);
    set_param_f32(cx, &pp.decimate, p.decimate as f32);
    set_param_enum(cx, &pp.packets, p.packets);
    set_param_f32(cx, &pp.packet_rate, p.packet_rate as f32);
    set_param_f32(cx, &pp.packet_size, p.packet_size as f32);
    set_param_enum(cx, &pp.filter_type, p.filter_type);
    set_param_f32(cx, &pp.filter_freq, p.filter_freq as f32);
    set_param_f32(cx, &pp.filter_width, p.filter_width as f32);

    let slope = match p.filter_slope {
        6 => FilterSlope::Slope6,
        96 => FilterSlope::Slope96,
        _ => FilterSlope::Slope24,
    };
    let ptr = pp.filter_slope.as_ptr();
    let norm = pp.filter_slope.preview_normalized(slope);
    cx.emit(RawParamEvent::BeginSetParameter(ptr));
    cx.emit(RawParamEvent::SetParameterNormalized(ptr, norm));
    cx.emit(RawParamEvent::EndSetParameter(ptr));

    set_param_f32(cx, &pp.verb, p.verb as f32);
    set_param_f32(cx, &pp.decay, p.decay as f32);
    set_param_bool(cx, &pp.freeze, p.freeze != 0);
    set_param_f32(cx, &pp.freezer, p.freezer as f32);
    set_param_f32(cx, &pp.gate, p.gate as f32);
    set_param_f32(cx, &pp.threshold, p.threshold as f32);
    set_param_f32(cx, &pp.auto_gain, p.auto_gain as f32);
    set_param_f32(cx, &pp.loss_gain, p.loss_gain as f32);
    set_param_f32(cx, &pp.wet_dry, p.wet_dry as f32);
}
