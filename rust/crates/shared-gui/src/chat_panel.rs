/// AI tuner chat panel — trait-based LLM integration.
use vizia::prelude::*;

/// Events from the LLM backend to the GUI.
#[derive(Debug, Clone)]
pub enum LlmEvent {
    TextDelta(String),
    ParamUpdate(String),
    IterateRequest,
    Done,
    Error(String),
}

/// Trait for LLM backend implementations.
pub trait LlmBackend: Send + 'static {
    fn send(&mut self, user_text: &str, params_json: &str, metrics_json: &str);
    fn poll(&mut self) -> Option<LlmEvent>;
    fn undo_params(&mut self) -> Option<String>;
    fn reset_session(&mut self);
}

/// A chat message.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub text: String,
}

impl Data for ChatMessage {
    fn same(&self, other: &Self) -> bool {
        self == other
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    User,
    Assistant,
    System,
}

/// Chat panel data model.
#[derive(Debug, Clone, Lens)]
pub struct ChatPanelData {
    pub messages: Vec<ChatMessage>,
    pub input_text: String,
    pub is_busy: bool,
}

impl Default for ChatPanelData {
    fn default() -> Self {
        Self {
            messages: vec![ChatMessage {
                role: ChatRole::System,
                text: "AI tuner ready. Describe the sound you want.".to_string(),
            }],
            input_text: String::new(),
            is_busy: false,
        }
    }
}

pub enum ChatPanelEvent {
    SetInput(String),
    Send,
    Undo,
    NewSession,
    AppendAssistantText(String),
    MarkDone,
}

impl Model for ChatPanelData {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|e, _| match e {
            ChatPanelEvent::SetInput(text) => {
                self.input_text = text.clone();
            }
            ChatPanelEvent::Send => {
                if !self.input_text.is_empty() && !self.is_busy {
                    self.messages.push(ChatMessage {
                        role: ChatRole::User,
                        text: self.input_text.clone(),
                    });
                    self.input_text.clear();
                    self.is_busy = true;
                }
            }
            ChatPanelEvent::Undo => {
                self.messages.push(ChatMessage {
                    role: ChatRole::System,
                    text: "Undone to previous params.".to_string(),
                });
            }
            ChatPanelEvent::NewSession => {
                self.messages.clear();
                self.messages.push(ChatMessage {
                    role: ChatRole::System,
                    text: "New session started.".to_string(),
                });
                self.is_busy = false;
            }
            ChatPanelEvent::AppendAssistantText(text) => {
                if let Some(last) = self.messages.last_mut() {
                    if last.role == ChatRole::Assistant {
                        last.text.push_str(text);
                        return;
                    }
                }
                self.messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    text: text.clone(),
                });
            }
            ChatPanelEvent::MarkDone => {
                self.is_busy = false;
            }
        });
    }
}

/// Build the chat panel view.
pub fn chat_panel_view(cx: &mut Context) {
    VStack::new(cx, |cx| {
        // Message list
        ScrollView::new(cx, |cx| {
            List::new(cx, ChatPanelData::messages, |cx, _idx, item| {
                Label::new(cx, item.map(|m| m.text.clone()));
            });
        });

        // Input area
        HStack::new(cx, |cx| {
            Textbox::new(cx, ChatPanelData::input_text)
                .on_edit(|cx, text| cx.emit(ChatPanelEvent::SetInput(text)));
            Button::new(cx, |cx| Label::new(cx, "Ask"))
                .on_press(|cx| cx.emit(ChatPanelEvent::Send));
            Button::new(cx, |cx| Label::new(cx, "Undo"))
                .on_press(|cx| cx.emit(ChatPanelEvent::Undo));
            Button::new(cx, |cx| Label::new(cx, "New"))
                .on_press(|cx| cx.emit(ChatPanelEvent::NewSession));
        })
        .height(Auto)
        .horizontal_gap(Pixels(4.0));
    })
    .vertical_gap(Pixels(4.0));
}

impl ChatMessage {
    fn role_class(&self) -> &'static str {
        match self.role {
            ChatRole::User => "chat-user",
            ChatRole::Assistant => "chat-assistant",
            ChatRole::System => "chat-system",
        }
    }
}
