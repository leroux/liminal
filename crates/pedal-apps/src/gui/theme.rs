//! Pedal theme definitions and CSS generation.

/// Color theme for a pedal.
#[derive(Debug, Clone)]
pub struct PedalTheme {
    pub name: &'static str,
    /// Primary accent color (e.g. slider fill, headings)
    pub accent: [u8; 3],
    /// Dimmer accent for secondary text
    pub dim: [u8; 3],
    /// Bright accent for highlights
    pub bright: [u8; 3],
    /// Dark background
    pub bg_dark: [u8; 3],
    /// Section background
    pub bg_section: [u8; 3],
    /// Slider track background
    pub bg_slider: [u8; 3],
    /// Very faint background tint
    pub faint: [u8; 3],
}

pub const REVERB_THEME: PedalTheme = PedalTheme {
    name: "reverb",
    accent: [70, 150, 255],
    dim: [30, 65, 120],
    bright: [100, 180, 255],
    bg_dark: [8, 10, 20],
    bg_section: [16, 20, 36],
    bg_slider: [20, 24, 42],
    faint: [8, 14, 30],
};

pub const LOSSY_THEME: PedalTheme = PedalTheme {
    name: "lossy",
    accent: [70, 220, 100],
    dim: [30, 100, 50],
    bright: [100, 255, 130],
    bg_dark: [8, 16, 10],
    bg_section: [14, 24, 16],
    bg_slider: [18, 30, 20],
    faint: [8, 20, 10],
};

pub const FRACTAL_THEME: PedalTheme = PedalTheme {
    name: "fractal",
    accent: [255, 180, 50],
    dim: [140, 90, 20],
    bright: [255, 210, 100],
    bg_dark: [18, 14, 6],
    bg_section: [28, 22, 10],
    bg_slider: [34, 28, 14],
    faint: [22, 16, 6],
};

fn rgb(c: [u8; 3]) -> String {
    format!("rgb({}, {}, {})", c[0], c[1], c[2])
}

fn rgba(c: [u8; 3], a: f32) -> String {
    format!("rgba({}, {}, {}, {a})", c[0], c[1], c[2])
}

/// Generate Vizia CSS theme string for a pedal.
pub fn theme_css(theme: &PedalTheme) -> String {
    format!(
        r#"
* {{
    font-family: "monospace";
    font-size: 13px;
    color: {accent};
}}

.app-root {{
    background-color: {bg_dark};
}}

.section {{
    background-color: {bg_section};
    border-width: 1px;
    border-color: {faint};
}}

.section-header {{
    color: {bright};
    font-weight: bold;
    font-size: 14px;
}}

label {{
    color: {accent};
}}

label.dim {{
    color: {dim};
}}

label.bright {{
    color: {bright};
}}

textbox {{
    background-color: {bg_slider};
    color: {accent};
    border-width: 1px;
    border-color: {faint};
}}

slider {{
    background-color: {bg_slider};
    height: 20px;
}}

slider .track {{
    background-color: {bg_slider};
}}

slider .active {{
    background-color: {accent_40};
}}

slider .thumb {{
    background-color: {accent};
    width: 4px;
}}

button {{
    background-color: {bg_section};
    color: {accent};
    border-width: 1px;
    border-color: {faint};
    height: 28px;
}}

button:hover {{
    background-color: {bg_slider};
}}

button:active {{
    background-color: {accent_20};
}}

scrollview > scrollbar {{
    background-color: {bg_dark};
    width: 8px;
}}

scrollview > scrollbar > .thumb {{
    background-color: {dim};
}}

.toolbar {{
    background-color: {bg_section};
    height: 40px;
    child-space: 4px;
}}

.tab-bar {{
    background-color: {bg_dark};
    height: 32px;
}}

.tab-header {{
    color: {dim};
    height: 30px;
}}

.tab-header:checked {{
    color: {bright};
    border-bottom-width: 2px;
    border-color: {accent};
}}

.status-bar {{
    background-color: {bg_section};
    height: 24px;
}}

.status-text {{
    color: {dim};
    font-size: 11px;
}}

.preset-name {{
    color: {bright};
    font-weight: bold;
}}

.preset-category {{
    color: {accent};
    font-weight: bold;
    font-size: 12px;
}}

.metrics-text {{
    color: {dim};
    font-size: 11px;
}}

.chat-user {{
    color: rgb(100, 160, 255);
}}

.chat-assistant {{
    color: rgb(180, 130, 255);
}}

.chat-system {{
    color: rgb(120, 120, 140);
}}
"#,
        accent = rgb(theme.accent),
        dim = rgb(theme.dim),
        bright = rgb(theme.bright),
        bg_dark = rgb(theme.bg_dark),
        bg_section = rgb(theme.bg_section),
        bg_slider = rgb(theme.bg_slider),
        faint = rgb(theme.faint),
        accent_40 = rgba(theme.accent, 0.4),
        accent_20 = rgba(theme.accent, 0.2),
    )
}
