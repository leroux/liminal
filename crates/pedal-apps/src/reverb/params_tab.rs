/// Reverb parameters tab — sliders organized by section.
use crate::reverb::data::{ReverbAppData, ReverbAppEvent};
use vizia::prelude::*;

/// Build the parameters tab view.
pub fn params_tab_view(cx: &mut Context) {
    ScrollView::new(cx, |cx| {
        HStack::new(cx, |cx| {
            // Left column — Global + Matrix
            VStack::new(cx, |cx| {
                global_section(cx);
                matrix_section(cx);
            })
            .width(Stretch(1.0));

            // Right column — Per-node + Modulation
            VStack::new(cx, |cx| {
                per_node_section(cx);
                modulation_section(cx);
            })
            .width(Stretch(1.0));
        })
        .horizontal_gap(Pixels(16.0))
        .padding(Pixels(8.0));
    });
}

fn section_header(cx: &mut Context, title: &'static str) {
    Label::new(cx, title)
        .class("section-header")
        .height(Pixels(24.0));
}

fn param_slider(
    cx: &mut Context,
    label: &'static str,
    lens: impl Lens<Target = f64>,
    min: f64,
    max: f64,
    key: &'static str,
) {
    HStack::new(cx, |cx| {
        Label::new(cx, label)
            .width(Pixels(140.0))
            .class("dim");
        Slider::new(cx, lens.map(move |v| ((*v - min) / (max - min)) as f32))
            .on_change(move |cx, val| {
                let value = min + (val as f64) * (max - min);
                cx.emit(ReverbAppEvent::SetParam(
                    key.to_string(),
                    serde_json::Value::from(value),
                ));
            })
            .width(Stretch(1.0));
        Label::new(cx, lens.map(|v| format!("{v:.3}")))
            .width(Pixels(60.0))
            .class("dim");
    })
    .height(Pixels(24.0))
    .horizontal_gap(Pixels(4.0));
}

fn global_section(cx: &mut Context) {
    VStack::new(cx, |cx| {
        section_header(cx, "GLOBAL");

        param_slider(
            cx,
            "Feedback Gain",
            ReverbAppData::params.then(ParamsLens::feedback_gain),
            0.0,
            0.999,
            "feedback_gain",
        );
        param_slider(
            cx,
            "Wet/Dry",
            ReverbAppData::params.then(ParamsLens::wet_dry),
            0.0,
            1.0,
            "wet_dry",
        );
        param_slider(
            cx,
            "Diffusion",
            ReverbAppData::params.then(ParamsLens::diffusion),
            0.0,
            1.0,
            "diffusion",
        );
        param_slider(
            cx,
            "Saturation",
            ReverbAppData::params.then(ParamsLens::saturation),
            0.0,
            1.0,
            "saturation",
        );
        param_slider(
            cx,
            "Stereo Width",
            ReverbAppData::params.then(ParamsLens::stereo_width),
            0.0,
            1.0,
            "stereo_width",
        );
    })
    .class("section")
    .vertical_gap(Pixels(2.0))
    .padding(Pixels(8.0));
}

fn matrix_section(cx: &mut Context) {
    VStack::new(cx, |cx| {
        section_header(cx, "MATRIX");
        Label::new(
            cx,
            ReverbAppData::params.map(|p| format!("Type: {}", p.matrix_type)),
        )
        .class("dim");
        Label::new(
            cx,
            ReverbAppData::params.map(|p| format!("Seed: {}", p.matrix_seed)),
        )
        .class("dim");
        Label::new(cx, "(Matrix heatmap editor -- coming soon)")
            .class("dim")
            .font_size(11.0);
    })
    .class("section")
    .vertical_gap(Pixels(2.0))
    .padding(Pixels(8.0));
}

fn per_node_section(cx: &mut Context) {
    VStack::new(cx, |cx| {
        section_header(cx, "PER-NODE (8 nodes)");

        Label::new(
            cx,
            ReverbAppData::params.map(|p| {
                let times: Vec<String> = p.delay_times.iter().map(|t| format!("{t}")).collect();
                format!("Delays: [{}]", times.join(", "))
            }),
        )
        .class("dim")
        .font_size(11.0);

        Label::new(
            cx,
            ReverbAppData::params.map(|p| {
                let damps: Vec<String> =
                    p.damping_coeffs.iter().map(|d| format!("{d:.2}")).collect();
                format!("Damping: [{}]", damps.join(", "))
            }),
        )
        .class("dim")
        .font_size(11.0);

        Label::new(
            cx,
            ReverbAppData::params.map(|p| {
                let gains: Vec<String> =
                    p.output_gains.iter().map(|g| format!("{g:.2}")).collect();
                format!("Out Gains: [{}]", gains.join(", "))
            }),
        )
        .class("dim")
        .font_size(11.0);

        Label::new(
            cx,
            ReverbAppData::params.map(|p| {
                let pans: Vec<String> = p.node_pans.iter().map(|n| format!("{n:.2}")).collect();
                format!("Pans: [{}]", pans.join(", "))
            }),
        )
        .class("dim")
        .font_size(11.0);
    })
    .class("section")
    .vertical_gap(Pixels(2.0))
    .padding(Pixels(8.0));
}

fn modulation_section(cx: &mut Context) {
    VStack::new(cx, |cx| {
        section_header(cx, "MODULATION");

        param_slider(
            cx,
            "Master Rate",
            ReverbAppData::params.then(ParamsLens::mod_master_rate),
            0.0,
            10.0,
            "mod_master_rate",
        );
        param_slider(
            cx,
            "Correlation",
            ReverbAppData::params.then(ParamsLens::mod_correlation),
            0.0,
            1.0,
            "mod_correlation",
        );
    })
    .class("section")
    .vertical_gap(Pixels(2.0))
    .padding(Pixels(8.0));
}

/// Custom lenses to extract f64 fields from FdnParams.
#[allow(non_snake_case, non_camel_case_types)]
mod ParamsLens {
    use reverb_dsp::FdnParams;
    use vizia::prelude::*;

    macro_rules! param_lens {
        ($name:ident, $field:ident, $ty:ty) => {
            #[derive(Debug, Copy, Clone)]
            pub struct $name;

            impl Lens for $name {
                type Source = FdnParams;
                type Target = $ty;

                fn view<'a>(
                    &self,
                    source: &'a Self::Source,
                ) -> Option<LensValue<'a, Self::Target>> {
                    Some(LensValue::Borrowed(&source.$field))
                }
            }
        };
    }

    param_lens!(feedback_gain, feedback_gain, f64);
    param_lens!(wet_dry, wet_dry, f64);
    param_lens!(diffusion, diffusion, f64);
    param_lens!(saturation, saturation, f64);
    param_lens!(stereo_width, stereo_width, f64);
    param_lens!(mod_master_rate, mod_master_rate, f64);
    param_lens!(mod_correlation, mod_correlation, f64);
}
