//! Declarative parameter schema for audio effects.
//!
//! A pedal's parameter contract is defined as a list of [`ParamDef`] objects.
//! [`ParamSchema`] wraps the list and derives default/bypass params, ranges,
//! sections, and choice ranges. It also provides [`ParamSchema::validate_and_clamp`]
//! for sanitizing LLM-generated parameter dicts.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Parameter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamType {
    Float,
    Int,
    Choice,
    Bool,
    FloatArray,
    IntArray,
}

/// Default/bypass value for a parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    FloatArray(Vec<f64>),
    IntArray(Vec<i64>),
}

impl ParamValue {
    /// Clone lists to prevent shared mutation of defaults.
    pub fn deep_clone(&self) -> Self {
        match self {
            Self::FloatArray(v) => Self::FloatArray(v.clone()),
            Self::IntArray(v) => Self::IntArray(v.clone()),
            other => other.clone(),
        }
    }
}

/// Definition of a single parameter.
#[derive(Debug, Clone)]
pub struct ParamDef {
    pub key: String,
    pub param_type: ParamType,
    pub default: ParamValue,
    pub section: String,
    pub label: String,
    /// Value when bypassed. If `None`, uses `default`.
    pub bypass: Option<ParamValue>,
    /// `(min, max)` for continuous params.
    pub range: Option<(f64, f64)>,
    /// Display names for `Choice` type.
    pub choices: Option<Vec<String>>,
    /// For array types.
    pub array_size: usize,
    /// Not shown in default GUI.
    pub hidden: bool,
    /// Excluded from randomization.
    pub randomize_skip: bool,
}

impl ParamDef {
    /// Create a float param.
    pub fn float(key: &str, default: f64, range: (f64, f64), section: &str) -> Self {
        Self {
            key: key.into(),
            param_type: ParamType::Float,
            default: ParamValue::Float(default),
            section: section.into(),
            label: String::new(),
            bypass: None,
            range: Some(range),
            choices: None,
            array_size: 0,
            hidden: false,
            randomize_skip: false,
        }
    }

    /// Create an int param.
    pub fn int(key: &str, default: i64, range: (f64, f64), section: &str) -> Self {
        Self {
            key: key.into(),
            param_type: ParamType::Int,
            default: ParamValue::Int(default),
            section: section.into(),
            label: String::new(),
            bypass: None,
            range: Some(range),
            choices: None,
            array_size: 0,
            hidden: false,
            randomize_skip: false,
        }
    }

    /// Create a choice param.
    pub fn choice(key: &str, default: i64, choices: &[&str], section: &str) -> Self {
        Self {
            key: key.into(),
            param_type: ParamType::Choice,
            default: ParamValue::Int(default),
            section: section.into(),
            label: String::new(),
            bypass: None,
            range: None,
            choices: Some(choices.iter().map(|s| (*s).to_string()).collect()),
            array_size: 0,
            hidden: false,
            randomize_skip: false,
        }
    }

    /// Create a bool param.
    pub fn bool(key: &str, default: bool, section: &str) -> Self {
        Self {
            key: key.into(),
            param_type: ParamType::Bool,
            default: ParamValue::Bool(default),
            section: section.into(),
            label: String::new(),
            bypass: None,
            range: None,
            choices: None,
            array_size: 0,
            hidden: false,
            randomize_skip: false,
        }
    }

    /// Create a float array param.
    pub fn float_array(key: &str, default: Vec<f64>, range: (f64, f64), section: &str) -> Self {
        let size = default.len();
        Self {
            key: key.into(),
            param_type: ParamType::FloatArray,
            default: ParamValue::FloatArray(default),
            section: section.into(),
            label: String::new(),
            bypass: None,
            range: Some(range),
            choices: None,
            array_size: size,
            hidden: false,
            randomize_skip: false,
        }
    }

    /// Builder: set label.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.into();
        self
    }

    /// Builder: set bypass value.
    pub fn with_bypass(mut self, bypass: ParamValue) -> Self {
        self.bypass = Some(bypass);
        self
    }

    /// Builder: mark hidden.
    pub fn with_hidden(mut self) -> Self {
        self.hidden = true;
        self
    }

    /// Builder: skip randomization.
    pub fn with_randomize_skip(mut self) -> Self {
        self.randomize_skip = true;
        self
    }
}

/// Derives all param structures from a declarative param list.
pub struct ParamSchema {
    params: Vec<ParamDef>,
    by_key: HashMap<String, usize>,
}

impl ParamSchema {
    pub fn new(params: Vec<ParamDef>) -> Self {
        let by_key = params
            .iter()
            .enumerate()
            .map(|(i, p)| (p.key.clone(), i))
            .collect();
        Self { params, by_key }
    }

    /// Default parameter values.
    pub fn default_params(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        for p in &self.params {
            map.insert(p.key.clone(), param_value_to_json(&p.default));
        }
        map
    }

    /// Bypass parameter values.
    pub fn bypass_params(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        for p in &self.params {
            let val = p.bypass.as_ref().unwrap_or(&p.default);
            map.insert(p.key.clone(), param_value_to_json(val));
        }
        map
    }

    /// Continuous param ranges: key -> (min, max).
    pub fn param_ranges(&self) -> HashMap<String, (f64, f64)> {
        let mut result = HashMap::new();
        for p in &self.params {
            if let Some(range) = p.range {
                if p.param_type != ParamType::Choice && p.param_type != ParamType::Bool {
                    result.insert(p.key.clone(), range);
                }
            }
        }
        result
    }

    /// Section name -> list of param keys.
    pub fn param_sections(&self) -> HashMap<String, Vec<String>> {
        let mut sections: HashMap<String, Vec<String>> = HashMap::new();
        for p in &self.params {
            sections
                .entry(p.section.clone())
                .or_default()
                .push(p.key.clone());
        }
        sections
    }

    /// Choice/bool param -> number of options.
    pub fn choice_ranges(&self) -> HashMap<String, usize> {
        let mut result = HashMap::new();
        for p in &self.params {
            match p.param_type {
                ParamType::Choice => {
                    if let Some(choices) = &p.choices {
                        result.insert(p.key.clone(), choices.len());
                    } else if let Some((lo, hi)) = p.range {
                        result.insert(p.key.clone(), (hi - lo) as usize + 1);
                    }
                }
                ParamType::Bool => {
                    result.insert(p.key.clone(), 2);
                }
                _ => {}
            }
        }
        result
    }

    /// Keys to skip during randomization.
    pub fn randomize_skip(&self) -> Vec<String> {
        self.params
            .iter()
            .filter(|p| p.randomize_skip)
            .map(|p| p.key.clone())
            .collect()
    }

    /// Validate and clamp a raw JSON params object (e.g. from LLM output).
    ///
    /// Unknown keys are dropped. Values are type-cast and clamped to range.
    pub fn validate_and_clamp(
        &self,
        raw: &serde_json::Map<String, serde_json::Value>,
    ) -> serde_json::Map<String, serde_json::Value> {
        let defaults = self.default_params();
        let mut result = serde_json::Map::new();

        for (key, value) in raw {
            let idx = match self.by_key.get(key) {
                Some(i) => *i,
                None => continue,
            };
            let p = &self.params[idx];
            let default_val = &defaults[key];

            match default_val {
                serde_json::Value::Array(default_arr) => {
                    let input_arr = match value.as_array() {
                        Some(a) => a,
                        None => continue,
                    };
                    let expected_len = default_arr.len();
                    let mut arr: Vec<serde_json::Value> = input_arr.clone();

                    // Pad or truncate
                    if arr.len() < expected_len {
                        arr.extend_from_slice(&default_arr[arr.len()..]);
                    } else if arr.len() > expected_len {
                        arr.truncate(expected_len);
                    }

                    // Determine if default elements are int
                    let is_int = default_arr
                        .first()
                        .map(|v| v.is_i64() || v.is_u64())
                        .unwrap_or(false);

                    let clamped: Vec<serde_json::Value> = arr
                        .iter()
                        .map(|v| {
                            let num = v.as_f64().unwrap_or(0.0);
                            let clamped = if let Some((lo, hi)) = p.range {
                                num.max(lo).min(hi)
                            } else {
                                num
                            };
                            if is_int {
                                serde_json::Value::Number(
                                    serde_json::Number::from(clamped.round() as i64),
                                )
                            } else {
                                serde_json::Value::Number(
                                    serde_json::Number::from_f64(clamped)
                                        .unwrap_or(serde_json::Number::from(0)),
                                )
                            }
                        })
                        .collect();

                    result.insert(key.clone(), serde_json::Value::Array(clamped));
                }

                serde_json::Value::Number(default_num) => {
                    let is_int = default_num.is_i64() || default_num.is_u64();
                    let num = match value.as_f64() {
                        Some(n) => n,
                        None => continue,
                    };
                    let clamped = if let Some((lo, hi)) = p.range {
                        num.max(lo).min(hi)
                    } else {
                        num
                    };
                    if is_int {
                        result.insert(
                            key.clone(),
                            serde_json::Value::Number(serde_json::Number::from(
                                clamped.round() as i64,
                            )),
                        );
                    } else {
                        result.insert(
                            key.clone(),
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(clamped)
                                    .unwrap_or(serde_json::Number::from(0)),
                            ),
                        );
                    }
                }

                serde_json::Value::Bool(_) => {
                    if let Some(b) = value.as_bool() {
                        result.insert(key.clone(), serde_json::Value::Bool(b));
                    } else if let Some(n) = value.as_i64() {
                        result.insert(key.clone(), serde_json::Value::Bool(n != 0));
                    }
                }

                serde_json::Value::String(_) => {
                    let s = value.as_str().unwrap_or("").to_string();
                    result.insert(key.clone(), serde_json::Value::String(s));
                }

                _ => {
                    result.insert(key.clone(), value.clone());
                }
            }
        }

        result
    }

    /// Look up a param definition by key.
    pub fn get(&self, key: &str) -> Option<&ParamDef> {
        self.by_key.get(key).map(|i| &self.params[*i])
    }

    /// Iterate over all param definitions.
    pub fn iter(&self) -> impl Iterator<Item = &ParamDef> {
        self.params.iter()
    }

    /// Number of params.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// Whether the schema is empty.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

fn param_value_to_json(val: &ParamValue) -> serde_json::Value {
    match val {
        ParamValue::Float(v) => {
            serde_json::Value::Number(serde_json::Number::from_f64(*v).unwrap_or(0.into()))
        }
        ParamValue::Int(v) => serde_json::Value::Number((*v).into()),
        ParamValue::Bool(v) => serde_json::Value::Bool(*v),
        ParamValue::String(v) => serde_json::Value::String(v.clone()),
        ParamValue::FloatArray(v) => serde_json::Value::Array(
            v.iter()
                .map(|x| {
                    serde_json::Value::Number(serde_json::Number::from_f64(*x).unwrap_or(0.into()))
                })
                .collect(),
        ),
        ParamValue::IntArray(v) => {
            serde_json::Value::Array(v.iter().map(|x| serde_json::Value::Number((*x).into())).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> ParamSchema {
        ParamSchema::new(vec![
            ParamDef::float("decay", 2.0, (0.1, 30.0), "reverb"),
            ParamDef::float("mix", 0.5, (0.0, 1.0), "reverb"),
            ParamDef::int("mode", 0, (0.0, 3.0), "reverb").with_label("Mode"),
            ParamDef::bool("bypass", false, "control").with_randomize_skip(),
            ParamDef::float_array("delays", vec![0.1; 8], (0.001, 1.0), "reverb"),
        ])
    }

    #[test]
    fn default_params() {
        let schema = test_schema();
        let defaults = schema.default_params();
        assert_eq!(defaults["decay"], 2.0);
        assert_eq!(defaults["mix"], 0.5);
        assert_eq!(defaults["mode"], 0);
        assert_eq!(defaults["bypass"], false);
        assert_eq!(defaults["delays"].as_array().unwrap().len(), 8);
    }

    #[test]
    fn validate_and_clamp_basic() {
        let schema = test_schema();
        let mut raw = serde_json::Map::new();
        raw.insert("decay".into(), serde_json::json!(50.0)); // over max
        raw.insert("mix".into(), serde_json::json!(-0.5)); // under min
        raw.insert("unknown".into(), serde_json::json!(1.0)); // unknown key

        let result = schema.validate_and_clamp(&raw);
        assert_eq!(result["decay"], 30.0); // clamped to max
        assert_eq!(result["mix"], 0.0); // clamped to min
        assert!(!result.contains_key("unknown")); // dropped
    }

    #[test]
    fn validate_and_clamp_int() {
        let schema = test_schema();
        let mut raw = serde_json::Map::new();
        raw.insert("mode".into(), serde_json::json!(2.7)); // float -> rounded int

        let result = schema.validate_and_clamp(&raw);
        assert_eq!(result["mode"], 3);
    }

    #[test]
    fn validate_and_clamp_array_pad() {
        let schema = test_schema();
        let mut raw = serde_json::Map::new();
        raw.insert("delays".into(), serde_json::json!([0.5, 0.6])); // too short

        let result = schema.validate_and_clamp(&raw);
        let arr = result["delays"].as_array().unwrap();
        assert_eq!(arr.len(), 8); // padded from defaults
        assert_eq!(arr[0], 0.5);
        assert_eq!(arr[1], 0.6);
        assert_eq!(arr[2], 0.1); // default padding
    }

    #[test]
    fn validate_and_clamp_array_truncate() {
        let schema = test_schema();
        let mut raw = serde_json::Map::new();
        raw.insert("delays".into(), serde_json::json!([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])); // too long

        let result = schema.validate_and_clamp(&raw);
        let arr = result["delays"].as_array().unwrap();
        assert_eq!(arr.len(), 8); // truncated
    }

    #[test]
    fn param_sections() {
        let schema = test_schema();
        let sections = schema.param_sections();
        assert_eq!(sections["reverb"].len(), 4);
        assert_eq!(sections["control"].len(), 1);
    }

    #[test]
    fn randomize_skip() {
        let schema = test_schema();
        let skip = schema.randomize_skip();
        assert_eq!(skip, vec!["bypass"]);
    }
}
