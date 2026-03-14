//! Serde deserializer helpers for flexible JSON number handling.

use serde::{Deserialize, Deserializer};

/// Accept both `42` and `42.0` from JSON, truncate to i32.
pub fn as_i32<'de, D: Deserializer<'de>>(d: D) -> Result<i32, D::Error> {
    let v: serde_json::Value = Deserialize::deserialize(d)?;
    match &v {
        serde_json::Value::Number(n) => n
            .as_i64()
            .map(|i| i as i32)
            .or_else(|| n.as_f64().map(|f| f as i32))
            .ok_or_else(|| serde::de::Error::custom(format!("cannot convert {n} to i32"))),
        _ => Err(serde::de::Error::custom(format!(
            "expected number, got {v}"
        ))),
    }
}

/// Accept both ints and floats in JSON arrays, convert to Vec<i32>.
pub fn as_i32_vec<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<i32>, D::Error> {
    let arr: Vec<serde_json::Value> = Deserialize::deserialize(d)?;
    arr.iter()
        .enumerate()
        .map(|(i, v)| match v {
            serde_json::Value::Number(n) => n
                .as_i64()
                .map(|x| x as i32)
                .or_else(|| n.as_f64().map(|f| f as i32))
                .ok_or_else(|| {
                    serde::de::Error::custom(format!("cannot convert [{i}]={n} to i32"))
                }),
            _ => Err(serde::de::Error::custom(format!(
                "expected number at [{i}], got {v}"
            ))),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper struct to test `as_i32` via serde deserialization.
    #[derive(Deserialize, Debug)]
    struct SingleI32 {
        #[serde(deserialize_with = "as_i32")]
        value: i32,
    }

    /// Helper struct to test `as_i32_vec` via serde deserialization.
    #[derive(Deserialize, Debug)]
    struct VecI32 {
        #[serde(deserialize_with = "as_i32_vec")]
        values: Vec<i32>,
    }

    #[test]
    fn as_i32_from_integer() {
        let json = r#"{"value": 42}"#;
        let parsed: SingleI32 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.value, 42);
    }

    #[test]
    fn as_i32_from_float_whole() {
        let json = r#"{"value": 42.0}"#;
        let parsed: SingleI32 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.value, 42);
    }

    #[test]
    fn as_i32_truncates_fractional() {
        let json = r#"{"value": 42.9}"#;
        let parsed: SingleI32 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.value, 42);
    }

    #[test]
    fn as_i32_negative() {
        let json = r#"{"value": -7}"#;
        let parsed: SingleI32 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.value, -7);
    }

    #[test]
    fn as_i32_zero() {
        let json = r#"{"value": 0}"#;
        let parsed: SingleI32 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.value, 0);
    }

    #[test]
    fn as_i32_rejects_string() {
        let json = r#"{"value": "hello"}"#;
        let result: Result<SingleI32, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn as_i32_vec_from_ints() {
        let json = r#"{"values": [1, 2, 3, 4]}"#;
        let parsed: VecI32 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn as_i32_vec_from_mixed() {
        let json = r#"{"values": [1, 2.0, 3, 4.5]}"#;
        let parsed: VecI32 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.values, vec![1, 2, 3, 4]);
    }

    #[test]
    fn as_i32_vec_empty() {
        let json = r#"{"values": []}"#;
        let parsed: VecI32 = serde_json::from_str(json).unwrap();
        assert!(parsed.values.is_empty());
    }

    #[test]
    fn as_i32_vec_rejects_string_element() {
        let json = r#"{"values": [1, "bad", 3]}"#;
        let result: Result<VecI32, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
