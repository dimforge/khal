use std::borrow::Cow;
use std::sync::LazyLock;

use regex::Regex;

/// Repair Naga 30's extra `.0` on subnormal decimal literals (nexus#48).
///
/// Its MSL writer tests `value.fract() == 0.0` before appending `.0`. With a
/// foreign flush-to-zero mode, this can produce `0.000...001.0`. Preserve the
/// printed digits and remove only that extra suffix, without doing floating-
/// point arithmetic or changing the thread's floating-point environment.
pub(super) fn repair_float_literals(source: &str) -> Cow<'_, str> {
    // Consume whole preprocessing numbers, identifiers, comments and quoted
    // strings so a numeric-looking substring inside another token is untouched.
    static TOKENS: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(concat!(
            r#"(?s)//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|"#,
            r"[a-zA-Z_][a-zA-Z_0-9]*|(?:[0-9]|\.[0-9])(?:[eEpP][+-]|[a-zA-Z_0-9.])*",
        ))
        .expect("valid MSL token regex")
    });

    let mut output = String::new();
    let mut copied = 0;
    for token in TOKENS.find_iter(source) {
        let text = token.as_str();
        let number = text.strip_suffix(['h', 'f']).unwrap_or(text);
        let Some(digits) = number
            .strip_prefix("0.")
            .and_then(|number| number.strip_suffix(".0"))
        else {
            continue;
        };
        if digits.is_empty()
            || !digits.bytes().all(|byte| byte.is_ascii_digit())
            || digits.bytes().all(|byte| byte == b'0')
        {
            continue;
        }

        if copied == 0 {
            output.reserve(source.len());
        }
        let suffix_end = token.start() + number.len();
        output.push_str(&source[copied..suffix_end - 2]);
        copied = suffix_end;
    }

    if copied == 0 {
        Cow::Borrowed(source)
    } else {
        output.push_str(&source[copied..]);
        Cow::Owned(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_subnormal_literals_without_changing_their_value() {
        for bits in [1, 2, 0x007f_ffff, 0x8000_0001, 0x807f_ffff] {
            let value = f32::from_bits(bits);
            let decimal = value.to_string();
            for suffix in ["", "f", "h"] {
                let malformed = format!("({decimal}.0{suffix})");
                let repaired = repair_float_literals(&malformed);
                assert_eq!(repaired, format!("({decimal}{suffix})"));
                let number = &repaired[1..repaired.len() - 1 - suffix.len()];
                assert_eq!(number.parse::<f32>().unwrap().to_bits(), bits);
            }
        }
        assert_eq!(
            repair_float_literals("float2(0.0001.0, -0.0002.0); half h = 0.000003.0h;"),
            "float2(0.0001, -0.0002); half h = 0.000003h;",
        );
    }

    #[test]
    fn preserves_valid_source_and_unrelated_tokens() {
        let source = r#"
            // 0.0001.0
            /* 0.0002.0h */
            "quoted \"0.0003.0\"; Unicode: λ" '0.0004.0'
            0.0 -0.0 1.0 0.25 0.0001h 1e-45 0x1.0p-2
            identifier0.0001.0 0.0001.0e-2 0.0001.0u 0.0001.0.0
            0.0.0 1.25.0
        "#;
        assert!(matches!(repair_float_literals(source), Cow::Borrowed(_)));
        assert_eq!(repair_float_literals(source), source);
    }

    #[test]
    fn skips_comments_and_strings_while_repairing_code() {
        let source = "/* 0.0001.0 */ 0.0001.0 + \"0.0002.0\"; // 0.0003.0\n-0.0004.0f";
        assert_eq!(
            repair_float_literals(source),
            "/* 0.0001.0 */ 0.0001 + \"0.0002.0\"; // 0.0003.0\n-0.0004f",
        );
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn repaired_literals_compile_on_metal() {
        let tiny = f32::from_bits(1);
        let source = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n\
             kernel void regression(device float* out [[buffer(0)]]) {{\n\
             out[0] = {tiny}.0;\n\
             out[1] = -{tiny}.0;\n\
             out[2] = float(0.000000059604645.0h);\n}}",
        );
        let device = metal::Device::system_default().expect("Metal device required");
        let options = metal::CompileOptions::new();
        options.set_fast_math_enabled(false);
        assert!(device.new_library_with_source(&source, &options).is_err());
        device
            .new_library_with_source(&repair_float_literals(&source), &options)
            .expect("repaired literals must compile")
            .get_function("regression", None)
            .expect("regression kernel must exist");
    }
}
