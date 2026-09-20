//! Laya's prompt, token for token (upstream `common.py`: `render_options`,
//! `serialize_state`, `build_prefix`, `build_sequence`). Tokenization is
//! the caller's (the HF tokenizer natively, transformers.js in the browser)
//! through [`Tokenize`]; the budgets and truncations live here so both
//! runtimes pack the same ids.
//!
//! ```text
//! [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
//! ```
//!
//! Every option is `[MASK]` + at most 48 tokens of ` <option text>`; the
//! options and the instructions share `head_max_len` tokens (options first:
//! when fewer than 16 are left for the instructions, options are cut to an
//! equal share), the state gets what remains of `max_len`.

use anyhow::{bail, Result};
use grande_core::{Question, Request};
use serde_json::Value;

use super::{LayaConfig, Sequence, QTYPES};

/// The tokenizer the prompt builder needs: text → ids, no special tokens.
pub trait Tokenize {
    fn encode(&self, text: &str) -> Vec<u32>;
}

impl<F: Fn(&str) -> Vec<u32>> Tokenize for F {
    fn encode(&self, text: &str) -> Vec<u32> {
        self(text)
    }
}

/// Python's `json.dumps(v, ensure_ascii=False)` with the default separators
/// (`", "`, `": "`); `ascii` = the default `ensure_ascii=True` instead.
pub fn py_json(v: &Value, ascii: bool) -> String {
    let mut out = String::new();
    write_py_json(v, ascii, &mut out);
    out
}

fn write_py_json(v: &Value, ascii: bool, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&py_number(n)),
        Value::String(s) => out.push_str(&py_string(s, ascii)),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(x, ascii, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&py_string(k, ascii));
                out.push_str(": ");
                write_py_json(x, ascii, out);
            }
            out.push('}');
        }
    }
}

/// Python's float repr for the values that come through JSON: integers as
/// is, floats with a `.0` when whole and `e+NN` / `e-NN` exponents.
fn py_number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    let f = n.as_f64().unwrap_or(0.0);
    if !f.is_finite() {
        return if f.is_nan() {
            "NaN".into()
        } else if f > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    let s = format!("{f:?}");
    match s.find('e') {
        None => s,
        Some(i) => {
            let (m, e) = s.split_at(i);
            let e = &e[1..];
            let (sign, digits) = match e.strip_prefix('-') {
                Some(d) => ("-", d),
                None => ("+", e),
            };
            let digits = if digits.len() < 2 {
                format!("0{digits}")
            } else {
                digits.to_string()
            };
            // Rust prints `1.0e16` where Python prints `1e+16`.
            let m = m.strip_suffix(".0").unwrap_or(m);
            format!("{m}e{sign}{digits}")
        }
    }
}

fn py_string(s: &str, ascii: bool) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ascii && !c.is_ascii() => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Strings pass through; anything structured becomes compact Python JSON.
fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_json(other, false),
    }
}

/// None or `""` mean "no description"; other falsy values are legitimate.
fn is_blank(v: &Option<Value>) -> bool {
    match v {
        None => true,
        Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        _ => false,
    }
}

/// Option texts in label-index order. Noul is always `[false, true]`.
pub fn render_options(q: &Question) -> Vec<String> {
    match q {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(k, v)| {
                if is_blank(v) {
                    k.clone()
                } else {
                    format!("{k}: {}", render_criterion(v.as_ref().unwrap()))
                }
            })
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
            .collect(),
        Question::Noul { criteria, .. } => {
            let get = |key: &str| -> Option<Value> {
                criteria
                    .as_ref()
                    .and_then(|m| m.get(key))
                    .cloned()
                    .flatten()
            };
            let f = get("false");
            let t = get("true");
            vec![
                format!(
                    "false: {}",
                    if is_blank(&f) {
                        "no, the statement does not hold".to_string()
                    } else {
                        render_criterion(f.as_ref().unwrap())
                    }
                ),
                format!(
                    "true: {}",
                    if is_blank(&t) {
                        "yes, the statement holds".to_string()
                    } else {
                        render_criterion(t.as_ref().unwrap())
                    }
                ),
            ]
        }
    }
}

/// 0 = choice, 1 = score, 2 = noul.
pub fn qtype(q: &Question) -> usize {
    match q {
        Question::Choice { .. } => 0,
        Question::Score { .. } => 1,
        Question::Noul { .. } => 2,
    }
}

fn instructions(q: &Question) -> String {
    let ins = match q {
        Question::Choice { instructions, .. }
        | Question::Score { instructions, .. }
        | Question::Noul { instructions, .. } => instructions,
    };
    match ins {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        // `json.dumps(instructions)`: ASCII-escaped, Python's default.
        Some(other) => py_json(other, true),
    }
}

/// A string state is used as is; anything else is `json.dumps(state,
/// ensure_ascii=False)`.
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => py_json(other, false),
    }
}

/// The question-only prefix: ids and the index of each option's `[MASK]`.
/// `order` lists the options (Laya's indices: Noul is `[false, true]`) in
/// the order they are laid out; None = the natural order.
pub fn build_prefix(
    tok: &dyn Tokenize,
    cfg: &LayaConfig,
    q: &Question,
    order: Option<&[usize]>,
) -> (Vec<u32>, Vec<usize>) {
    let head_max_len = cfg.head_max_len;
    let mask = cfg.mask_text.as_str();
    let opts = render_options(q);
    let natural: Vec<usize> = (0..opts.len()).collect();
    let order = order.unwrap_or(&natural);
    let ins = instructions(q).replace(mask, " ");
    let mut head_ids = tok.encode(&format!("{} question: {ins}", QTYPES[qtype(q)]));
    let mut opt_ids: Vec<Vec<u32>> = order
        .iter()
        .map(|&i| &opts[i])
        .map(|o| {
            let mut v = vec![cfg.mask];
            v.extend(
                tok.encode(&format!(" {}", o.replace(mask, " ")))
                    .into_iter()
                    .take(48),
            );
            v
        })
        .collect();
    let used = |opt_ids: &[Vec<u32>]| opt_ids.iter().map(Vec::len).sum::<usize>() as isize;
    let mut opt_budget = head_max_len as isize - used(&opt_ids);
    if opt_budget < 16 {
        let per = ((head_max_len.saturating_sub(16)) / opt_ids.len().max(1)).max(4);
        for o in &mut opt_ids {
            o.truncate(per);
        }
        opt_budget = head_max_len as isize - used(&opt_ids);
    }
    head_ids.truncate(opt_budget.max(8) as usize);
    let mut ids = Vec::with_capacity(head_max_len + 2);
    ids.push(cfg.cls);
    ids.extend(head_ids);
    ids.push(cfg.sep);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(cfg.sep);
    (ids, markers)
}

/// One question over `state_ids` (the tokenized state, see
/// [`serialize_state`] — tokenized once per request): the prefix, then as
/// much of the state as fits, then `[SEP]`, cut at `max_len`.
pub fn build_sequence(
    tok: &dyn Tokenize,
    cfg: &LayaConfig,
    state_ids: &[u32],
    q: &Question,
    order: Option<&[usize]>,
) -> Result<Sequence> {
    let (mut ids, markers) = build_prefix(tok, cfg, q, order);
    let n_opts = markers.len();
    let room = cfg.max_len.saturating_sub(ids.len() + 1);
    ids.extend_from_slice(&state_ids[..state_ids.len().min(room)]);
    ids.push(cfg.sep);
    ids.truncate(cfg.max_len);
    let markers: Vec<usize> = markers.into_iter().filter(|&m| m < cfg.max_len).collect();
    if markers.len() != n_opts {
        bail!("too many options for the token budget");
    }
    Ok(Sequence {
        ids,
        markers,
        qtype: qtype(q),
    })
}

/// Tokenize the state as laya does (the mask token's text neutralized).
pub fn state_ids(tok: &dyn Tokenize, cfg: &LayaConfig, state: &Value) -> Vec<u32> {
    tok.encode(&serialize_state(state).replace(cfg.mask_text.as_str(), " "))
}

/// Every question of a request as a sequence, in request order.
pub fn build_request(tok: &dyn Tokenize, cfg: &LayaConfig, req: &Request) -> Result<Vec<Sequence>> {
    let state = state_ids(tok, cfg, &req.state);
    req.questions
        .iter()
        .map(|(id, q)| {
            build_sequence(tok, cfg, &state, q, None)
                .map_err(|e| anyhow::anyhow!("question {id:?}: {e}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_separators_and_escapes() {
        let v = json!({"a": 1, "b": [1.5, "x\"y", null, true], "日本": "語\n"});
        assert_eq!(
            py_json(&v, false),
            "{\"a\": 1, \"b\": [1.5, \"x\\\"y\", null, true], \"日本\": \"語\\n\"}"
        );
        assert_eq!(py_json(&json!("日"), true), "\"\\u65e5\"");
        assert_eq!(py_json(&json!(2.0), false), "2.0");
        assert_eq!(py_json(&json!(1e16), false), "1e+16");
        assert_eq!(py_json(&json!(1e-7), false), "1e-07");
    }

    #[test]
    fn options_render_like_upstream() {
        let q: Question = serde_json::from_value(json!({"type": "choice", "instructions": "i",
            "criteria": {"a": "desc", "b": null, "c": "", "d": {"k": 1}}}))
        .unwrap();
        assert_eq!(
            render_options(&q),
            vec!["a: desc", "b", "c", "d: {\"k\": 1}"]
        );
        let q: Question =
            serde_json::from_value(json!({"type": "noul", "instructions": "i"})).unwrap();
        assert_eq!(
            render_options(&q),
            vec![
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        let q: Question = serde_json::from_value(json!({"type": "score", "instructions": "i",
            "criteria": ["low", "high"]}))
        .unwrap();
        assert_eq!(render_options(&q), vec!["level 0: low", "level 1: high"]);
    }
}
