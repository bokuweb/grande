//! Turn a request into a shared prefix and one branch per question.
//!
//! The renderer is the single place that decides what the model sees. Training
//! data and live requests must go through the same code so the model never
//! meets a layout at inference that it did not see in training.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::{content_text, state_text, Question, Request};

/// A piece of text or a control token. Backends tokenize `Text` without
/// parsing specials, and resolve `Special` by name, so user text can never
/// forge a delimiter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "lowercase")]
pub enum Segment {
    Bos,
    Text(String),
    Special(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

/// Which token of a segment the readout wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mark {
    /// Last token of the `i`-th option's closing segment (pointer readout).
    OptEnd(usize),
    /// The `<decide>` token (pointer readout).
    Decide,
    /// Last token of the branch (label readout).
    Last,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderedBranch {
    pub id: String,
    pub kind: Kind,
    /// Option keys in rendered order (Choice keys, Score level indices as
    /// strings, `true`/`false` for Noul). `order[i]` is the original index.
    pub keys: Vec<String>,
    pub order: Vec<usize>,
    pub segments: Vec<Segment>,
    /// `(segment index, mark)` pairs; the wanted token is the last token of
    /// that segment.
    pub marks: Vec<(usize, Mark)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rendered {
    pub prefix: Vec<Segment>,
    pub branches: Vec<RenderedBranch>,
}

/// Delimiter control tokens for the pointer layout. Defaults reuse Gemma's
/// reserved `<unused*>` tokens so no embedding rows have to be added.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delimiters {
    pub state: String,
    pub question: String,
    pub opt: String,
    pub opt_end: String,
    pub decide: String,
}

impl Default for Delimiters {
    fn default() -> Self {
        Delimiters {
            state: "<unused0>".into(),
            question: "<unused1>".into(),
            opt: "<unused2>".into(),
            opt_end: "<unused3>".into(),
            decide: "<unused4>".into(),
        }
    }
}

/// How the request is laid out for the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "layout", rename_all = "lowercase")]
pub enum Renderer {
    /// Zero-shot layout for an instruction-tuned chat model: the question and
    /// lettered options go in a user turn, the model turn is opened, and the
    /// readout looks at the next-token logits over the letters.
    Label {
        /// Model family the chat template belongs to (`gemma`, `qwen`,
        /// `deepseek`); names the layout in run metadata.
        #[serde(default = "default_family")]
        family: String,
        /// Everything before the state: BOS if the tokenizer prepends one,
        /// the user-turn opener and role line.
        user_open: Vec<Segment>,
        /// Everything after the question: closes the user turn, opens the
        /// model turn, and prefills what the template writes before the
        /// first answer token (an empty thinking block on Qwen 3.5 and
        /// DeepSeek R1). The readout position is the last token.
        model_open: Vec<Segment>,
        /// Drop the "Question:" / "Answer with one letter." scaffolding: the
        /// branch is the instruction, the lettered options and the model
        /// turn. Fewer tokens per question; the model turn already asks for
        /// the answer.
        #[serde(default)]
        terse: bool,
        /// Same prompt, read with a pointer head instead of the letter
        /// logits: every option line's last token is marked `OptEnd` and the
        /// model-turn position `Decide`. The backbone stays as it is (frozen,
        /// quantized); only the head is trained, on hidden states this
        /// engine produced.
        #[serde(default)]
        pointer: bool,
    },
    /// Packed layout with reserved delimiters and a trained pointer head.
    Pointer(Delimiters),
}

fn default_family() -> String {
    "gemma".into()
}

impl Renderer {
    /// A `<turn_start>role\n … <turn_end>\n` chat layout (Gemma, ChatML).
    fn turns(
        family: &str,
        bos: bool,
        turn_start: &str,
        turn_end: &str,
        user: &str,
        model: &str,
    ) -> Self {
        let mut user_open = Vec::new();
        if bos {
            user_open.push(Segment::Bos);
        }
        user_open.push(Segment::Special(turn_start.into()));
        user_open.push(Segment::Text(format!("{user}\n")));
        Renderer::Label {
            family: family.into(),
            user_open,
            model_open: vec![
                Segment::Special(turn_end.into()),
                Segment::Text("\n".into()),
                Segment::Special(turn_start.into()),
                Segment::Text(format!("{model}\n")),
            ],
            terse: false,
            pointer: false,
        }
    }

    /// Gemma 4 chat layout: `<bos><|turn>user\n…<turn|>\n<|turn>model\n`.
    /// Thinking stays off (no `<|think|>` in a system turn), so the first
    /// model token is the answer.
    pub fn gemma_label() -> Self {
        Self::turns("gemma", true, "<|turn>", "<turn|>", "user", "model")
    }

    /// Gemma 3 chat layout (`<start_of_turn>` / `<end_of_turn>`).
    pub fn gemma3_label() -> Self {
        Self::turns(
            "gemma",
            true,
            "<start_of_turn>",
            "<end_of_turn>",
            "user",
            "model",
        )
    }

    /// The empty thinking block a reasoning model's template writes when
    /// thinking is off, so the first answer token is the letter.
    fn no_think() -> [Segment; 4] {
        [
            Segment::Special("<think>".into()),
            Segment::Text("\n\n".into()),
            Segment::Special("</think>".into()),
            Segment::Text("\n\n".into()),
        ]
    }

    /// Qwen 3.5 chat layout (ChatML): `<|im_start|>user\n…<|im_end|>\n
    /// <|im_start|>assistant\n<think>\n\n</think>\n\n`, what the chat
    /// template writes with `enable_thinking` off. No BOS: Qwen's tokenizer
    /// does not prepend one.
    pub fn qwen_label() -> Self {
        let mut r = Self::turns(
            "qwen",
            false,
            "<|im_start|>",
            "<|im_end|>",
            "user",
            "assistant",
        );
        if let Renderer::Label { model_open, .. } = &mut r {
            model_open.extend(Self::no_think());
        }
        r
    }

    /// DeepSeek R1 (and its distills) chat layout:
    /// `<｜begin▁of▁sentence｜><｜User｜>…<｜Assistant｜><think>\n\n</think>\n\n`.
    /// The template opens `<think>\n` for the model; the closed empty block
    /// skips the reasoning so the next token is the answer.
    pub fn deepseek_label() -> Self {
        Renderer::Label {
            family: "deepseek".into(),
            user_open: vec![Segment::Bos, Segment::Special("<｜User｜>".into())],
            model_open: std::iter::once(Segment::Special("<｜Assistant｜>".into()))
                .chain(Self::no_think())
                .collect(),
            terse: false,
            pointer: false,
        }
    }

    /// The same layout with the terse branch template.
    pub fn terse(mut self, on: bool) -> Self {
        if let Renderer::Label { terse, .. } = &mut self {
            *terse = on;
        }
        self
    }

    /// The same layout read by a pointer head (see [`Renderer::Label`]).
    pub fn pointer(mut self, on: bool) -> Self {
        if let Renderer::Label { pointer, .. } = &mut self {
            *pointer = on;
        }
        self
    }

    /// Short name for logs and summaries (`gemma_label`, `qwen_label_terse`,
    /// `gemma_label_pointer`, `gemma_pointer`).
    pub fn layout_name(&self) -> String {
        match self {
            Renderer::Label {
                family,
                pointer: true,
                ..
            } => format!("{family}_label_pointer"),
            Renderer::Label {
                family,
                terse: true,
                ..
            } => format!("{family}_label_terse"),
            Renderer::Label { family, .. } => format!("{family}_label"),
            Renderer::Pointer(_) => "gemma_pointer".into(),
        }
    }

    pub fn gemma_pointer() -> Self {
        Renderer::Pointer(Delimiters::default())
    }

    /// Render with options in their request order.
    pub fn render(&self, req: &Request) -> Rendered {
        self.render_with(req, &IndexMap::new())
    }

    /// Render with an explicit option permutation for some questions
    /// (`orders[qid][i]` = original index of the option rendered at slot `i`).
    pub fn render_with(&self, req: &Request, orders: &IndexMap<String, Vec<usize>>) -> Rendered {
        let state = state_text(&req.state);
        let prefix = match self {
            Renderer::Label { user_open, .. } => coalesce(
                user_open
                    .iter()
                    .cloned()
                    .chain([Segment::Text(format!("State:\n{state}\n\n"))]),
            ),
            Renderer::Pointer(d) => vec![
                Segment::Bos,
                Segment::Special(d.state.clone()),
                Segment::Text(state),
            ],
        };
        let branches = req
            .questions
            .iter()
            .map(|(id, q)| self.branch(id, q, orders.get(id).map(Vec::as_slice)))
            .collect();
        Rendered { prefix, branches }
    }

    /// One branch for question `id` with the options at `order` (original
    /// indices, one per slot; a subset renders only those options).
    pub fn branch(&self, id: &str, q: &Question, order: Option<&[usize]>) -> RenderedBranch {
        let (kind, instr, options) = options_of(q);
        let order: Vec<usize> = match order {
            Some(o) => o.to_vec(),
            None => (0..options.len()).collect(),
        };
        let ordered: Vec<&(String, Option<String>)> = order.iter().map(|&i| &options[i]).collect();
        let keys = ordered.iter().map(|(k, _)| k.clone()).collect();
        let mut segments = Vec::new();
        let mut marks = Vec::new();
        match self {
            Renderer::Label {
                model_open,
                terse,
                pointer,
                ..
            } => {
                let mut text = if *terse {
                    format!("{instr}\n")
                } else {
                    format!("Question: {instr}\n")
                };
                for (i, (key, desc)) in ordered.iter().enumerate() {
                    // Past the last letter the branch cannot be read out; the
                    // engine splits such a Choice into groups before
                    // evaluating, so this render is only ever looked at for
                    // its keys and order.
                    let letter = crate::readout::LABELS
                        .get(i)
                        .map_or_else(|| i.to_string(), char::to_string);
                    match desc {
                        Some(d) if !d.is_empty() => {
                            text.push_str(&format!("{letter}: {key} — {d}"))
                        }
                        _ => text.push_str(&format!("{letter}: {key}")),
                    }
                    if *pointer {
                        // Each option line is its own segment so its last
                        // token can be read; the newline goes in the next
                        // segment so the read token is content, not "\n".
                        segments.push(Segment::Text(std::mem::take(&mut text)));
                        marks.push((segments.len() - 1, Mark::OptEnd(i)));
                    }
                    text.push('\n');
                }
                if *terse {
                    text.pop();
                } else {
                    text.push_str("Answer with one letter.");
                }
                segments.push(Segment::Text(text));
                segments.extend(model_open.iter().cloned());
                marks.push((
                    segments.len() - 1,
                    if *pointer { Mark::Decide } else { Mark::Last },
                ));
            }
            Renderer::Pointer(d) => {
                segments.push(Segment::Special(d.question.clone()));
                segments.push(Segment::Text(instr));
                for (i, (key, desc)) in ordered.iter().enumerate() {
                    segments.push(Segment::Special(d.opt.clone()));
                    let text = match desc {
                        Some(x) if !x.is_empty() => format!("{key} — {x}"),
                        _ => key.clone(),
                    };
                    segments.push(Segment::Text(text));
                    segments.push(Segment::Special(d.opt_end.clone()));
                    marks.push((segments.len() - 1, Mark::OptEnd(i)));
                }
                segments.push(Segment::Special(d.decide.clone()));
                marks.push((segments.len() - 1, Mark::Decide));
            }
        }
        RenderedBranch {
            id: id.to_string(),
            kind,
            keys,
            order,
            segments,
            marks,
        }
    }
}

/// Merge adjacent text segments so the role line and the state tokenize as
/// one string, exactly as before the opener became a segment list.
fn coalesce(segments: impl IntoIterator<Item = Segment>) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for s in segments {
        match (out.last_mut(), s) {
            (Some(Segment::Text(a)), Segment::Text(b)) => a.push_str(&b),
            (_, s) => out.push(s),
        }
    }
    out
}

/// `(kind, instructions text, [(key, description)])` for a question.
fn options_of(q: &Question) -> (Kind, String, Vec<(String, Option<String>)>) {
    let text = |v: &Option<Value>| v.as_ref().map(content_text);
    match q {
        Question::Choice {
            instructions,
            criteria,
        } => (
            Kind::Choice,
            text(instructions).unwrap_or_else(|| "Choose the best matching option.".into()),
            criteria.iter().map(|(k, d)| (k.clone(), text(d))).collect(),
        ),
        Question::Score {
            instructions,
            criteria,
        } => (
            Kind::Score,
            text(instructions).unwrap_or_else(|| "Select the most appropriate level.".into()),
            criteria
                .iter()
                .enumerate()
                .map(|(i, d)| (i.to_string(), Some(content_text(d))))
                .collect(),
        ),
        Question::Noul {
            instructions,
            criteria,
        } => {
            let c = criteria.clone().unwrap_or_default();
            let get = |k: &str| c.get(k).and_then(|v| v.as_ref().map(content_text));
            (
                Kind::Noul,
                text(instructions).unwrap_or_else(|| "Is the statement true?".into()),
                vec![("true".into(), get("true")), ("false".into(), get("false"))],
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req() -> Request {
        serde_json::from_value(json!({
            "state": {"ticket": "二重請求です。返金してください。"},
            "questions": {
                "refund": {"type": "noul", "instructions": "返金を要求しているか"},
                "dept": {"type": "choice", "instructions": "担当部署は？",
                         "criteria": {"billing": "請求・返金", "technical": null}},
                "urgency": {"type": "score", "instructions": "緊急度は？",
                            "criteria": ["通常", "早め", "即時"]}
            }
        }))
        .unwrap()
    }

    #[test]
    fn label_layout_marks_last_token_and_letters_options() {
        let r = Renderer::gemma_label().render(&req());
        assert_eq!(r.branches.len(), 3);
        let dept = &r.branches[1];
        assert_eq!(dept.keys, vec!["billing", "technical"]);
        assert_eq!(dept.marks, vec![(dept.segments.len() - 1, Mark::Last)]);
        let Segment::Text(t) = &dept.segments[0] else {
            panic!()
        };
        assert!(t.contains("A: billing — 請求・返金\nB: technical\n"));
        let Segment::Text(p) = &r.prefix[2] else {
            panic!()
        };
        assert!(p.contains("ticket: 二重請求です。返金してください。"));
    }

    #[test]
    fn label_pointer_layout_keeps_the_prompt_and_marks_option_ends() {
        let plain = Renderer::gemma_label().render(&req());
        let r = Renderer::gemma_label().pointer(true).render(&req());
        let dept = &r.branches[1];
        // Same text as the label layout, split so each option line's last
        // token and the model-turn position can be read.
        let text = |b: &RenderedBranch| {
            b.segments
                .iter()
                .map(|s| match s {
                    Segment::Text(t) => t.clone(),
                    Segment::Special(t) => t.clone(),
                    Segment::Bos => String::new(),
                })
                .collect::<String>()
        };
        assert_eq!(text(dept), text(&plain.branches[1]));
        assert_eq!(
            dept.marks,
            vec![
                (0, Mark::OptEnd(0)),
                (1, Mark::OptEnd(1)),
                (dept.segments.len() - 1, Mark::Decide)
            ]
        );
        let Segment::Text(t) = &dept.segments[0] else {
            panic!()
        };
        assert!(t.ends_with("A: billing — 請求・返金"));
        let Segment::Text(t) = &dept.segments[1] else {
            panic!()
        };
        assert_eq!(t, "\nB: technical");
        assert_eq!(
            Renderer::gemma_label().pointer(true).layout_name(),
            "gemma_label_pointer"
        );
    }

    #[test]
    fn gemma_prefix_is_one_text_after_the_opener() {
        let r = Renderer::gemma_label().render(&req());
        assert_eq!(r.prefix.len(), 3);
        assert_eq!(r.prefix[0], Segment::Bos);
        assert_eq!(r.prefix[1], Segment::Special("<|turn>".into()));
        let Segment::Text(p) = &r.prefix[2] else {
            panic!()
        };
        assert!(p.starts_with("user\nState:\nticket: "));
        let dept = &r.branches[1];
        let n = dept.segments.len();
        assert_eq!(
            &dept.segments[n - 4..],
            &[
                Segment::Special("<turn|>".into()),
                Segment::Text("\n".into()),
                Segment::Special("<|turn>".into()),
                Segment::Text("model\n".into()),
            ]
        );
    }

    #[test]
    fn qwen_layout_has_no_bos_and_prefills_an_empty_thinking_block() {
        let r = Renderer::qwen_label().render(&req());
        assert_eq!(r.prefix[0], Segment::Special("<|im_start|>".into()));
        assert!(matches!(&r.prefix[1], Segment::Text(t) if t.starts_with("user\nState:\n")));
        let dept = &r.branches[1];
        let n = dept.segments.len();
        assert_eq!(dept.marks, vec![(n - 1, Mark::Last)]);
        assert_eq!(
            &dept.segments[n - 8..],
            &[
                Segment::Special("<|im_end|>".into()),
                Segment::Text("\n".into()),
                Segment::Special("<|im_start|>".into()),
                Segment::Text("assistant\n".into()),
                Segment::Special("<think>".into()),
                Segment::Text("\n\n".into()),
                Segment::Special("</think>".into()),
                Segment::Text("\n\n".into()),
            ]
        );
        assert_eq!(Renderer::qwen_label().layout_name(), "qwen_label");
        assert_eq!(
            Renderer::qwen_label().terse(true).layout_name(),
            "qwen_label_terse"
        );
    }

    #[test]
    fn deepseek_layout_has_bos_and_no_role_lines() {
        let r = Renderer::deepseek_label().render(&req());
        assert_eq!(r.prefix[0], Segment::Bos);
        assert_eq!(r.prefix[1], Segment::Special("<｜User｜>".into()));
        assert!(matches!(&r.prefix[2], Segment::Text(t) if t.starts_with("State:\n")));
        let dept = &r.branches[1];
        let n = dept.segments.len();
        assert_eq!(dept.marks, vec![(n - 1, Mark::Last)]);
        assert_eq!(
            dept.segments[n - 5],
            Segment::Special("<｜Assistant｜>".into())
        );
        assert_eq!(dept.segments[n - 4], Segment::Special("<think>".into()));
        assert_eq!(Renderer::deepseek_label().layout_name(), "deepseek_label");
        assert_eq!(
            Renderer::deepseek_label().pointer(true).layout_name(),
            "deepseek_label_pointer"
        );
    }

    #[test]
    fn label_layout_json_defaults_to_the_gemma_family() {
        // The browser passes the layout as JSON; `family` may be omitted.
        let j: Renderer = serde_json::from_value(json!({
            "layout": "label",
            "user_open": [{"kind": "bos"}, {"kind": "special", "value": "<|turn>"},
                          {"kind": "text", "value": "user\n"}],
            "model_open": [{"kind": "special", "value": "<turn|>"}, {"kind": "text", "value": "\n"},
                           {"kind": "special", "value": "<|turn>"}, {"kind": "text", "value": "model\n"}]
        }))
        .unwrap();
        assert_eq!(j.layout_name(), "gemma_label");
        let a = Renderer::gemma_label().render(&req());
        let b = j.render(&req());
        assert_eq!(a.prefix, b.prefix);
        assert_eq!(a.branches[1].segments, b.branches[1].segments);
    }

    #[test]
    fn pointer_layout_marks_every_option_and_decide() {
        let r = Renderer::gemma_pointer().render(&req());
        let urgency = &r.branches[2];
        assert_eq!(urgency.keys, vec!["0", "1", "2"]);
        let kinds: Vec<Mark> = urgency.marks.iter().map(|(_, m)| *m).collect();
        assert_eq!(
            kinds,
            vec![
                Mark::OptEnd(0),
                Mark::OptEnd(1),
                Mark::OptEnd(2),
                Mark::Decide
            ]
        );
        assert!(matches!(urgency.segments.last(), Some(Segment::Special(s)) if s == "<unused4>"));
    }

    #[test]
    fn permutation_reorders_keys_and_keeps_origin() {
        let mut orders = IndexMap::new();
        orders.insert("dept".to_string(), vec![1, 0]);
        let r = Renderer::gemma_label().render_with(&req(), &orders);
        assert_eq!(r.branches[1].keys, vec!["technical", "billing"]);
        assert_eq!(r.branches[1].order, vec![1, 0]);
    }

    #[test]
    fn noul_renders_true_false() {
        let r = Renderer::gemma_pointer().render(&req());
        assert_eq!(r.branches[0].keys, vec!["true", "false"]);
        assert_eq!(r.branches[0].kind, Kind::Noul);
    }
}
