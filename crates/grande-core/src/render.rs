//! Turn a request into a shared prefix and one branch per question.
//!
//! The renderer is the single place that decides what the model sees. Training
//! data and live requests must go through the same code so the model never
//! meets a layout at inference that it did not see in training.

use indexmap::IndexMap;
use serde_json::Value;

use crate::api::{content_text, state_text, Question, Request};

/// A piece of text or a control token. Backends tokenize `Text` without
/// parsing specials, and resolve `Special` by name, so user text can never
/// forge a delimiter.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    Bos,
    Text(String),
    Special(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

/// Which token of a segment the readout wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    /// Last token of the `i`-th option's closing segment (pointer readout).
    OptEnd(usize),
    /// The `<decide>` token (pointer readout).
    Decide,
    /// Last token of the branch (label readout).
    Last,
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct Rendered {
    pub prefix: Vec<Segment>,
    pub branches: Vec<RenderedBranch>,
}

/// Delimiter control tokens for the pointer layout. Defaults reuse Gemma's
/// reserved `<unused*>` tokens so no embedding rows have to be added.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub enum Renderer {
    /// Zero-shot layout for an instruction-tuned chat model: the question and
    /// lettered options go in a user turn, the model turn is opened, and the
    /// readout looks at the next-token logits over the letters.
    Label {
        turn_start: String,
        turn_end: String,
        user: String,
        model: String,
    },
    /// Packed layout with reserved delimiters and a trained pointer head.
    Pointer(Delimiters),
}

impl Renderer {
    /// Gemma 4 chat layout: `<bos><|turn>user\n…<turn|>\n<|turn>model\n`.
    /// Thinking stays off (no `<|think|>` in a system turn), so the first
    /// model token is the answer.
    pub fn gemma_label() -> Self {
        Renderer::Label {
            turn_start: "<|turn>".into(),
            turn_end: "<turn|>".into(),
            user: "user".into(),
            model: "model".into(),
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
            Renderer::Label {
                turn_start, user, ..
            } => vec![
                Segment::Bos,
                Segment::Special(turn_start.clone()),
                Segment::Text(format!("{user}\nState:\n{state}\n\n")),
            ],
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

    fn branch(&self, id: &str, q: &Question, order: Option<&[usize]>) -> RenderedBranch {
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
                turn_start,
                turn_end,
                model,
                ..
            } => {
                let mut text = format!("Question: {instr}\n");
                for (i, (key, desc)) in ordered.iter().enumerate() {
                    let letter = crate::readout::LABELS[i];
                    match desc {
                        Some(d) if !d.is_empty() => {
                            text.push_str(&format!("{letter}: {key} — {d}\n"))
                        }
                        _ => text.push_str(&format!("{letter}: {key}\n")),
                    }
                }
                text.push_str("Answer with one letter.");
                segments.push(Segment::Text(text));
                segments.push(Segment::Special(turn_end.clone()));
                segments.push(Segment::Text("\n".into()));
                segments.push(Segment::Special(turn_start.clone()));
                segments.push(Segment::Text(format!("{model}\n")));
                marks.push((segments.len() - 1, Mark::Last));
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
