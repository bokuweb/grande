//! Request / response shapes compatible with TypeSafe's `POST /v1/systemone`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Error, Result};

/// A question sent with the request. Tagged by `type` like the TypeSafe API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<IndexMap<String, Option<Value>>>,
    },
    Choice {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<Value>,
        criteria: IndexMap<String, Option<Value>>,
    },
    Score {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        instructions: Option<Value>,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    #[serde(default = "default_model")]
    pub model: String,
    pub state: Value,
    pub questions: IndexMap<String, Question>,
}

fn default_model() -> String {
    "grande-latest".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        legend: IndexMap<String, Value>,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
}

/// Maximum number of options a Choice may carry. The pointer readout has no
/// intrinsic limit; the label readout is capped by [`crate::readout::LABELS`].
pub const MAX_CHOICE: usize = 255;
pub const MAX_SCORE: usize = 10;

impl Request {
    /// Validate the request the way the hosted API would (422 on failure).
    pub fn validate(&self) -> Result<()> {
        if !(self.state.is_string() || self.state.is_object() || self.state.is_array()) {
            return Err(Error::invalid("state", "expected string, object or array"));
        }
        if self.questions.is_empty() {
            return Err(Error::invalid(
                "questions",
                "expected a nonempty question map",
            ));
        }
        for (id, q) in &self.questions {
            let path = format!("questions.{id}");
            match q {
                Question::Choice { criteria, .. } => {
                    if criteria.is_empty() || criteria.len() > MAX_CHOICE {
                        return Err(Error::invalid(
                            format!("{path}.criteria"),
                            format!("choice requires 1..{MAX_CHOICE} options"),
                        ));
                    }
                }
                Question::Score { criteria, .. } => {
                    if criteria.len() < 2 || criteria.len() > MAX_SCORE {
                        return Err(Error::invalid(
                            format!("{path}.criteria"),
                            format!("score requires 2..{MAX_SCORE} ordered levels"),
                        ));
                    }
                }
                Question::Noul {
                    instructions,
                    criteria,
                } => {
                    if let Some(c) = criteria {
                        if !c.keys().all(|k| k == "true" || k == "false") {
                            return Err(Error::invalid(
                                format!("{path}.criteria"),
                                "only true and false keys are allowed",
                            ));
                        }
                    }
                    if instructions.is_none() && criteria.as_ref().is_none_or(|c| c.is_empty()) {
                        return Err(Error::invalid(
                            path,
                            "noul requires instructions or criteria",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Flatten structured content (string / object / array) to text the model reads.
/// Strings pass through; everything else is compact JSON with unicode kept.
pub fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Render `state` as a block of labelled lines when it is an object, so
/// questions can refer to keys by name; otherwise as [`content_text`].
pub fn state_text(v: &Value) -> String {
    match v {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| format!("{k}: {}", content_text(v)))
            .collect::<Vec<_>>()
            .join("\n"),
        other => content_text(other),
    }
}
