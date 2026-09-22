//! kev-style frozen suites: TypeSafe-shaped records with a gold `label` on
//! every question and `_meta.source` / `_meta.variant` for grouping.
//! Lets omg be scored on exactly the questions kev and Jev were scored on.

use anyhow::{anyhow, Context, Result};
use indexmap::IndexMap;
use omg_core::{Question, Request};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SuiteRecord {
    pub id: String,
    pub source: String,
    pub variant: String,
    pub request: Request,
    /// Gold option index per question, in request order (None = unlabeled).
    pub gold: Vec<Option<usize>>,
    /// kev's task name per question (`agnews`, `agnews_yn`, ...).
    pub tasks: Vec<String>,
}

pub fn load(path: &Path) -> Result<Vec<SuiteRecord>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).with_context(|| format!("line {}", n + 1))?;
        out.push(record(v).with_context(|| format!("line {}", n + 1))?);
    }
    Ok(out)
}

fn record(v: Value) -> Result<SuiteRecord> {
    let meta = &v["_meta"];
    let source = meta["source"].as_str().unwrap_or("unknown").to_string();
    let variant = meta["variant"].as_str().unwrap_or("clean").to_string();
    let id = meta["id"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| meta["row"].to_string());
    let qs = v["questions"]
        .as_object()
        .ok_or_else(|| anyhow!("questions"))?;
    let mut questions = IndexMap::new();
    let mut gold = Vec::new();
    let mut tasks = Vec::new();
    for (qid, q) in qs {
        let mut q = q.clone();
        let label = q.as_object_mut().and_then(|m| m.remove("label"));
        if let Some(m) = q.as_object_mut() {
            m.remove("src");
        }
        let question: Question =
            serde_json::from_value(q).with_context(|| format!("question {qid}"))?;
        let g = match (&question, label) {
            (Question::Choice { criteria, .. }, Some(Value::String(s))) => {
                criteria.get_index_of(&s)
            }
            (Question::Score { .. }, Some(Value::Number(n))) => n.as_u64().map(|x| x as usize),
            (Question::Noul { .. }, Some(Value::Bool(b))) => Some(usize::from(!b)), // true → index 0
            _ => None,
        };
        let task = match &question {
            Question::Noul { .. } if source == "agnews" || source == "yelp" => {
                format!("{source}_yn")
            }
            _ => source.clone(),
        };
        gold.push(g);
        tasks.push(task);
        questions.insert(qid.clone(), question);
    }
    let request = Request {
        model: "grande-latest".into(),
        state: v["state"].clone(),
        questions,
    };
    Ok(SuiteRecord {
        id,
        source,
        variant,
        request,
        gold,
        tasks,
    })
}
