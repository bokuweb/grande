use anyhow::{anyhow, Context, Result};
use grande_core::Request;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

pub const REVISION: &str = "6f071c09316baae89c3d083a90985b4b1cb9968c";
pub const JNLI_LABELS: [&str; 3] = ["entailment", "contradiction", "neutral"];
pub const JSTS_INSTR: &str = "2つの文の意味がどの程度似ているかを判定してください。";
pub const JSTS_LEVELS: [&str; 6] = [
    "0: 全く関係がない",
    "1: ほとんど関係がない",
    "2: 一部の話題が共通する",
    "3: おおよそ同じ内容",
    "4: 細部を除いて同じ",
    "5: 完全に同じ意味",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Jnli,
    Jcqa,
    Jsts,
}

impl Task {
    pub fn name(self) -> &'static str {
        match self {
            Task::Jnli => "jnli",
            Task::Jcqa => "jcommonsenseqa",
            Task::Jsts => "jsts",
        }
    }

    pub fn url(self, split: &str) -> String {
        format!(
            "https://raw.githubusercontent.com/yahoojapan/JGLUE/{REVISION}/datasets/{}-v1.3/{split}-v1.3.json",
            self.name()
        )
    }
}

/// One JGLUE record turned into a request plus its gold option index.
#[derive(Debug, Clone)]
pub struct Item {
    pub id: String,
    pub request: Request,
    /// Index of the gold option in the request's criteria order.
    pub gold: usize,
}

pub fn load(task: Task, path: &Path) -> Result<Vec<Item>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut items = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(line).with_context(|| format!("line {}", n + 1))?;
        items.push(item(task, &row).with_context(|| format!("line {}", n + 1))?);
    }
    Ok(items)
}

fn item_jsts(row: &Value) -> Result<Item> {
    let s = |k: &str| {
        row[k]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow!("missing {k}"))
    };
    let gold = row["label"]
        .as_f64()
        .ok_or_else(|| anyhow!("label"))?
        .round()
        .clamp(0.0, 5.0) as usize;
    let mut questions = IndexMap::new();
    questions.insert(
        "answer".to_string(),
        grande_core::Question::Score {
            instructions: Some(json!(JSTS_INSTR)),
            criteria: JSTS_LEVELS.iter().map(|l| json!(l)).collect(),
        },
    );
    Ok(Item {
        id: s("sentence_pair_id")?,
        request: Request {
            model: "grande-latest".into(),
            state: json!({"文1": s("sentence1")?, "文2": s("sentence2")?}),
            questions,
        },
        gold,
    })
}

fn item(task: Task, row: &Value) -> Result<Item> {
    if task == Task::Jsts {
        return item_jsts(row);
    }
    let s = |k: &str| {
        row[k]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow!("missing {k}"))
    };
    let (id, state, instructions, criteria, gold) = match task {
        Task::Jnli => {
            let label = s("label")?;
            let gold = JNLI_LABELS
                .iter()
                .position(|l| *l == label)
                .ok_or_else(|| anyhow!("label {label}"))?;
            let criteria: IndexMap<String, Option<Value>> = JNLI_LABELS
                .iter()
                .zip([
                    "含意：前提から仮説が正しいと必ず言える",
                    "矛盾：前提から仮説が誤りだと必ず言える",
                    "中立：前提だけでは仮説が正しいとも誤りとも判断できない",
                ])
                .map(|(k, d)| (k.to_string(), Some(json!(d))))
                .collect();
            (
                s("sentence_pair_id")?,
                json!({"前提": s("sentence1")?, "仮説": s("sentence2")?}),
                "前提が正しいとき、仮説との論理的な関係を判定してください。前提から分からない情報を補わないでください。",
                criteria,
                gold,
            )
        }
        Task::Jsts => unreachable!(),
        Task::Jcqa => {
            let gold = row["label"].as_u64().ok_or_else(|| anyhow!("label"))? as usize;
            let criteria: IndexMap<String, Option<Value>> = (0..5)
                .map(|i| Ok((i.to_string(), Some(json!(s(&format!("choice{i}"))?)))))
                .collect::<Result<_>>()?;
            (
                row["q_id"].to_string(),
                json!({"質問": s("question")?}),
                "質問に対して、常識に基づく最も適切な答えを選択肢から1つ選んでください。",
                criteria,
                gold,
            )
        }
    };
    let mut questions = IndexMap::new();
    questions.insert(
        "answer".to_string(),
        grande_core::Question::Choice {
            instructions: Some(json!(instructions)),
            criteria,
        },
    );
    Ok(Item {
        id,
        request: Request {
            model: "grande-latest".into(),
            state,
            questions,
        },
        gold,
    })
}
