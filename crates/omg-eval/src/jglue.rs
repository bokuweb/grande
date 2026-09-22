use anyhow::{anyhow, Context, Result};
use indexmap::IndexMap;
use omg_core::Request;
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
        omg_core::Question::Score {
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
        omg_core::Question::Choice {
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

/// Few-shot examples for a task: `n` records from `train` (fixed seed,
/// labels balanced by round robin) rendered as a block of text that goes
/// under an `例` key in front of every state. Labels are written as the
/// option key plus its Japanese name so the model can map them to the
/// lettered options; JCQA shows the answer string.
pub fn shots(task: Task, train: &Path, n: usize, seed: u64) -> Result<String> {
    let text =
        std::fs::read_to_string(train).with_context(|| format!("reading {}", train.display()))?;
    let rows: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    // Deterministic shuffle (splitmix64) so runs with the same seed compare.
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
    for i in (1..idx.len()).rev() {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        idx.swap(i, (z % (i as u64 + 1)) as usize);
    }
    let classes = match task {
        Task::Jnli => 3,
        Task::Jcqa => 5,
        Task::Jsts => 6,
    };
    let mut picked: Vec<&Value> = Vec::new();
    let mut want = 0usize;
    while picked.len() < n {
        let before = picked.len();
        for &i in &idx {
            let row = &rows[i];
            let class = match task {
                Task::Jnli => JNLI_LABELS
                    .iter()
                    .position(|l| Some(*l) == row["label"].as_str())
                    .unwrap_or(0),
                Task::Jcqa => row["label"].as_u64().unwrap_or(0) as usize,
                Task::Jsts => row["label"].as_f64().unwrap_or(0.0).round() as usize,
            };
            if class == want % classes && !picked.iter().any(|p| std::ptr::eq(*p, row)) {
                picked.push(row);
                want += 1;
                break;
            }
        }
        if picked.len() == before {
            want += 1; // class exhausted
            if want > n * classes {
                break;
            }
        }
    }
    let s = |row: &Value, k: &str| row[k].as_str().unwrap_or("").to_string();
    let blocks: Vec<String> = picked
        .iter()
        .map(|row| match task {
            Task::Jnli => {
                let label = s(row, "label");
                let ja = match label.as_str() {
                    "entailment" => "含意",
                    "contradiction" => "矛盾",
                    _ => "中立",
                };
                format!(
                    "前提: {}\n仮説: {}\n判定: {label}（{ja}）",
                    s(row, "sentence1"),
                    s(row, "sentence2")
                )
            }
            Task::Jcqa => {
                let gold = row["label"].as_u64().unwrap_or(0) as usize;
                let choices: Vec<String> = (0..5).map(|i| s(row, &format!("choice{i}"))).collect();
                format!(
                    "質問: {}\n選択肢: {}\n答え: {}",
                    s(row, "question"),
                    choices.join(" / "),
                    choices[gold]
                )
            }
            Task::Jsts => format!(
                "文1: {}\n文2: {}\n類似度: {}",
                s(row, "sentence1"),
                s(row, "sentence2"),
                row["label"].as_f64().unwrap_or(0.0).round() as usize
            ),
        })
        .collect();
    Ok(blocks.join("\n\n"))
}

/// Put the few-shot block in front of every item's state (as `例`).
pub fn with_shots(items: &mut [Item], block: &str) {
    for item in items {
        let mut m = serde_json::Map::new();
        m.insert("例".to_string(), json!(block));
        if let Value::Object(old) = &item.request.state {
            for (k, v) in old {
                m.insert(k.clone(), v.clone());
            }
        }
        item.request.state = Value::Object(m);
    }
}
