use grande_core::calibration::{brier, ece, fit_temperature, nll, Labeled};
use grande_core::math::{argmax, softmax};
use serde::Serialize;

/// Per-record result persisted as one JSON line.
#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub id: String,
    pub gold: usize,
    pub logits: Vec<f32>,
    pub pred: usize,
    pub candidate_mass: Option<f64>,
    pub ms: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub n: usize,
    pub accuracy: f64,
    pub nll: f64,
    pub brier: f64,
    pub ece: f64,
    pub mean_confidence: f64,
    /// Share of answers with max probability >= 0.9 that were wrong.
    pub confident_error_rate: f64,
}

pub fn metrics(rows: &[Row], t: f32) -> Metrics {
    let data: Vec<Labeled> = rows
        .iter()
        .map(|r| Labeled {
            logits: r.logits.clone(),
            label: r.gold,
        })
        .collect();
    let n = rows.len();
    if n == 0 {
        return Metrics {
            n,
            accuracy: 0.0,
            nll: 0.0,
            brier: 0.0,
            ece: 0.0,
            mean_confidence: 0.0,
            confident_error_rate: 0.0,
        };
    }
    let mut correct = 0usize;
    let mut conf_sum = 0.0;
    let mut confident = 0usize;
    let mut confident_wrong = 0usize;
    for r in rows {
        let p = softmax(&r.logits, t);
        let a = argmax(&p);
        let ok = a == r.gold;
        correct += usize::from(ok);
        conf_sum += p[a];
        if p[a] >= 0.9 {
            confident += 1;
            confident_wrong += usize::from(!ok);
        }
    }
    Metrics {
        n,
        accuracy: correct as f64 / n as f64,
        nll: nll(&data, t),
        brier: brier(&data, t),
        ece: ece(&data, t, 10),
        mean_confidence: conf_sum / n as f64,
        confident_error_rate: if confident == 0 {
            0.0
        } else {
            confident_wrong as f64 / confident as f64
        },
    }
}

/// Split rows into calibration (even index) and test (odd index), fit the
/// temperature on calibration, report test before and after.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub temperature: f32,
    pub calibration_n: usize,
    pub test_raw: Metrics,
    pub test_scaled: Metrics,
    pub all_raw: Metrics,
    pub mean_ms: f64,
}

pub fn summarize(rows: &[Row]) -> Summary {
    let calib: Vec<Labeled> = rows
        .iter()
        .step_by(2)
        .map(|r| Labeled {
            logits: r.logits.clone(),
            label: r.gold,
        })
        .collect();
    let test: Vec<Row> = rows.iter().skip(1).step_by(2).cloned().collect();
    let t = if calib.is_empty() {
        1.0
    } else {
        fit_temperature(&calib)
    };
    Summary {
        temperature: t,
        calibration_n: calib.len(),
        test_raw: metrics(&test, 1.0),
        test_scaled: metrics(&test, t),
        all_raw: metrics(rows, 1.0),
        mean_ms: rows.iter().map(|r| r.ms as f64).sum::<f64>() / rows.len().max(1) as f64,
    }
}
