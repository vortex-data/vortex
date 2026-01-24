use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;

/// Git commit metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitInfo {
    pub hash: String,
    pub timestamp: DateTime<Utc>,
    pub message: Option<String>,
    pub author: Option<String>,
}

/// A single data point on a chart.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataPoint {
    pub commit_idx: usize,
    pub value_ns: u64,
}

impl DataPoint {
    pub fn value_ms(&self) -> f64 {
        self.value_ns as f64 / 1_000_000.0
    }
}

/// A series of data points for one benchmark target.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Series {
    pub name: String,
    pub display_name: String,
    pub color: String,
    pub points: Vec<DataPoint>,
}

/// Complete chart data for rendering a benchmark chart.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChartData {
    pub title: String,
    pub commits: Vec<CommitInfo>,
    pub series: Vec<Series>,
    pub unit: String,
}

impl ChartData {
    /// Find min/max Y values across all series (in ms).
    pub fn y_range(&self) -> (f64, f64) {
        let values: Vec<f64> = self
            .series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.value_ms()))
            .collect();

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(0.0, f64::max);
        (min.min(0.0), max * 1.1) // Start at 0, add 10% headroom
    }
}
