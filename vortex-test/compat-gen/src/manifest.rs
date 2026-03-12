use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;

/// Manifest listing all fixtures generated for a given version.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: String,
    pub generated_at: DateTime<Utc>,
    pub fixtures: Vec<String>,
}
