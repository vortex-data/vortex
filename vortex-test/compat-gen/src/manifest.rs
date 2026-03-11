use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Manifest listing all fixtures generated for a given version.
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: String,
    pub generated_at: DateTime<Utc>,
    pub fixtures: Vec<String>,
}
