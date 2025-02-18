use std::time::Duration;

use serde::Serialize;

use crate::Format;

pub trait ToJson {
    fn to_json(&self) -> JsonValue;
}

pub trait ToGeneric {
    fn to_generic(&self) -> GenericMeasurement;
}

#[derive(Serialize)]
pub struct JsonValue {
    pub name: String,
    pub storage: String,
    pub unit: String,
    pub value: u128,
    pub commit_id: String,
}

#[derive(Clone, Debug)]
pub struct GenericMeasurement {
    pub id: usize,
    pub name: String,
    pub storage: String,
    pub format: Format,
    pub time: Duration,
}

impl ToGeneric for GenericMeasurement {
    fn to_generic(&self) -> GenericMeasurement {
        self.clone()
    }
}

impl ToJson for GenericMeasurement {
    fn to_json(&self) -> JsonValue {
        JsonValue {
            name: self.name.clone(),
            storage: self.storage.clone(),
            unit: "ns".to_string(),
            value: self.time.as_nanos(),
            commit_id: crate::GIT_COMMIT_ID.to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QueryMeasurement {
    pub query_idx: usize,
    /// The storage backend against which this test was run. One of: s3, gcs, nvme.
    pub storage: String,
    pub time: Duration,
    pub format: Format,
    pub dataset: String,
}

impl ToJson for QueryMeasurement {
    fn to_json(&self) -> JsonValue {
        let name = format!(
            "{dataset}_q{query_idx:02}/{format}",
            dataset = self.dataset,
            format = self.format.name(),
            query_idx = self.query_idx
        );

        JsonValue {
            name,
            storage: self.storage.clone(),
            unit: "ns".to_string(),
            value: self.time.as_nanos(),
            commit_id: crate::GIT_COMMIT_ID.to_string(),
        }
    }
}

impl ToGeneric for QueryMeasurement {
    fn to_generic(&self) -> GenericMeasurement {
        GenericMeasurement {
            id: self.query_idx,
            name: format!(
                "{dataset}_q{query_idx:02}_{storage}/{format}",
                dataset = self.dataset,
                format = self.format.name(),
                query_idx = self.query_idx,
                storage = self.storage,
            ),
            storage: self.storage.clone(),
            format: self.format,
            time: self.time,
        }
    }
}
