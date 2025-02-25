use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::ops::{Add, Div, Sub};
use std::time::Duration;

use serde::{Serialize, Serializer};
use vortex::error::vortex_panic;

use crate::{Format, GIT_COMMIT_ID};

pub trait ToJson {
    fn to_json(&self) -> JsonValue;
}

pub trait ToTable {
    fn to_table(&self) -> TableValue;
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub enum MeasurementValue {
    Int(u128),
    Float(f64),
}

impl Display for MeasurementValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasurementValue::Int(i) => write!(f, "{}", i),
            MeasurementValue::Float(fl) => match f.precision() {
                None => write!(f, "{}", fl),
                Some(p) => write!(f, "{1:.*}", p, fl),
            },
        }
    }
}

impl Serialize for MeasurementValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            MeasurementValue::Int(i) => serializer.serialize_u128(*i),
            MeasurementValue::Float(f) => serializer.serialize_f64(*f),
        }
    }
}

impl Div<MeasurementValue> for MeasurementValue {
    type Output = MeasurementValue;

    fn div(self, rhs: MeasurementValue) -> Self::Output {
        match (self, rhs) {
            (MeasurementValue::Float(a), MeasurementValue::Float(b)) => {
                MeasurementValue::Float(a / b)
            }
            (MeasurementValue::Int(a), MeasurementValue::Int(b)) => {
                MeasurementValue::Float(a as f64 / b as f64)
            }
            _ => vortex_panic!("Can't divide two measurement values of different kinds"),
        }
    }
}

impl Div<usize> for MeasurementValue {
    type Output = MeasurementValue;

    fn div(self, rhs: usize) -> Self::Output {
        match self {
            MeasurementValue::Float(a) => MeasurementValue::Float(a / rhs as f64),
            MeasurementValue::Int(a) => MeasurementValue::Int(a / rhs as u128),
        }
    }
}

impl Add<MeasurementValue> for MeasurementValue {
    type Output = MeasurementValue;

    fn add(self, rhs: MeasurementValue) -> Self::Output {
        match (self, rhs) {
            (MeasurementValue::Float(a), MeasurementValue::Float(b)) => {
                MeasurementValue::Float(a + b)
            }
            (MeasurementValue::Int(a), MeasurementValue::Int(b)) => MeasurementValue::Int(a + b),
            _ => vortex_panic!("Can't subtract two measurement values of different kinds"),
        }
    }
}

impl Sub<MeasurementValue> for MeasurementValue {
    type Output = MeasurementValue;

    fn sub(self, rhs: MeasurementValue) -> Self::Output {
        match (self, rhs) {
            (MeasurementValue::Float(a), MeasurementValue::Float(b)) => {
                MeasurementValue::Float(a - b)
            }
            (MeasurementValue::Int(a), MeasurementValue::Int(b)) => MeasurementValue::Int(a - b),
            _ => vortex_panic!("Can't subtract two measurement values of different kinds"),
        }
    }
}

impl PartialOrd<Self> for MeasurementValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (MeasurementValue::Float(a), MeasurementValue::Float(b)) => a.partial_cmp(b),
            (MeasurementValue::Int(a), MeasurementValue::Int(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

#[derive(Serialize)]
pub struct JsonValue {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage: Option<String>,
    pub unit: Option<Cow<'static, str>>,
    pub value: MeasurementValue,
    pub time: Option<u128>,
    pub bytes: Option<u64>,
    pub commit_id: Cow<'static, str>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TableValue {
    pub id: Option<usize>,
    pub name: String,
    pub format: Format,
    pub unit: Cow<'static, str>,
    pub value: MeasurementValue,
}

impl Eq for TableValue {}

impl PartialOrd<Self> for TableValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TableValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id
            .zip(other.id)
            .map(|(a, b)| a.cmp(&b))
            .unwrap_or_else(|| self.name.cmp(&other.name))
    }
}

pub struct TimingMeasurement {
    pub name: String,
    pub format: Format,
    pub storage: String,
    pub time: Duration,
}

impl ToTable for TimingMeasurement {
    fn to_table(&self) -> TableValue {
        TableValue {
            id: None,
            name: self.name.clone(),
            format: self.format,
            unit: Cow::from("μs"),
            value: MeasurementValue::Int(self.time.as_micros()),
        }
    }
}

impl ToJson for TimingMeasurement {
    fn to_json(&self) -> JsonValue {
        JsonValue {
            name: self.name.clone(),
            storage: Some(self.storage.clone()),
            unit: Some(Cow::from("ns")),
            value: MeasurementValue::Int(self.time.as_nanos()),
            bytes: None,
            time: None,
            commit_id: Cow::from(GIT_COMMIT_ID.as_str()),
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
            storage: Some(self.storage.clone()),
            unit: Some(Cow::from("ns")),
            value: MeasurementValue::Int(self.time.as_nanos()),
            bytes: None,
            time: None,
            commit_id: Cow::from(GIT_COMMIT_ID.as_str()),
        }
    }
}

impl ToTable for QueryMeasurement {
    fn to_table(&self) -> TableValue {
        TableValue {
            id: Some(self.query_idx),
            name: self.query_idx.to_string(),
            format: self.format,
            unit: Cow::from("μs"),
            value: MeasurementValue::Int(self.time.as_micros()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ThroughputMeasurement {
    pub name: String,
    pub format: Format,
    pub time: Duration,
    pub bytes: u64,
}

impl ToJson for ThroughputMeasurement {
    fn to_json(&self) -> JsonValue {
        let name = match self.format {
            Format::OnDiskVortex => format!("{} throughput", self.name),
            Format::Parquet => format!("parquet_rs-zstd {} throughput", self.name),
            _ => vortex_panic!("Throughput only supports vortex and parquet formats"),
        };

        JsonValue {
            name,
            storage: None,
            unit: Some(Cow::from("bytes/ns")),
            value: MeasurementValue::Float((self.bytes as f64) / self.time.as_nanos() as f64),
            time: Some(self.time.as_nanos()),
            bytes: Some(self.bytes),
            commit_id: Cow::from(GIT_COMMIT_ID.as_str()),
        }
    }
}

impl ToTable for ThroughputMeasurement {
    fn to_table(&self) -> TableValue {
        TableValue {
            id: None,
            name: self.name.clone(),
            format: self.format,
            unit: Cow::from("bytes / μs"),
            value: MeasurementValue::Float((self.bytes as f64) / self.time.as_micros() as f64),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CustomUnitMeasurement {
    pub name: String,
    pub format: Format,
    pub unit: Cow<'static, str>,
    pub value: f64,
}

impl ToJson for CustomUnitMeasurement {
    fn to_json(&self) -> JsonValue {
        JsonValue {
            name: self.name.clone(),
            storage: None,
            unit: None,
            value: MeasurementValue::Float(self.value),
            time: None,
            bytes: None,
            commit_id: Cow::from(GIT_COMMIT_ID.as_str()),
        }
    }
}

impl ToTable for CustomUnitMeasurement {
    fn to_table(&self) -> TableValue {
        TableValue {
            id: None,
            name: self.name.clone(),
            format: self.format,
            unit: self.unit.clone(),
            value: MeasurementValue::Float(self.value),
        }
    }
}
