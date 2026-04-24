// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::Display;
use std::fmt::Formatter;
use std::ops::Add;
use std::ops::Div;
use std::ops::Sub;
use std::time::Duration;

use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use serde::Serializer;
use target_lexicon::Triple;
use vortex::error::VortexExpect;
use vortex::error::vortex_panic;

use crate::BenchmarkDataset;
use crate::Engine;
use crate::Format;
use crate::Target;
use crate::memory::MemoryMeasurementResult;
use crate::utils::GIT_COMMIT_ID;

pub trait ToJson {
    fn to_json(&self) -> serde_json::Value;
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
            MeasurementValue::Int(i) => write!(f, "{i}"),
            MeasurementValue::Float(fl) => match f.precision() {
                None => write!(f, "{fl}"),
                Some(p) => write!(f, "{fl:.p$}"),
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
    pub target: Target,
    pub time: Option<u128>,
    pub bytes: Option<u64>,
    pub commit_id: Cow<'static, str>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TableValue {
    pub id: Option<usize>,
    pub name: String,
    pub target: Target,
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
    pub target: Target,
    pub storage: String,
    pub runs: Vec<Duration>,
}

impl TimingMeasurement {
    pub fn mean_time(&self) -> Duration {
        let len = self.runs.len();
        if len == 0 {
            vortex_panic!("cannot have no runs");
        }

        let total_nanos: u128 = self.runs.iter().map(|d| d.as_nanos()).sum();
        let mean_nanos = total_nanos / len as u128;
        Duration::new(
            u64::try_from(mean_nanos / 1_000_000_000)
                .vortex_expect("nanosecond conversion must fit in u64/u32"),
            u32::try_from(mean_nanos % 1_000_000_000)
                .vortex_expect("nanosecond conversion must fit in u64/u32"),
        )
    }

    pub fn median_time(&self) -> Duration {
        let len = self.runs.len();
        if len == 0 {
            vortex_panic!("cannot have no runs");
        }

        let mut sorted_runs = self.runs.clone();
        sorted_runs.sort();

        if len % 2 == 1 {
            sorted_runs[len / 2]
        } else {
            let mid1 = sorted_runs[len / 2 - 1];
            let mid2 = sorted_runs[len / 2];
            (mid1 + mid2) / 2
        }
    }
}

impl ToTable for TimingMeasurement {
    fn to_table(&self) -> TableValue {
        TableValue {
            id: None,
            name: self.name.clone(),
            target: self.target,
            unit: Cow::from("μs"),
            value: MeasurementValue::Int(self.median_time().as_micros()),
        }
    }
}

impl ToJson for TimingMeasurement {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(JsonValue {
            name: self.name.clone(),
            storage: Some(self.storage.clone()),
            unit: Some(Cow::from("ns")),
            value: MeasurementValue::Int(self.median_time().as_nanos()),
            bytes: None,
            time: None,
            commit_id: Cow::from(GIT_COMMIT_ID.as_str()),
            target: self.target,
        })
        .expect("value is valid JSON")
    }
}

#[derive(Clone, Debug)]
pub struct QueryMeasurement {
    pub query_idx: usize,
    pub target: Target,
    pub benchmark_dataset: BenchmarkDataset,
    pub benchmark_runner: String,
    /// The storage backend against which this test was run. One of: s3, gcs, nvme.
    pub storage: String,
    pub runs: Vec<Duration>,
}

impl QueryMeasurement {
    pub fn median_run(&self) -> Duration {
        let len = self.runs.len();
        if len == 0 {
            vortex_panic!("cannot have no runs");
        }

        let mut sorted_runs = self.runs.clone();
        sorted_runs.sort();

        if len % 2 == 1 {
            sorted_runs[len / 2]
        } else {
            let mid1 = sorted_runs[len / 2 - 1];
            let mid2 = sorted_runs[len / 2];
            let avg_nanos = (mid1.as_nanos() + mid2.as_nanos()) / 2;
            Duration::new(
                u64::try_from(avg_nanos / 1_000_000_000)
                    .vortex_expect("nanosecond conversion must fit in u64/u32"),
                u32::try_from(avg_nanos % 1_000_000_000)
                    .vortex_expect("nanosecond conversion must fit in u64/u32"),
            )
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct QueryMeasurementJson {
    pub name: String,
    pub storage: String,
    pub dataset: BenchmarkDataset,
    /// The cloud runner used to run this
    pub runner: String,
    pub unit: String,
    pub value: u128,
    pub all_runtimes: Vec<u128>,
    pub target: Target,
    pub commit_id: String,
    pub env_triple: TripleJson,
}

#[derive(Serialize, Deserialize)]
pub struct TripleJson {
    pub architecture: String,
    pub operating_system: String,
    pub environment: String,
}

impl ToJson for QueryMeasurement {
    fn to_json(&self) -> serde_json::Value {
        let name = format!(
            "{dataset}_q{query_idx:02}/{engine}:{format}",
            dataset = self.benchmark_dataset.name(),
            engine = self.target.engine,
            format = self.target.format.name(),
            query_idx = self.query_idx
        );

        let host = Triple::host();

        serde_json::to_value(QueryMeasurementJson {
            name,
            storage: self.storage.clone(),
            dataset: self.benchmark_dataset.clone(),
            runner: self.benchmark_runner.clone(),
            unit: "ns".to_string(),
            value: self.median_run().as_nanos(),
            all_runtimes: self.runs.iter().map(|r| r.as_nanos()).collect_vec(),
            commit_id: GIT_COMMIT_ID.to_string(),
            target: self.target,
            env_triple: TripleJson {
                architecture: host.architecture.to_string(),
                operating_system: host.operating_system.to_string(),
                environment: host.environment.to_string(),
            },
        })
        .expect("value is valid json")
    }
}

impl ToTable for QueryMeasurement {
    fn to_table(&self) -> TableValue {
        TableValue {
            id: Some(self.query_idx),
            name: self.query_idx.to_string(),
            target: self.target,
            unit: Cow::from("μs"),
            value: MeasurementValue::Int(self.median_run().as_micros()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompressionTimingMeasurement {
    pub name: String,
    pub format: Format,
    pub time: Duration,
}

impl ToJson for CompressionTimingMeasurement {
    fn to_json(&self) -> serde_json::Value {
        let (name, engine) = match self.format {
            Format::OnDiskVortex => (self.name.to_string(), Engine::Vortex),
            Format::Parquet => (format!("parquet_rs-zstd {}", self.name), Engine::Arrow),
            Format::Lance => (format!("lance {}", self.name), Engine::Arrow),
            _ => vortex_panic!(
                "CompressionTimingMeasurement only supports vortex, lance, and parquet formats"
            ),
        };

        serde_json::to_value(JsonValue {
            name,
            storage: None,
            unit: Some(Cow::from("ns")),
            value: MeasurementValue::Float(self.time.as_nanos() as f64),
            time: Some(self.time.as_nanos()),
            bytes: None,
            commit_id: Cow::from(GIT_COMMIT_ID.as_str()),
            target: Target::new(engine, self.format),
        })
        .expect("value is valid JSON")
    }
}

impl ToTable for CompressionTimingMeasurement {
    fn to_table(&self) -> TableValue {
        TableValue {
            id: None,
            name: self.name.clone(),
            target: Target::new(Engine::default(), self.format),
            unit: Cow::from("μs"),
            value: MeasurementValue::Float(self.time.as_micros() as f64),
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
    fn to_json(&self) -> serde_json::Value {
        let engine = match self.format {
            Format::OnDiskVortex | Format::VortexCompact => Engine::Vortex,
            Format::Parquet => Engine::Arrow,
            Format::Lance => Engine::Arrow,
            _ => Engine::Vortex, // Default to Vortex for other formats.
        };

        serde_json::to_value(JsonValue {
            name: self.name.clone(),
            storage: None,
            unit: Some(self.unit.clone()),
            value: MeasurementValue::Float(self.value),
            // time & bytes are only used for throughputs
            time: None,
            bytes: None,
            commit_id: Cow::from(GIT_COMMIT_ID.as_str()),
            target: Target::new(engine, self.format),
        })
        .expect("value is valid JSON")
    }
}

impl ToTable for CustomUnitMeasurement {
    fn to_table(&self) -> TableValue {
        TableValue {
            id: None,
            name: self.name.clone(),
            target: Target::new(Engine::default(), self.format),
            unit: self.unit.clone(),
            value: MeasurementValue::Float(self.value),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryMeasurement {
    pub query_idx: usize,
    pub target: Target,
    pub benchmark_dataset: BenchmarkDataset,
    pub benchmark_runner: String,
    pub storage: String,
    pub physical_memory_delta: i64,
    pub virtual_memory_delta: i64,
    pub peak_physical_memory: u64,
    pub peak_virtual_memory: u64,
}

impl MemoryMeasurement {
    pub fn new(
        query_idx: usize,
        target: Target,
        benchmark_dataset: BenchmarkDataset,
        benchmark_runner: String,
        storage: String,
        memory_result: MemoryMeasurementResult,
    ) -> Self {
        Self {
            query_idx,
            target,
            benchmark_dataset,
            benchmark_runner,
            storage,
            physical_memory_delta: memory_result.physical_memory_delta,
            virtual_memory_delta: memory_result.virtual_memory_delta,
            peak_physical_memory: memory_result.peak_physical_memory,
            peak_virtual_memory: memory_result.peak_virtual_memory,
        }
    }
}

impl ToJson for MemoryMeasurement {
    fn to_json(&self) -> serde_json::Value {
        let name = format!(
            "{dataset}_q{query_idx:02}_memory/{engine}:{format}",
            dataset = self.benchmark_dataset.name(),
            engine = self.target.engine,
            format = self.target.format.name(),
            query_idx = self.query_idx
        );

        let host = Triple::host();

        serde_json::to_value(MemoryMeasurementJson {
            name,
            storage: self.storage.clone(),
            dataset: self.benchmark_dataset.clone(),
            runner: self.benchmark_runner.clone(),
            physical_memory_delta: self.physical_memory_delta,
            virtual_memory_delta: self.virtual_memory_delta,
            peak_physical_memory: self.peak_physical_memory,
            peak_virtual_memory: self.peak_virtual_memory,
            commit_id: GIT_COMMIT_ID.to_string(),
            target: self.target,
            env_triple: TripleJson {
                architecture: host.architecture.to_string(),
                operating_system: host.operating_system.to_string(),
                environment: host.environment.to_string(),
            },
        })
        .expect("value is valid JSON")
    }
}

impl ToTable for MemoryMeasurement {
    fn to_table(&self) -> TableValue {
        TableValue {
            id: Some(self.query_idx),
            name: format!("q{}_peak", self.query_idx),
            target: self.target,
            unit: Cow::from("MB"),
            value: MeasurementValue::Float(self.peak_physical_memory as f64 / 1024.0 / 1024.0),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct MemoryMeasurementJson {
    pub name: String,
    pub storage: String,
    pub dataset: BenchmarkDataset,
    pub runner: String,
    pub physical_memory_delta: i64,
    pub virtual_memory_delta: i64,
    pub peak_physical_memory: u64,
    pub peak_virtual_memory: u64,
    pub commit_id: String,
    pub target: Target,
    pub env_triple: TripleJson,
}
