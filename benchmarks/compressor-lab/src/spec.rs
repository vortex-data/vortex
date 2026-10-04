// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The user-written plan specification, loaded from TOML.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::bail;
use anyhow::ensure;
use serde::Deserialize;
use serde::Serialize;

/// The chunk size used when a spec does not set `chunk_rows`.
pub const DEFAULT_CHUNK_ROWS: u64 = 65_536;

/// Everything a plan is expanded from, apart from the data itself and the code identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanSpec {
    /// A human-readable name for the plan.
    pub name: String,
    /// The number of rows per chunk. Chunk boundaries are `[i * chunk_rows, (i + 1) * chunk_rows)`.
    #[serde(default = "default_chunk_rows")]
    pub chunk_rows: u64,
    /// The dtype classes to plan for. Columns of other classes are skipped.
    pub dtypes: Vec<DTypeClass>,
    /// Which registered compression schemes the search may use.
    #[serde(default)]
    pub schemes: SchemeSelection,
    /// Parameters for the search over encoding trees.
    #[serde(default)]
    pub search: SearchParams,
    /// Parameters for timing decompression and pushdown.
    #[serde(default)]
    pub measure: MeasureParams,
    /// The data sources to chunk.
    pub sources: Vec<SourceSpec>,
}

fn default_chunk_rows() -> u64 {
    DEFAULT_CHUNK_ROWS
}

impl PlanSpec {
    /// Loads and validates a spec from a TOML file.
    pub fn from_toml_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading plan spec {}", path.display()))?;
        Self::from_toml_str(&text).with_context(|| format!("parsing plan spec {}", path.display()))
    }

    /// Parses and validates a spec from a TOML string.
    pub fn from_toml_str(text: &str) -> anyhow::Result<Self> {
        let spec: Self = toml::from_str(text)?;
        spec.validate()?;
        Ok(spec)
    }

    fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.chunk_rows > 0, "chunk_rows must be greater than zero");
        ensure!(!self.dtypes.is_empty(), "dtypes must name at least one dtype class");
        ensure!(self.search.k >= 1, "search.k must be at least 1");
        ensure!(
            self.search.epsilon.is_finite() && self.search.epsilon >= 0.0,
            "search.epsilon must be a non-negative number"
        );
        ensure!(self.measure.reps >= 1, "measure.reps must be at least 1");
        ensure!(!self.sources.is_empty(), "a plan needs at least one source");

        let mut names = BTreeSet::new();
        for source in &self.sources {
            if !names.insert(source.name()) {
                bail!("duplicate source name `{}`", source.name());
            }
            if let SourceSpec::Synthetic { rows, seeds, .. } = source {
                ensure!(*rows > 0, "synthetic source `{}` needs rows > 0", source.name());
                ensure!(
                    !seeds.is_empty(),
                    "synthetic source `{}` needs at least one seed",
                    source.name()
                );
            }
        }
        Ok(())
    }
}

/// The coarse dtype classes that scheme selection is trained per.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DTypeClass {
    /// Booleans.
    Bool,
    /// Signed and unsigned integers.
    Int,
    /// Floating point numbers.
    Float,
    /// Fixed-point decimals.
    Decimal,
    /// UTF-8 strings.
    String,
    /// Raw bytes.
    Binary,
    /// Dates, times, timestamps and durations.
    Temporal,
}

impl DTypeClass {
    /// The lowercase name used in files and on the command line.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int => "int",
            Self::Float => "float",
            Self::Decimal => "decimal",
            Self::String => "string",
            Self::Binary => "binary",
            Self::Temporal => "temporal",
        }
    }
}

/// Selects schemes from the ones registered in the compression session, in registration order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchemeSelection {
    /// If set, only these scheme ids are used.
    #[serde(default)]
    pub include: Option<Vec<String>>,
    /// Scheme ids to leave out.
    #[serde(default)]
    pub exclude: Vec<String>,
}

/// How the search explores encoding trees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchParams {
    /// The search strategy.
    #[serde(default)]
    pub strategy: SearchStrategy,
    /// How many alternatives to keep per child node.
    #[serde(default = "default_k")]
    pub k: u32,
    /// The ε-dominance margin used to prune alternatives.
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::default(),
            k: default_k(),
            epsilon: default_epsilon(),
        }
    }
}

fn default_k() -> u32 {
    4
}

fn default_epsilon() -> f64 {
    0.02
}

/// The search strategies the lab can run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStrategy {
    /// Try every eligible scheme at every node.
    #[default]
    Exhaustive,
    /// Force each root scheme and let today's compressor choose the children.
    ForcedRoot,
    /// Record today's compressor choices only.
    Default,
}

/// How decompression and pushdown are timed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasureParams {
    /// Repetitions per measurement.
    #[serde(default = "default_reps")]
    pub reps: u32,
    /// The operations to time on each encoding.
    #[serde(default = "default_ops")]
    pub ops: Vec<MeasureOp>,
}

impl Default for MeasureParams {
    fn default() -> Self {
        Self {
            reps: default_reps(),
            ops: default_ops(),
        }
    }
}

fn default_reps() -> u32 {
    9
}

fn default_ops() -> Vec<MeasureOp> {
    vec![MeasureOp::Decode]
}

/// An operation timed on an encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasureOp {
    /// Decompress to the canonical array.
    Decode,
    /// Compare against a constant.
    Compare,
    /// Range predicate.
    Between,
    /// Filter by a mask.
    Filter,
    /// Gather by indices.
    Take,
}

/// A source of chunks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceSpec {
    /// One or more Parquet files, matched by a glob relative to the spec file.
    Parquet {
        /// The source name, used for provenance and dataset splits.
        name: String,
        /// A glob pattern relative to the spec file's directory.
        path: String,
        /// Column name glob patterns to include. Empty includes every column.
        #[serde(default)]
        columns: Vec<String>,
        /// Column name glob patterns to exclude.
        #[serde(default)]
        exclude_columns: Vec<String>,
    },
    /// Seeded synthetic columns, one chunk per (parameter combination, seed).
    Synthetic {
        /// The source name, used for provenance and dataset splits.
        name: String,
        /// The generator to run.
        generator: String,
        /// The physical type of the generated values.
        ptype: SyntheticPType,
        /// Rows per generated chunk.
        rows: u64,
        /// One chunk is generated per seed for every parameter combination.
        seeds: Vec<u64>,
        /// Generator parameters. A list expands into a grid over every combination.
        #[serde(default)]
        params: BTreeMap<String, ParamValues>,
    },
}

impl SourceSpec {
    /// The source name.
    pub fn name(&self) -> &str {
        match self {
            Self::Parquet { name, .. } | Self::Synthetic { name, .. } => name,
        }
    }
}

/// The physical type produced by a synthetic generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticPType {
    /// `u8`
    U8,
    /// `u16`
    U16,
    /// `u32`
    U32,
    /// `u64`
    U64,
    /// `i8`
    I8,
    /// `i16`
    I16,
    /// `i32`
    I32,
    /// `i64`
    I64,
    /// `f32`
    F32,
    /// `f64`
    F64,
}

impl SyntheticPType {
    /// The dtype class of this physical type.
    pub fn dtype_class(self) -> DTypeClass {
        match self {
            Self::F32 | Self::F64 => DTypeClass::Float,
            _ => DTypeClass::Int,
        }
    }
}

/// One parameter value, or a list of values to expand into a grid.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValues {
    /// A list of values; the grid takes each in turn.
    Many(Vec<ParamValue>),
    /// A single value.
    One(ParamValue),
}

impl ParamValues {
    /// The values in declaration order.
    pub fn values(&self) -> &[ParamValue] {
        match self {
            Self::Many(values) => values,
            Self::One(value) => std::slice::from_ref(value),
        }
    }
}

/// A scalar generator parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A string.
    Str(String),
}

/// Expands a parameter map into every combination, in key order then value order.
pub fn expand_grid(params: &BTreeMap<String, ParamValues>) -> Vec<BTreeMap<String, ParamValue>> {
    let mut grid = vec![BTreeMap::new()];
    for (name, values) in params {
        grid = grid
            .into_iter()
            .flat_map(|combination| {
                values.values().iter().map(move |value| {
                    let mut next = combination.clone();
                    next.insert(name.clone(), value.clone());
                    next
                })
            })
            .collect();
    }
    grid
}
