// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Resolving sources into fingerprinted files, columns and chunk inputs.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::Context;
use anyhow::bail;
use arrow_schema::DataType;
use glob::Pattern;
use parquet::arrow::parquet_to_arrow_schema;
use parquet::file::reader::FileReader;
use parquet::file::reader::SerializedFileReader;
use serde::Deserialize;
use serde::Serialize;

use crate::key::file_digest;
use crate::spec::DTypeClass;
use crate::spec::ParamValue;
use crate::spec::SourceSpec;
use crate::spec::SyntheticPType;
use crate::spec::expand_grid;

/// A source after resolution, recorded in the plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedSource {
    /// Parquet files matched by the source's glob, in sorted path order.
    Parquet {
        /// The source name.
        name: String,
        /// The matched files.
        files: Vec<ResolvedFile>,
    },
    /// Generated TPC-H tables (each recorded as a file named `tpch/<table>`).
    Tpch {
        /// The source name.
        name: String,
        /// The TPC-H scale factor.
        scale_factor: f64,
        /// The tables and their selected columns.
        tables: Vec<ResolvedFile>,
    },
    /// A synthetic grid.
    Synthetic {
        /// The source name.
        name: String,
        /// The number of parameter combinations.
        combinations: u64,
        /// The number of seeds per combination.
        seeds: u64,
    },
}

/// A Parquet file, identified by the hash of its contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedFile {
    /// The path relative to the spec file's directory, with `/` separators.
    pub path: String,
    /// The SHA-256 of the file's contents.
    pub fingerprint: String,
    /// The number of rows in the file.
    pub num_rows: u64,
    /// The columns selected for planning.
    pub columns: Vec<ResolvedColumn>,
}

/// A selected column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedColumn {
    /// The top-level column name.
    pub name: String,
    /// The Arrow type, as displayed by Arrow.
    pub arrow_type: String,
    /// The dtype class.
    pub dtype: DTypeClass,
}

/// A column that matched the source's column patterns but was not planned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedColumn {
    /// The source name.
    pub source: String,
    /// The file path, relative to the spec file's directory.
    pub file: String,
    /// The column name.
    pub column: String,
    /// The Arrow type, as displayed by Arrow.
    pub arrow_type: String,
    /// Why the column was skipped.
    pub reason: String,
}

/// The input of one chunk task: where its rows come from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source_kind", rename_all = "snake_case")]
pub enum ChunkInput {
    /// A row range of one Parquet column.
    Parquet {
        /// The source name.
        source: String,
        /// The file path, relative to the spec file's directory.
        file: String,
        /// The SHA-256 of the file's contents.
        file_fingerprint: String,
        /// The column name.
        column: String,
        /// The Arrow type, as displayed by Arrow.
        arrow_type: String,
        /// The dtype class.
        dtype: DTypeClass,
        /// The first row, inclusive.
        row_start: u64,
        /// The last row, exclusive.
        row_end: u64,
    },
    /// A row range of one generated TPC-H column.
    Tpch {
        /// The source name.
        source: String,
        /// The table.
        table: String,
        /// The TPC-H scale factor.
        scale_factor: f64,
        /// The column name.
        column: String,
        /// The Arrow type, as displayed by Arrow.
        arrow_type: String,
        /// The dtype class.
        dtype: DTypeClass,
        /// The first row, inclusive.
        row_start: u64,
        /// The last row, exclusive.
        row_end: u64,
    },
    /// One seeded draw from a synthetic generator.
    Synthetic {
        /// The source name.
        source: String,
        /// The generator name.
        generator: String,
        /// The physical type of the generated values.
        ptype: SyntheticPType,
        /// The dtype class.
        dtype: DTypeClass,
        /// This chunk's parameter combination.
        params: BTreeMap<String, ParamValue>,
        /// The generator seed.
        seed: u64,
        /// The number of rows.
        rows: u64,
    },
}

impl ChunkInput {
    /// The chunk's dtype class.
    pub fn dtype(&self) -> DTypeClass {
        match self {
            Self::Parquet { dtype, .. }
            | Self::Tpch { dtype, .. }
            | Self::Synthetic { dtype, .. } => *dtype,
        }
    }

    /// The column name, or the generator name for synthetic chunks.
    pub fn column(&self) -> &str {
        match self {
            Self::Parquet { column, .. } | Self::Tpch { column, .. } => column,
            Self::Synthetic { generator, .. } => generator,
        }
    }

    /// The source name.
    pub fn source(&self) -> &str {
        match self {
            Self::Parquet { source, .. }
            | Self::Tpch { source, .. }
            | Self::Synthetic { source, .. } => source,
        }
    }
}

/// The resolution of one source: its record for the plan, its chunks and its skipped columns.
pub(crate) struct Resolution {
    pub(crate) source: ResolvedSource,
    pub(crate) chunks: Vec<ChunkInput>,
    pub(crate) skipped: Vec<SkippedColumn>,
}

/// Resolves a source into its chunks. Output order is deterministic.
pub(crate) fn resolve_source(
    spec: &SourceSpec,
    spec_dir: &Path,
    chunk_rows: u64,
    max_chunks: Option<u64>,
    dtypes: &[DTypeClass],
) -> anyhow::Result<Resolution> {
    match spec {
        SourceSpec::Tpch {
            name,
            scale_factor,
            tables,
            columns,
        } => resolve_tpch(
            name,
            *scale_factor,
            tables,
            &patterns(columns)?,
            chunk_rows,
            max_chunks,
            dtypes,
        ),
        SourceSpec::Parquet {
            name,
            path,
            columns,
            exclude_columns,
        } => resolve_parquet(
            name,
            path,
            &patterns(columns)?,
            &patterns(exclude_columns)?,
            spec_dir,
            chunk_rows,
            max_chunks,
            dtypes,
        ),
        SourceSpec::Synthetic {
            name,
            generator,
            ptype,
            rows,
            seeds,
            params,
        } => {
            let grid = expand_grid(params);
            let dtype = ptype.dtype_class();
            let chunks = if dtypes.contains(&dtype) {
                grid.iter()
                    .flat_map(|combination| {
                        seeds.iter().map(|seed| ChunkInput::Synthetic {
                            source: name.clone(),
                            generator: generator.clone(),
                            ptype: *ptype,
                            dtype,
                            params: combination.clone(),
                            seed: *seed,
                            rows: *rows,
                        })
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Ok(Resolution {
                source: ResolvedSource::Synthetic {
                    name: name.clone(),
                    combinations: grid.len() as u64,
                    seeds: seeds.len() as u64,
                },
                chunks,
                skipped: Vec::new(),
            })
        }
    }
}

fn patterns(globs: &[String]) -> anyhow::Result<Vec<Pattern>> {
    globs
        .iter()
        .map(|glob| Pattern::new(glob).with_context(|| format!("invalid column pattern `{glob}`")))
        .collect()
}

/// Row ranges of `chunk_rows`, keeping at most `max` of them spread evenly through the column.
fn chunk_ranges(num_rows: u64, chunk_rows: u64, max: Option<u64>) -> Vec<(u64, u64)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < num_rows {
        let end = start.saturating_add(chunk_rows).min(num_rows);
        ranges.push((start, end));
        start = end;
    }
    match max {
        Some(max) if (ranges.len() as u64) > max => {
            let total = ranges.len() as u64;
            (0..max)
                .map(|i| ranges[usize::try_from(i * total / max).unwrap_or(0)])
                .collect()
        }
        _ => ranges,
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_tpch(
    name: &str,
    scale_factor: f64,
    tables: &[String],
    include: &[Pattern],
    chunk_rows: u64,
    max_chunks: Option<u64>,
    dtypes: &[DTypeClass],
) -> anyhow::Result<Resolution> {
    let mut resolved = Vec::new();
    let mut chunks = Vec::new();
    let mut skipped = Vec::new();
    for table in tables {
        let batches = crate::tpch::table(table, scale_factor)?;
        let Some(first) = batches.first() else {
            continue;
        };
        let num_rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        let mut columns = Vec::new();
        for field in first.schema().fields() {
            let column = field.name();
            if !(include.is_empty() || include.iter().any(|p| p.matches(column))) {
                continue;
            }
            let arrow_type = field.data_type().to_string();
            match classify(field.data_type()) {
                Some(dtype) if dtypes.contains(&dtype) => columns.push(ResolvedColumn {
                    name: column.clone(),
                    arrow_type,
                    dtype,
                }),
                other => skipped.push(SkippedColumn {
                    source: name.to_string(),
                    file: format!("tpch/{table}"),
                    column: column.clone(),
                    arrow_type,
                    reason: other.map_or_else(
                        || "nested or unsupported type".to_string(),
                        |d| format!("dtype `{}` is not in the plan", d.as_str()),
                    ),
                }),
            }
        }
        for column in &columns {
            for (row_start, row_end) in chunk_ranges(num_rows, chunk_rows, max_chunks) {
                chunks.push(ChunkInput::Tpch {
                    source: name.to_string(),
                    table: table.clone(),
                    scale_factor,
                    column: column.name.clone(),
                    arrow_type: column.arrow_type.clone(),
                    dtype: column.dtype,
                    row_start,
                    row_end,
                });
            }
        }
        resolved.push(ResolvedFile {
            path: format!("tpch/{table}"),
            fingerprint: format!("tpch:{table}:sf={scale_factor}"),
            num_rows,
            columns,
        });
    }
    Ok(Resolution {
        source: ResolvedSource::Tpch {
            name: name.to_string(),
            scale_factor,
            tables: resolved,
        },
        chunks,
        skipped,
    })
}

#[allow(clippy::too_many_arguments)]
fn resolve_parquet(
    name: &str,
    path_glob: &str,
    include: &[Pattern],
    exclude: &[Pattern],
    spec_dir: &Path,
    chunk_rows: u64,
    max_chunks: Option<u64>,
    dtypes: &[DTypeClass],
) -> anyhow::Result<Resolution> {
    let full_glob = spec_dir.join(path_glob);
    let full_glob = full_glob
        .to_str()
        .with_context(|| format!("source `{name}`: path is not valid UTF-8"))?;
    let mut paths = glob::glob(full_glob)
        .with_context(|| format!("source `{name}`: invalid path glob `{path_glob}`"))?
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("source `{name}`: listing `{path_glob}`"))?;
    paths.sort();
    if paths.is_empty() {
        bail!("source `{name}`: no files match `{path_glob}` (relative to the spec file)");
    }

    let mut files = Vec::with_capacity(paths.len());
    let mut chunks = Vec::new();
    let mut skipped = Vec::new();

    for path in paths {
        let relative = path
            .strip_prefix(spec_dir)
            .unwrap_or(&path)
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");

        let reader = SerializedFileReader::new(
            File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )
        .with_context(|| format!("reading Parquet footer of {}", path.display()))?;
        let file_metadata = reader.metadata().file_metadata();
        let num_rows = u64::try_from(file_metadata.num_rows())?;
        let schema = parquet_to_arrow_schema(
            file_metadata.schema_descr(),
            file_metadata.key_value_metadata(),
        )
        .with_context(|| format!("converting the schema of {}", path.display()))?;
        let fingerprint = file_digest(&path)?;

        let mut columns = Vec::new();
        for field in schema.fields() {
            let column = field.name();
            let selected = (include.is_empty() || include.iter().any(|p| p.matches(column)))
                && !exclude.iter().any(|p| p.matches(column));
            if !selected {
                continue;
            }
            let arrow_type = field.data_type().to_string();
            let reason = match classify(field.data_type()) {
                Some(dtype) if dtypes.contains(&dtype) => {
                    columns.push(ResolvedColumn {
                        name: column.clone(),
                        arrow_type,
                        dtype,
                    });
                    continue;
                }
                Some(dtype) => format!("dtype `{}` is not in the plan", dtype.as_str()),
                None => "nested or unsupported type".to_string(),
            };
            skipped.push(SkippedColumn {
                source: name.to_string(),
                file: relative.clone(),
                column: column.clone(),
                arrow_type,
                reason,
            });
        }

        for column in &columns {
            for (row_start, row_end) in chunk_ranges(num_rows, chunk_rows, max_chunks) {
                chunks.push(ChunkInput::Parquet {
                    source: name.to_string(),
                    file: relative.clone(),
                    file_fingerprint: fingerprint.clone(),
                    column: column.name.clone(),
                    arrow_type: column.arrow_type.clone(),
                    dtype: column.dtype,
                    row_start,
                    row_end,
                });
            }
        }

        files.push(ResolvedFile {
            path: relative,
            fingerprint,
            num_rows,
            columns,
        });
    }

    Ok(Resolution {
        source: ResolvedSource::Parquet {
            name: name.to_string(),
            files,
        },
        chunks,
        skipped,
    })
}

/// Maps an Arrow type to its dtype class, or `None` for nested and unsupported types.
fn classify(data_type: &DataType) -> Option<DTypeClass> {
    match data_type {
        DataType::Boolean => Some(DTypeClass::Bool),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Some(DTypeClass::Int),
        DataType::Float16 | DataType::Float32 | DataType::Float64 => Some(DTypeClass::Float),
        DataType::Decimal32(..)
        | DataType::Decimal64(..)
        | DataType::Decimal128(..)
        | DataType::Decimal256(..) => Some(DTypeClass::Decimal),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Some(DTypeClass::String),
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => Some(DTypeClass::Binary),
        DataType::Date32
        | DataType::Date64
        | DataType::Timestamp(..)
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Duration(_) => Some(DTypeClass::Temporal),
        DataType::Dictionary(_, values) => classify(values),
        _ => None,
    }
}
