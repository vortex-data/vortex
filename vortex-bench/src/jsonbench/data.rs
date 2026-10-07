// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Data preparation for JSONBench.
//!
//! The raw data is ClickHouse's JSONBench Bluesky event dump: gzipped JSON-lines files of one
//! million events each. Every format is derived from the same raw lines:
//!
//! - `parquet`: one `data` column holding each event as a JSON string.
//! - `parquet-variant`: one `data` column holding each event as a shredded Parquet Variant.
//! - `vortex`/`vortex-compact`: the `parquet-variant` files converted to Vortex, so the Vortex
//!   `Variant` column carries exactly the same shredded values.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use flate2::read::MultiGzDecoder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::properties::WriterProperties;
use parquet_variant::VariantPath;
use parquet_variant::VariantPathElement;
use parquet_variant_compute::ShreddedSchemaBuilder;
use parquet_variant_compute::json_to_variant;
use parquet_variant_compute::shred_variant;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use tracing::info;
use tracing::warn;

/// Rows in each raw JSONBench file.
pub const ROWS_PER_FILE: usize = 1_000_000;

/// Rows per record batch while converting.
const BATCH_ROWS: usize = 65_536;

/// Rows sampled from the first raw file to infer the shredding schema.
const SHREDDING_SAMPLE_ROWS: usize = 100_000;

/// Minimum fraction of sampled rows a path must appear in to be shredded.
const SHREDDING_MIN_PRESENCE: f64 = 0.01;

/// Minimum fraction of a path's occurrences that must share one scalar type for it to be shredded.
const SHREDDING_MIN_TYPE_SHARE: f64 = 0.99;

/// Deepest object nesting the shredding inference descends into.
const SHREDDING_MAX_DEPTH: usize = 6;

/// URL of the `file_idx`-th (1-based) raw JSONBench file.
pub fn raw_json_url(file_idx: usize) -> String {
    format!(
        "https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_{file_idx:04}.json.gz"
    )
}

/// File name of the `file_idx`-th (1-based) raw JSONBench file.
pub fn raw_json_name(file_idx: usize) -> String {
    format!("file_{file_idx:04}.json.gz")
}

/// Stem shared by every derived file of the `file_idx`-th raw file.
pub fn output_stem(file_idx: usize) -> String {
    format!("bluesky_{file_idx:04}")
}

/// The scalar type a shredded path is stored as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShredType {
    Boolean,
    Int64,
    Float64,
    Utf8,
}

impl ShredType {
    fn of(value: &Value) -> Option<Self> {
        match value {
            Value::Bool(_) => Some(Self::Boolean),
            Value::Number(n) if n.is_i64() => Some(Self::Int64),
            Value::Number(_) => Some(Self::Float64),
            Value::String(_) => Some(Self::Utf8),
            Value::Null | Value::Array(_) | Value::Object(_) => None,
        }
    }

    fn data_type(self) -> DataType {
        match self {
            Self::Boolean => DataType::Boolean,
            Self::Int64 => DataType::Int64,
            Self::Float64 => DataType::Float64,
            Self::Utf8 => DataType::Utf8,
        }
    }
}

/// One shredded path: object field names from the root, and the type the leaf is stored as.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShreddedPath {
    pub path: Vec<String>,
    pub shred_type: ShredType,
}

/// The paths shredded into typed columns by both Variant formats.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShreddingSchema {
    pub paths: Vec<ShreddedPath>,
}

impl ShreddingSchema {
    /// Infer the shredding schema from the first `SHREDDING_SAMPLE_ROWS` rows of `raw`.
    ///
    /// A scalar leaf is shredded when it appears in at least `SHREDDING_MIN_PRESENCE` of the sampled
    /// rows and at least `SHREDDING_MIN_TYPE_SHARE` of its occurrences share one type. This mirrors
    /// the automatic path discovery engines with native JSON storage perform, and is independent of
    /// the benchmark queries.
    pub fn infer(raw: &Path) -> anyhow::Result<Self> {
        let mut stats: BTreeMap<Vec<String>, BTreeMap<Option<ShredType>, usize>> = BTreeMap::new();
        let mut rows = 0usize;
        for line in json_lines(raw)? {
            let line = line?;
            let Ok(value) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            rows += 1;
            let mut path = Vec::new();
            collect_leaf_types(&value, &mut path, &mut stats);
            if rows == SHREDDING_SAMPLE_ROWS {
                break;
            }
        }
        anyhow::ensure!(rows > 0, "no JSON rows in {}", raw.display());

        let mut paths = Vec::new();
        for (path, types) in stats {
            let occurrences: usize = types.values().sum();
            let Some((Some(shred_type), &count)) = types.iter().max_by_key(|(_, count)| **count)
            else {
                continue;
            };
            #[expect(clippy::cast_precision_loss)]
            let presence = occurrences as f64 / rows as f64;
            #[expect(clippy::cast_precision_loss)]
            let type_share = count as f64 / occurrences as f64;
            if presence >= SHREDDING_MIN_PRESENCE && type_share >= SHREDDING_MIN_TYPE_SHARE {
                paths.push(ShreddedPath {
                    path,
                    shred_type: *shred_type,
                });
            }
        }
        Ok(Self { paths })
    }

    /// The Arrow shredding type passed to [`shred_variant`].
    pub fn arrow_shredding_type(&self) -> anyhow::Result<DataType> {
        let mut builder = ShreddedSchemaBuilder::new();
        for shredded in &self.paths {
            let path = VariantPath::from_iter(
                shredded
                    .path
                    .iter()
                    .map(|name| VariantPathElement::from(name.as_str())),
            );
            builder = builder.with_path(path, &shredded.shred_type.data_type())?;
        }
        Ok(builder.build())
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        serde_json::to_writer_pretty(File::create(path)?, self)?;
        Ok(())
    }
}

/// Record the type of every scalar leaf under `value`, keyed by its object path.
///
/// Values inside arrays are not descended into: Parquet Variant shredding cannot address list
/// elements by path.
fn collect_leaf_types(
    value: &Value,
    path: &mut Vec<String>,
    stats: &mut BTreeMap<Vec<String>, BTreeMap<Option<ShredType>, usize>>,
) {
    if let Value::Object(fields) = value
        && path.len() < SHREDDING_MAX_DEPTH
    {
        // An object is not a scalar leaf, but counting it keeps a path that holds an object in
        // some rows and a scalar in others from being shredded as that scalar.
        if !path.is_empty() {
            *stats
                .entry(path.clone())
                .or_default()
                .entry(None)
                .or_default() += 1;
        }
        for (name, child) in fields {
            path.push(name.clone());
            collect_leaf_types(child, path, stats);
            path.pop();
        }
    } else if !path.is_empty() {
        *stats
            .entry(path.clone())
            .or_default()
            .entry(ShredType::of(value))
            .or_default() += 1;
    }
}

/// Iterate the lines of a gzipped JSON-lines file.
fn json_lines(raw: &Path) -> anyhow::Result<impl Iterator<Item = std::io::Result<String>>> {
    let file = File::open(raw).with_context(|| format!("opening {}", raw.display()))?;
    Ok(BufReader::with_capacity(1 << 20, MultiGzDecoder::new(file)).lines())
}

/// Batches of the valid JSON documents in `raw`, as a `data` string column.
///
/// Lines that do not parse as JSON are skipped, so every format holds the same rows.
fn json_batches(raw: &Path) -> anyhow::Result<impl Iterator<Item = anyhow::Result<StringArray>>> {
    let mut lines = json_lines(raw)?;
    let mut skipped = 0usize;
    let raw_display = raw.display().to_string();
    Ok(std::iter::from_fn(move || {
        let mut batch = Vec::with_capacity(BATCH_ROWS);
        for line in lines.by_ref() {
            let line = match line {
                Ok(line) => line,
                Err(err) => return Some(Err(err.into())),
            };
            if serde_json::from_str::<serde::de::IgnoredAny>(&line).is_err() {
                skipped += 1;
                continue;
            }
            batch.push(line);
            if batch.len() == BATCH_ROWS {
                break;
            }
        }
        if batch.is_empty() {
            if skipped > 0 {
                warn!("skipped {skipped} invalid JSON lines in {raw_display}");
            }
            return None;
        }
        Some(Ok(StringArray::from(batch)))
    }))
}

/// Parquet writer properties for every Parquet file this benchmark writes.
fn parquet_writer_properties() -> anyhow::Result<WriterProperties> {
    Ok(WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build())
}

/// Write the JSON documents of `raw` to `output` as a Parquet file with a `data` string column.
pub fn write_json_parquet(raw: &Path, output: &Path) -> anyhow::Result<()> {
    let schema: SchemaRef = Arc::new(Schema::new(vec![Field::new("data", DataType::Utf8, false)]));
    let mut writer = ArrowWriter::try_new(
        File::create(output)?,
        Arc::clone(&schema),
        Some(parquet_writer_properties()?),
    )?;
    for batch in json_batches(raw)? {
        let column: ArrayRef = Arc::new(batch?);
        writer.write(&RecordBatch::try_new(Arc::clone(&schema), vec![column])?)?;
    }
    writer.close()?;
    info!("wrote {}", output.display());
    Ok(())
}

/// Write the JSON documents of `raw` to `output` as a Parquet file with a shredded Variant `data`
/// column.
pub fn write_variant_parquet(
    raw: &Path,
    output: &Path,
    shredding: &ShreddingSchema,
) -> anyhow::Result<()> {
    let shredding_type = shredding.arrow_shredding_type()?;
    let mut writer: Option<ArrowWriter<File>> = None;
    for batch in json_batches(raw)? {
        let strings: ArrayRef = Arc::new(batch?);
        let variant = shred_variant(&json_to_variant(&strings)?, &shredding_type)?;
        let schema = Arc::new(Schema::new(vec![
            variant.field("data").with_nullable(false),
        ]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ArrayRef::from(variant)])?;
        let writer = match writer.as_mut() {
            Some(writer) => writer,
            None => writer.insert(ArrowWriter::try_new(
                File::create(output)?,
                schema,
                Some(parquet_writer_properties()?),
            )?),
        };
        writer.write(&batch)?;
    }
    writer
        .ok_or_else(|| anyhow::anyhow!("no JSON rows in {}", raw.display()))?
        .close()?;
    info!("wrote {}", output.display());
    Ok(())
}
