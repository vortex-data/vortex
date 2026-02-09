// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::path::Path;
use std::path::PathBuf;

use arrow_array::PrimitiveArray;
use arrow_array::RecordBatch;
use arrow_array::types::Int64Type;
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use async_trait::async_trait;
use futures::stream;
use itertools::Itertools;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::file::metadata::RowGroupMetaData;
use stream::StreamExt;
use vortex::array::Array;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::expr::Expression;
use vortex::expr::get_item;
use vortex::expr::root;
use vortex::file::OpenOptionsSessionExt;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
use crate::SESSION;
use crate::random_access::FieldPath;
use crate::random_access::ProjectingRandomAccessor;
use crate::random_access::RandomAccessor;

/// Random accessor for Vortex format files.
pub struct VortexRandomAccessor {
    path: PathBuf,
    name: String,
    format: Format,
}

impl VortexRandomAccessor {
    /// Create a new Vortex random accessor for the given file path.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/vortex-tokio-local-disk".to_string(),
            format: Format::OnDiskVortex,
        }
    }

    /// Create a new Vortex random accessor with a custom name.
    pub fn with_name(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            path,
            name: name.into(),
            format: Format::OnDiskVortex,
        }
    }

    /// Create a new Vortex random accessor for compact format.
    pub fn compact(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/vortex-compact-tokio-local-disk".to_string(),
            format: Format::VortexCompact,
        }
    }
}

#[async_trait]
impl RandomAccessor for VortexRandomAccessor {
    fn format(&self) -> Format {
        self.format
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }

    async fn take(&self, indices: Vec<u64>) -> anyhow::Result<usize> {
        let result = take_vortex(&self.path, indices.into()).await?;
        Ok(result.len())
    }
}

/// Random accessor for Parquet format files.
pub struct ParquetRandomAccessor {
    path: PathBuf,
    name: String,
}

impl ParquetRandomAccessor {
    /// Create a new Parquet random accessor for the given file path.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/parquet-tokio-local-disk".to_string(),
        }
    }

    /// Create a new Parquet random accessor with a custom name.
    pub fn with_name(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            path,
            name: name.into(),
        }
    }
}

#[async_trait]
impl RandomAccessor for ParquetRandomAccessor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }

    async fn take(&self, indices: Vec<u64>) -> anyhow::Result<usize> {
        let result = take_parquet(&self.path, indices).await?;
        Ok(result.num_rows())
    }
}

async fn take_vortex(reader: impl AsRef<Path>, indices: Buffer<u64>) -> anyhow::Result<ArrayRef> {
    let array = SESSION
        .open_options()
        .open_path(reader.as_ref())
        .await?
        .scan()?
        .with_row_indices(indices)
        .into_array_stream()?
        .read_all()
        .await?;

    // We canonicalize / decompress for equivalence to Arrow's `RecordBatch`es.
    let mut ctx = SESSION.create_execution_ctx();
    // TODO(joe): should we go to a vector.
    Ok(array.execute::<Canonical>(&mut ctx)?.into_array())
}

pub async fn take_parquet(path: &Path, indices: Vec<u64>) -> anyhow::Result<RecordBatch> {
    let file = tokio::fs::File::open(path).await?;

    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        file,
        ArrowReaderOptions::new().with_page_index(true),
    )
    .await?;

    // We figure out which row groups we need to read and a selection filter for each of them.
    let mut row_groups = HashMap::new();
    let row_group_offsets = iter::once(0)
        .chain(
            builder
                .metadata()
                .row_groups()
                .iter()
                .map(RowGroupMetaData::num_rows),
        )
        .scan(0i64, |acc, x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<_>>();

    for idx in indices {
        let row_group_idx = row_group_offsets
            .binary_search(&(idx as i64))
            .unwrap_or_else(|e| e - 1);
        row_groups
            .entry(row_group_idx)
            .or_insert_with(Vec::new)
            .push((idx as i64) - row_group_offsets[row_group_idx]);
    }
    let row_group_indices = row_groups
        .keys()
        .sorted()
        .map(|i| row_groups[i].clone())
        .collect_vec();

    let reader = builder
        .with_row_groups(row_groups.keys().copied().collect_vec())
        // FIXME(ngates): our indices code assumes the batch size == the row group sizes
        .with_batch_size(10_000_000)
        .build()?;

    let schema = reader.schema().clone();

    let batches = reader
        .enumerate()
        .map(|(idx, batch)| {
            let batch = batch.unwrap();
            let indices = PrimitiveArray::<Int64Type>::from(row_group_indices[idx].clone());
            take_record_batch(&batch, &indices).unwrap()
        })
        .collect::<Vec<_>>()
        .await;

    Ok(concat_batches(&schema, &batches)?)
}

/// Random accessor for Vortex format files with field projection support.
pub struct VortexProjectingAccessor {
    path: PathBuf,
    name: String,
    format: Format,
}

impl VortexProjectingAccessor {
    /// Create a new projecting Vortex accessor for the given file path.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/vortex-projected-tokio-local-disk".to_string(),
            format: Format::OnDiskVortex,
        }
    }

    /// Create a new projecting Vortex accessor with a custom name.
    pub fn with_name(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            path,
            name: name.into(),
            format: Format::OnDiskVortex,
        }
    }

    /// Create a new projecting Vortex accessor for compact format.
    pub fn compact(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/vortex-compact-projected-tokio-local-disk".to_string(),
            format: Format::VortexCompact,
        }
    }
}

#[async_trait]
impl ProjectingRandomAccessor for VortexProjectingAccessor {
    fn format(&self) -> Format {
        self.format
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }

    async fn take_projected(
        &self,
        indices: Vec<u64>,
        field_path: &FieldPath,
    ) -> anyhow::Result<usize> {
        let result = take_vortex_projected(&self.path, indices.into(), field_path).await?;
        Ok(result.len())
    }
}

/// Random accessor for Parquet format files with field projection support.
pub struct ParquetProjectingAccessor {
    path: PathBuf,
    name: String,
}

impl ParquetProjectingAccessor {
    /// Create a new projecting Parquet accessor for the given file path.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/parquet-projected-tokio-local-disk".to_string(),
        }
    }

    /// Create a new projecting Parquet accessor with a custom name.
    pub fn with_name(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            path,
            name: name.into(),
        }
    }
}

#[async_trait]
impl ProjectingRandomAccessor for ParquetProjectingAccessor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }

    async fn take_projected(
        &self,
        indices: Vec<u64>,
        field_path: &FieldPath,
    ) -> anyhow::Result<usize> {
        let result = take_parquet_projected(&self.path, indices, field_path).await?;
        Ok(result.num_rows())
    }
}

/// Build a projection expression for a nested field path.
fn build_projection_expr(field_path: &FieldPath) -> Expression {
    let mut expr = root();
    for field in field_path {
        expr = get_item(field.as_str(), expr);
    }
    expr
}

/// Take rows from a Vortex file with a projection to a nested field.
async fn take_vortex_projected(
    reader: impl AsRef<Path>,
    indices: Buffer<u64>,
    field_path: &FieldPath,
) -> anyhow::Result<ArrayRef> {
    let projection = build_projection_expr(field_path);

    let array = SESSION
        .open_options()
        .open_path(reader.as_ref())
        .await?
        .scan()?
        .with_projection(projection)
        .with_row_indices(indices)
        .into_array_stream()?
        .read_all()
        .await?;

    // We canonicalize / decompress for equivalence to Arrow's `RecordBatch`es.
    let mut ctx = SESSION.create_execution_ctx();
    Ok(array.execute::<Canonical>(&mut ctx)?.into_array())
}

/// Take rows from a Parquet file with a projection to a nested field.
pub async fn take_parquet_projected(
    path: &Path,
    indices: Vec<u64>,
    field_path: &FieldPath,
) -> anyhow::Result<RecordBatch> {
    let file = tokio::fs::File::open(path).await?;

    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        file,
        ArrowReaderOptions::new().with_page_index(true),
    )
    .await?;

    // Build projection mask for the nested field.
    // For Parquet, we need to find the leaf column indices that correspond to the nested path.
    let parquet_schema = builder.parquet_schema();
    let arrow_schema = builder.schema();

    // Find the column indices for the projected field path.
    let projection_mask = build_parquet_projection_mask(parquet_schema, arrow_schema, field_path)?;

    // We figure out which row groups we need to read and a selection filter for each of them.
    let mut row_groups = HashMap::new();
    let row_group_offsets = iter::once(0)
        .chain(
            builder
                .metadata()
                .row_groups()
                .iter()
                .map(RowGroupMetaData::num_rows),
        )
        .scan(0i64, |acc, x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<_>>();

    for idx in indices {
        let row_group_idx = row_group_offsets
            .binary_search(&(idx as i64))
            .unwrap_or_else(|e| e - 1);
        row_groups
            .entry(row_group_idx)
            .or_insert_with(Vec::new)
            .push((idx as i64) - row_group_offsets[row_group_idx]);
    }
    let row_group_indices = row_groups
        .keys()
        .sorted()
        .map(|i| row_groups[i].clone())
        .collect_vec();

    let reader = builder
        .with_projection(projection_mask)
        .with_row_groups(row_groups.keys().copied().collect_vec())
        // FIXME(ngates): our indices code assumes the batch size == the row group sizes
        .with_batch_size(10_000_000)
        .build()?;

    let schema = reader.schema().clone();

    let batches = reader
        .enumerate()
        .map(|(idx, batch)| {
            let batch = batch.unwrap();
            let indices = PrimitiveArray::<Int64Type>::from(row_group_indices[idx].clone());
            take_record_batch(&batch, &indices).unwrap()
        })
        .collect::<Vec<_>>()
        .await;

    Ok(concat_batches(&schema, &batches)?)
}

/// Build a Parquet projection mask for a nested field path.
fn build_parquet_projection_mask(
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    arrow_schema: &arrow_schema::SchemaRef,
    field_path: &FieldPath,
) -> anyhow::Result<parquet::arrow::ProjectionMask> {
    use parquet::arrow::ProjectionMask;

    if field_path.is_empty() {
        return Ok(ProjectionMask::all());
    }

    // Find the leaf column indices for the nested field.
    let leaf_indices = find_parquet_leaf_indices(parquet_schema, arrow_schema, field_path)?;

    Ok(ProjectionMask::leaves(parquet_schema, leaf_indices))
}

/// Find the leaf column indices in the Parquet schema for a nested field path.
fn find_parquet_leaf_indices(
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    arrow_schema: &arrow_schema::SchemaRef,
    field_path: &FieldPath,
) -> anyhow::Result<Vec<usize>> {
    use arrow_schema::DataType;

    // Navigate the Arrow schema to find the target field.
    let mut current_field: Option<&arrow_schema::Field> = None;

    for (i, field_name) in field_path.iter().enumerate() {
        if i == 0 {
            // Find in root schema
            current_field = arrow_schema.field_with_name(field_name).ok();
        } else if let Some(field) = current_field {
            // Navigate into nested struct
            match field.data_type() {
                DataType::Struct(fields) => {
                    current_field = fields
                        .iter()
                        .find(|f| f.name() == field_name)
                        .map(|f| f.as_ref());
                }
                _ => {
                    anyhow::bail!(
                        "Cannot navigate into non-struct field '{}' at path position {}",
                        field.name(),
                        i
                    );
                }
            }
        }
    }

    // Now find all leaf columns under this field in the Parquet schema.
    // We need to map from Arrow field to Parquet column indices.
    let parquet_path = field_path.join(".");

    let mut leaf_indices = Vec::new();
    for (idx, col) in parquet_schema.columns().iter().enumerate() {
        let col_path = col.path().string();
        if col_path.starts_with(&parquet_path) {
            // Check if it's exactly the field or a child of it
            if col_path == parquet_path || col_path.starts_with(&format!("{}.", parquet_path)) {
                leaf_indices.push(idx);
            }
        }
    }

    if leaf_indices.is_empty() {
        anyhow::bail!(
            "Could not find field path '{}' in Parquet schema",
            parquet_path
        );
    }

    Ok(leaf_indices)
}
