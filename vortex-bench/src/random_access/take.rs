// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fs::File as StdFile;
use std::iter::once;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use arrow_array::PrimitiveArray;
use arrow_array::types::Int64Type;
use arrow_ipc::reader::FileReader;
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use async_trait::async_trait;
use futures::stream;
use itertools::Itertools;
use parking_lot::Mutex;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::file::metadata::PageIndexPolicy;
use stream::StreamExt;
use tokio::fs::File;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
use crate::SESSION;
use crate::random_access::ARROW_ROW_OFFSETS_METADATA_KEY;
use crate::random_access::RandomAccessor;
use crate::random_access::RandomAccessorRet;

/// Random accessor for uncompressed Arrow IPC files.
pub struct ArrowRandomAccessor {
    name: String,
    reader: Mutex<FileReader<StdFile>>,
    row_offsets: Vec<u64>,
    schema: arrow_schema::SchemaRef,
}

impl ArrowRandomAccessor {
    pub fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        let reader = FileReader::try_new(StdFile::open(path)?, None)?;
        let row_offsets = reader
            .custom_metadata()
            .get(ARROW_ROW_OFFSETS_METADATA_KEY)
            .context("Arrow IPC file is missing row-offset metadata")?
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<u64>, _>>()?;
        anyhow::ensure!(
            row_offsets.len() == reader.num_batches() + 1,
            "Arrow IPC row-offset metadata does not match the record batches"
        );
        anyhow::ensure!(
            row_offsets.first() == Some(&0),
            "Arrow IPC row offsets must start at zero"
        );
        anyhow::ensure!(
            row_offsets.windows(2).all(|window| window[0] <= window[1]),
            "Arrow IPC row offsets must be sorted"
        );
        let schema = reader.schema();
        Ok(Self {
            name: name.into(),
            reader: Mutex::new(reader),
            row_offsets,
            schema,
        })
    }
}

#[async_trait]
impl RandomAccessor for ArrowRandomAccessor {
    fn format(&self) -> Format {
        Format::Arrow
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        let mut by_batch = BTreeMap::<usize, Vec<i64>>::new();
        for &index in indices {
            let batch_idx = self.row_offsets.partition_point(|offset| *offset <= index);
            anyhow::ensure!(
                batch_idx > 0 && batch_idx < self.row_offsets.len(),
                "Arrow row index {index} is out of bounds"
            );
            let batch_idx = batch_idx - 1;
            by_batch
                .entry(batch_idx)
                .or_default()
                .push(i64::try_from(index - self.row_offsets[batch_idx])?);
        }

        let mut reader = self.reader.lock();
        let mut batches = Vec::with_capacity(by_batch.len());
        for (batch_idx, local_indices) in by_batch {
            reader.set_index(batch_idx)?;
            let batch = reader
                .next()
                .context("Arrow IPC record batch is missing")??;
            let indices = PrimitiveArray::<Int64Type>::from(local_indices);
            batches.push(take_record_batch(&batch, &indices)?);
        }

        Ok(RandomAccessorRet::RecordBatch(concat_batches(
            &self.schema,
            &batches,
        )?))
    }
}

/// Random accessor for Vortex format files.
///
/// The file handle is opened at construction time and reused across `take()` calls.
pub struct VortexRandomAccessor {
    name: String,
    format: Format,
    file: VortexFile,
}

impl VortexRandomAccessor {
    /// Open a Vortex file and return a ready-to-use accessor.
    pub async fn open(
        path: impl AsRef<std::path::Path>,
        name: impl Into<String>,
        format: Format,
    ) -> anyhow::Result<Self> {
        let file = SESSION
            .open_options()
            .with_layout_reader_cache()
            .open_path(path.as_ref())
            .await?;
        Ok(Self {
            name: name.into(),
            format,
            file,
        })
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

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        let indices_buf: Buffer<u64> = Buffer::from(indices.to_vec());
        let array = self
            .file
            .scan()?
            .with_row_indices(StrictSortedBuffer::try_new(indices_buf)?)
            .into_array_stream()?
            .read_all()
            .await?;

        // We canonicalize / decompress for equivalence to Arrow's `RecordBatch`es.
        let mut ctx = SESSION.create_execution_ctx();
        let canonical = array.execute::<Canonical>(&mut ctx)?.into_array();
        Ok(RandomAccessorRet::ArrayRef(canonical))
    }
}

/// Random accessor for Parquet format files.
///
/// Parquet footer and row group offsets are parsed at construction time and
/// reused to map indices to row groups in each `take()` call.
pub struct ParquetRandomAccessor {
    name: String,
    /// Cumulative row offsets per row group (length = num_row_groups + 1).
    row_group_offsets: Vec<i64>,
    /// Cached Arrow reader metadata (footer) to avoid re-parsing on each take.
    arrow_metadata: ArrowReaderMetadata,
    /// Path to the Parquet file (for re-opening on each take).
    path: PathBuf,
}

impl ParquetRandomAccessor {
    /// Open a Parquet file, parse the footer, and return a ready-to-use accessor.
    pub async fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        let mut file = File::open(&path).await?;
        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let arrow_metadata = ArrowReaderMetadata::load_async(&mut file, options).await?;

        let row_group_offsets = once(0)
            .chain(
                arrow_metadata
                    .metadata()
                    .row_groups()
                    .iter()
                    .map(|rg| rg.num_rows()),
            )
            .scan(0i64, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect::<Vec<_>>();

        Ok(Self {
            name: name.into(),
            row_group_offsets,
            arrow_metadata,
            path,
        })
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

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        // Map indices to row groups.
        let mut row_groups = HashMap::new();
        for &idx in indices {
            let row_group_idx = self
                .row_group_offsets
                .binary_search(&(idx as i64))
                .unwrap_or_else(|e| e - 1);
            row_groups
                .entry(row_group_idx)
                .or_insert_with(Vec::new)
                .push((idx as i64) - self.row_group_offsets[row_group_idx]);
        }

        let sorted_row_group_keys = row_groups.keys().copied().sorted().collect_vec();
        let row_group_indices = sorted_row_group_keys
            .iter()
            .map(|i| row_groups[i].clone())
            .collect_vec();

        // Re-open the file but reuse cached metadata (avoids re-parsing the footer).
        let file = File::open(&self.path).await?;
        let builder =
            ParquetRecordBatchStreamBuilder::new_with_metadata(file, self.arrow_metadata.clone());

        let reader = builder
            .with_row_groups(sorted_row_group_keys)
            // FIXME(ngates): our indices code assumes the batch size == the row group sizes
            .with_batch_size(10_000_000)
            .build()?;

        let schema = Arc::clone(reader.schema());

        let batches = reader
            .enumerate()
            .map(|(idx, batch)| {
                let batch = batch.unwrap();
                let indices = PrimitiveArray::<Int64Type>::from(row_group_indices[idx].clone());
                take_record_batch(&batch, &indices).unwrap()
            })
            .collect::<Vec<_>>()
            .await;

        let result = concat_batches(&schema, &batches)?;
        Ok(RandomAccessorRet::RecordBatch(result))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_ipc::writer::FileWriter;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;

    use super::*;

    #[tokio::test]
    async fn arrow_random_accessor_takes_rows_across_record_batches() -> anyhow::Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        {
            let mut writer = FileWriter::try_new(file.reopen()?, schema.as_ref())?;
            writer.write(&RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![0, 1]))],
            )?)?;
            writer.write(&RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![2, 3, 4]))],
            )?)?;
            writer.write_metadata(ARROW_ROW_OFFSETS_METADATA_KEY, "0,2,5");
            writer.finish()?;
        }

        let accessor = ArrowRandomAccessor::open(file.path().to_path_buf(), "arrow")?;
        let RandomAccessorRet::RecordBatch(actual) = accessor.take(&[1, 3, 4]).await? else {
            anyhow::bail!("Arrow accessor returned a Vortex array")
        };
        let expected =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 3, 4]))])?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
