// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::stream::FuturesUnordered;
use futures::{StreamExt, TryStreamExt, stream};
use log::info;
use parquet::arrow::AsyncArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tpchgen::generators::{
    CustomerGenerator, LineItemGenerator, NationGenerator, OrderGenerator, PartGenerator,
    PartSuppGenerator, RegionGenerator, SupplierGenerator,
};
use tpchgen_arrow::RecordBatchIterator;
use vortex::ArrayRef;
use vortex::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexExpect;
use vortex::file::VortexWriteOptions;
use vortex::stream::ArrayStreamAdapter;

use crate::utils::file_utils::idempotent_async;
use crate::{Format, IdempotentPath};

/// Configuration for TPC-H data generation
#[derive(Debug, Clone)]
pub struct TpchGenOptions {
    /// Scale factor (0.01, 0.1, 1, 10, 100, 1000)
    pub scale_factor: String,
    /// Output directory
    pub output_dir: PathBuf,
    /// Output format
    pub format: Format,
    /// Batch size for streaming
    pub batch_size: usize,
    /// Number of partitions for parallel generation
    pub num_partitions: i32,
    /// Current partition (1-indexed)
    pub partition: i32,
}

impl Default for TpchGenOptions {
    fn default() -> Self {
        Self {
            scale_factor: "1.0".to_string(),
            output_dir: "tpch".to_data_path(),
            format: Format::Parquet,
            batch_size: 8192 * 64,
            num_partitions: 1,
            partition: 1,
        }
    }
}

impl TpchGenOptions {
    pub fn new(scale_factor: String, output_dir: impl AsRef<Path>) -> Self {
        Self {
            scale_factor,
            output_dir: output_dir.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    pub fn with_format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_partitions(mut self, num_partitions: i32, partition: i32) -> Self {
        self.num_partitions = num_partitions;
        self.partition = partition;
        self
    }
}

/// Generate all TPC-H tables for a single scale factor
pub async fn generate_tpch_tables(options: &TpchGenOptions) -> Result<()> {
    fs::create_dir_all(&options.output_dir)?;

    let tables = [
        ("nation", TableGenerator::Nation),
        ("region", TableGenerator::Region),
        ("part", TableGenerator::Part),
        ("supplier", TableGenerator::Supplier),
        ("customer", TableGenerator::Customer),
        ("partsupp", TableGenerator::PartSupp),
        ("orders", TableGenerator::Orders),
        ("lineitem", TableGenerator::LineItem),
    ];

    stream::iter(tables)
        .map(|(table_name, generator)| {
            info!(
                "Generating {} table for scale factor {} in format: {:?}",
                table_name, options.scale_factor, options.format
            );

            tokio::spawn(generate_table_files(table_name, generator, options.clone()))
        })
        .collect::<FuturesUnordered<_>>()
        .await
        .try_collect::<Vec<_>>()
        .await?;

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum TableGenerator {
    Nation,
    Region,
    Part,
    Supplier,
    Customer,
    PartSupp,
    Orders,
    LineItem,
}

/// Generate files for a specific table in streaming fashion
async fn generate_table_files(
    table_name: &str,
    generator: TableGenerator,
    options: TpchGenOptions,
) -> Result<()> {
    // Determine output path based on format
    let (output_path, format_name) = match options.format {
        Format::Parquet | Format::Arrow | Format::OnDiskDuckDB => {
            let output_dir = options.output_dir.join(Format::Parquet.to_string());
            fs::create_dir_all(&output_dir)?;
            (
                output_dir.join(format!("{}.parquet", table_name)),
                "Parquet",
            )
        }
        Format::OnDiskVortex => {
            let output_dir = options.output_dir.join(Format::OnDiskVortex.to_string());
            fs::create_dir_all(&output_dir)?;
            (output_dir.join(format!("{}.vortex", table_name)), "Vortex")
        }
        f @ Format::Csv => {
            anyhow::bail!("{f} format is not supported by tpchgen");
        }
    };

    idempotent_async(&output_path, |path| async move {
        info!("Generating {table_name} table as {format_name}");

        // Create generator and process batches in streaming fashion
        let batch_iter = create_batch_iterator(generator, &options)?;
        let schema = batch_iter.schema().clone();

        // Create writer based on format
        let mut writer: Box<dyn FileWriter + Send> = match options.format {
            Format::Parquet | Format::Arrow | Format::OnDiskDuckDB => {
                Box::new(ParquetWriter::new(path, schema).await?)
            }
            Format::OnDiskVortex => Box::new(VortexWriter::new(path, schema)?),
            _ => unreachable!(),
        };

        for batch in batch_iter {
            writer.write_batch(&batch).await?;
        }

        writer.finalize().await?;
        Ok::<(), anyhow::Error>(())
    })
    .await?;

    Ok(())
}

/// Create a batch iterator for the specified table generator
#[allow(clippy::cast_possible_truncation)]
fn create_batch_iterator(
    generator: TableGenerator,
    options: &TpchGenOptions,
) -> Result<Box<dyn RecordBatchIterator>> {
    let scale_factor = options.scale_factor.parse::<f64>()?;
    match generator {
        TableGenerator::Nation => {
            let generator = tpchgen_arrow::NationArrow::new(NationGenerator::new(
                scale_factor,
                options.partition,
                options.num_partitions,
            ))
            .with_batch_size(options.batch_size);
            Ok(Box::new(generator))
        }
        TableGenerator::Region => {
            let generator = tpchgen_arrow::RegionArrow::new(RegionGenerator::new(
                scale_factor,
                options.partition,
                options.num_partitions,
            ))
            .with_batch_size(options.batch_size);
            Ok(Box::new(generator))
        }
        TableGenerator::Part => {
            let generator = tpchgen_arrow::PartArrow::new(PartGenerator::new(
                scale_factor,
                options.partition,
                options.num_partitions,
            ))
            .with_batch_size(options.batch_size);
            Ok(Box::new(generator))
        }
        TableGenerator::Supplier => {
            let generator = tpchgen_arrow::SupplierArrow::new(SupplierGenerator::new(
                scale_factor,
                options.partition,
                options.num_partitions,
            ))
            .with_batch_size(options.batch_size);
            Ok(Box::new(generator))
        }
        TableGenerator::Customer => {
            let generator = tpchgen_arrow::CustomerArrow::new(CustomerGenerator::new(
                scale_factor,
                options.partition,
                options.num_partitions,
            ))
            .with_batch_size(options.batch_size);
            Ok(Box::new(generator))
        }
        TableGenerator::PartSupp => {
            let generator = tpchgen_arrow::PartSuppArrow::new(PartSuppGenerator::new(
                scale_factor,
                options.partition,
                options.num_partitions,
            ))
            .with_batch_size(options.batch_size);
            Ok(Box::new(generator))
        }
        TableGenerator::Orders => {
            let generator = tpchgen_arrow::OrderArrow::new(OrderGenerator::new(
                scale_factor,
                options.partition,
                options.num_partitions,
            ))
            .with_batch_size(options.batch_size);
            Ok(Box::new(generator))
        }
        TableGenerator::LineItem => {
            let generator = tpchgen_arrow::LineItemArrow::new(LineItemGenerator::new(
                scale_factor,
                options.partition,
                options.num_partitions,
            ))
            .with_batch_size(options.batch_size);
            Ok(Box::new(generator))
        }
    }
}

/// Common interface for file writers
#[async_trait::async_trait]
trait FileWriter {
    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()>;
    async fn finalize(self: Box<Self>) -> Result<()>;
}

/// Parquet writer for streaming TPC-H data
struct ParquetWriter {
    writer: AsyncArrowWriter<TokioFile>,
}

impl ParquetWriter {
    async fn new(path: PathBuf, schema: SchemaRef) -> Result<Self> {
        let file = TokioFile::create(&path).await?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_bloom_filter_enabled(true)
            .build();
        let writer = AsyncArrowWriter::try_new(file, schema, Some(props))?;
        Ok(Self { writer })
    }
}

#[async_trait::async_trait]
impl FileWriter for ParquetWriter {
    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch).await.map_err(|e| anyhow!(e))
    }

    async fn finalize(self: Box<Self>) -> Result<()> {
        self.writer.close().await?;
        Ok(())
    }
}

/// Vortex writer for streaming TPC-H data
struct VortexWriter {
    sender: Option<mpsc::Sender<vortex::error::VortexResult<ArrayRef>>>,
    write_task: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl VortexWriter {
    fn new(path: PathBuf, schema: SchemaRef) -> Result<Self> {
        // limit the number of in flight rows.
        let (sender, receiver) = mpsc::channel(32);
        let dtype = DType::from_arrow(schema);
        let file_path = path;
        let write_task = Some(tokio::spawn(async move {
            let stream = ArrayStreamAdapter::new(dtype, ReceiverStream::new(receiver));

            let file = TokioFile::create(&file_path).await?;
            VortexWriteOptions::default()
                .write(file, stream)
                .await
                .map_err(|e| anyhow!("Vortex write failed: {}", e))?;

            Ok(())
        }));

        Ok(Self {
            sender: Some(sender),
            write_task,
        })
    }
}

#[async_trait::async_trait]
impl FileWriter for VortexWriter {
    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let array = ArrayRef::from_arrow(batch, false);
        self.sender
            .as_ref()
            .vortex_expect("sender closed early")
            .send(Ok(array))
            .await
            .map_err(|_| anyhow!("Failed to send array to write task"))
    }

    async fn finalize(mut self: Box<Self>) -> Result<()> {
        // Close the sender to signal end of stream
        drop(self.sender);

        // Wait for write task to complete
        if let Some(task) = self.write_task.take() {
            task.await
                .map_err(|e| anyhow!("Write task failed: {}", e))??;
        }

        Ok(())
    }
}
