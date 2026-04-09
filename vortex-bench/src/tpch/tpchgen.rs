// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use parquet::arrow::AsyncArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;
use tokio::sync::Semaphore;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tpchgen::generators::CustomerGenerator;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::NationGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen::generators::PartGenerator;
use tpchgen::generators::PartSuppGenerator;
use tpchgen::generators::RegionGenerator;
use tpchgen::generators::SupplierGenerator;
use tpchgen_arrow::RecordBatchIterator;
use tracing::info;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::error::VortexExpect;
use vortex::file::WriteOptionsSessionExt;

use crate::CompactionStrategy;
use crate::Format;
use crate::IdempotentPath;
use crate::SESSION;
use crate::utils::file::idempotent_async;

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
    /// The max size of uncompressed file .tbl that we should generate
    pub max_file_size_mb: Option<u64>,
}

impl Default for TpchGenOptions {
    fn default() -> Self {
        Self {
            scale_factor: "1.0".to_string(),
            output_dir: "tpch".to_data_path(),
            format: Format::Parquet,
            batch_size: 8192 * 64,
            max_file_size_mb: None,
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

    pub fn with_max_file_size_mb(mut self, max_file_size_mb: Option<u64>) -> Self {
        self.max_file_size_mb = max_file_size_mb;
        self
    }
}

/// Generate all TPC-H tables for a single scale factor
pub async fn generate_tpch_tables(options: TpchGenOptions) -> Result<()> {
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

    const MAX_CONCURRENT_FILES: usize = 32;

    let all_futures = tables
        .iter()
        .map(|(table_name, generator)| {
            info!(
                scale_factor = options.scale_factor,
                format = %options.format,
                table = table_name,
                "Generating TPC-H table",
            );
            let table_name = table_name.to_string();

            generate_table_files(table_name, *generator, options.clone())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<BoxFuture<'static, Result<()>>>>();

    let limiter = Arc::new(Semaphore::new(MAX_CONCURRENT_FILES));
    let (tx, rx) = mpsc::unbounded_channel();

    for f in all_futures {
        let limiter = Arc::clone(&limiter);
        let tx = tx.clone();
        tokio::task::spawn(async move {
            let _guard = limiter.acquire().await?;
            tx.send(f.await)?;

            anyhow::Ok(())
        });
    }

    drop(tx);

    let mut rx = UnboundedReceiverStream::new(rx);

    while let Some(r) = rx.next().await {
        r?;
    }

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

impl TableGenerator {
    // Returns the size in MBs of the table at scale factor 1, if the table is constant size
    // return None
    fn uncompressed_data_size(&self) -> Option<u64> {
        match self {
            TableGenerator::Nation => None,
            TableGenerator::Region => None,
            TableGenerator::Part => Some(113),
            TableGenerator::Supplier => Some(1),
            TableGenerator::Customer => Some(23),
            TableGenerator::PartSupp => Some(113),
            TableGenerator::Orders => Some(164),
            TableGenerator::LineItem => Some(725),
        }
    }
}

/// Generate files for a specific table in streaming fashion
fn generate_table_files(
    table_name: String,
    generator: TableGenerator,
    options: TpchGenOptions,
) -> Result<Vec<BoxFuture<'static, Result<()>>>> {
    let write_format = match options.format {
        Format::Parquet | Format::Arrow | Format::OnDiskDuckDB => Format::Parquet,
        Format::OnDiskVortex => Format::OnDiskVortex,
        Format::VortexCompact => Format::VortexCompact,
        f => {
            anyhow::bail!("{f} format is not supported by tpchgen");
        }
    };

    let output_dir = options.output_dir.join(write_format.name());

    fs::create_dir_all(&output_dir)?;

    // Calculate number of partitions without creating expensive iterators
    let num_parts = calculate_num_parts(generator, &options)?;

    let mut futures = Vec::new();

    for partition_idx in 0..num_parts {
        let output_file = output_dir.join(format!(
            "{table_name}_{partition_idx}.{}",
            write_format.ext()
        ));

        // Clone necessary data for the async closure
        let options_clone = options.clone();
        let table_name = table_name.to_string();

        let future = async move {
            let of = output_file.clone();
            idempotent_async(output_file.to_string_lossy().as_ref(), |path| async move {
                info!(
                    "Generating {table_name} table as {write_format}, at {}",
                    of.to_string_lossy()
                );

                // Create the specific iterator for this partition only when we need to generate
                let iter = create_single_batch_iterator(generator, &options_clone, partition_idx)?;

                // Create generator and process batches in streaming fashion
                let schema = Arc::clone(iter.schema());

                // Create writer based on format
                let mut writer: Box<dyn FileWriter + Send> = match write_format {
                    Format::Parquet => Box::new(ParquetWriter::new(path, schema).await?),
                    Format::OnDiskVortex => Box::new(VortexWriter::new(
                        path,
                        schema,
                        CompactionStrategy::Default,
                    )?),
                    Format::VortexCompact => Box::new(VortexWriter::new(
                        path,
                        schema,
                        CompactionStrategy::Compact,
                    )?),
                    _ => unreachable!(),
                };

                for batch in iter {
                    writer.write_batch(&batch).await?;
                }

                writer.finalize().await?;
                Ok::<(), anyhow::Error>(())
            })
            .await?;
            Ok(())
        }
        .boxed();

        futures.push(future);
    }

    Ok(futures)
}

/// Calculate the number of partitions without creating expensive iterators
#[allow(clippy::cast_possible_truncation)]
fn calculate_num_parts(generator: TableGenerator, options: &TpchGenOptions) -> Result<usize> {
    let scale_factor = options.scale_factor.parse::<f64>()?;

    let num_parts = if let Some((data_size, max_file_size)) = generator
        .uncompressed_data_size()
        .zip(options.max_file_size_mb)
    {
        #[allow(clippy::cast_precision_loss)]
        let file_size = (data_size as f64 * scale_factor).ceil() as u64;
        file_size.div_ceil(max_file_size)
    } else {
        1
    };

    Ok(num_parts as usize)
}

/// Create a single batch iterator for a specific partition
#[allow(clippy::cast_possible_truncation)]
fn create_single_batch_iterator(
    generator: TableGenerator,
    options: &TpchGenOptions,
    partition_idx: usize,
) -> Result<Box<dyn RecordBatchIterator>> {
    let scale_factor = options.scale_factor.parse::<f64>()?;
    let num_parts = calculate_num_parts(generator, options)? as i32;
    let part = (partition_idx + 1) as i32; // 1-indexed
    let batch_size = options.batch_size;

    let iterator: Box<dyn RecordBatchIterator> = match generator {
        TableGenerator::Nation => Box::new(
            tpchgen_arrow::NationArrow::new(NationGenerator::new(scale_factor, part, num_parts))
                .with_batch_size(batch_size),
        ),
        TableGenerator::Region => Box::new(
            tpchgen_arrow::RegionArrow::new(RegionGenerator::new(scale_factor, part, num_parts))
                .with_batch_size(batch_size),
        ),
        TableGenerator::Part => Box::new(
            tpchgen_arrow::PartArrow::new(PartGenerator::new(scale_factor, part, num_parts))
                .with_batch_size(batch_size),
        ),
        TableGenerator::Supplier => Box::new(
            tpchgen_arrow::SupplierArrow::new(SupplierGenerator::new(
                scale_factor,
                part,
                num_parts,
            ))
            .with_batch_size(batch_size),
        ),
        TableGenerator::Customer => Box::new(
            tpchgen_arrow::CustomerArrow::new(CustomerGenerator::new(
                scale_factor,
                part,
                num_parts,
            ))
            .with_batch_size(batch_size),
        ),
        TableGenerator::PartSupp => Box::new(
            tpchgen_arrow::PartSuppArrow::new(PartSuppGenerator::new(
                scale_factor,
                part,
                num_parts,
            ))
            .with_batch_size(batch_size),
        ),
        TableGenerator::Orders => Box::new(
            tpchgen_arrow::OrderArrow::new(OrderGenerator::new(scale_factor, part, num_parts))
                .with_batch_size(batch_size),
        ),
        TableGenerator::LineItem => Box::new(
            tpchgen_arrow::LineItemArrow::new(LineItemGenerator::new(
                scale_factor,
                part,
                num_parts,
            ))
            .with_batch_size(batch_size),
        ),
    };

    Ok(iterator)
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
    fn new(
        path: PathBuf,
        schema: SchemaRef,
        compaction_strategy: CompactionStrategy,
    ) -> Result<Self> {
        // Increase buffer size to avoid backpressure issues
        let (sender, receiver) = mpsc::channel(2);
        let dtype = DType::from_arrow(schema);
        let file_path = path;
        let write_task = Some(tokio::spawn(async move {
            let stream = ArrayStreamAdapter::new(dtype, ReceiverStream::new(receiver));

            let mut file = TokioFile::create(&file_path).await?;
            compaction_strategy
                .apply_options(SESSION.write_options())
                .write(&mut file, stream)
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
        let array = ArrayRef::from_arrow(batch, false)?;
        self.sender
            .as_ref()
            .vortex_expect("sender closed early")
            .send(Ok(array))
            .await
            .map_err(|_| anyhow!("Failed to send array to write task"))
    }

    async fn finalize(mut self: Box<Self>) -> Result<()> {
        // Close the sender to signal end of stream
        self.sender.take();

        // Wait for write task to complete
        if let Some(task) = self.write_task.take() {
            task.await
                .map_err(|e| anyhow!("Write task failed: {}", e))??;
        }

        Ok(())
    }
}
