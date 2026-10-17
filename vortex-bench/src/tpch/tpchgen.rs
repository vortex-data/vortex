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
use parquet::arrow::ArrowWriter;
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
use vortex::array::stream::ArrayStreamAdapter;
use vortex::error::VortexExpect;
use vortex::file::WriteOptionsSessionExt;
use vortex::utils::parallelism::get_available_parallelism;
use vortex_arrow::ArrowSessionExt;

use crate::CompactionStrategy;
use crate::Format;
use crate::IdempotentPath;
use crate::SESSION;
use crate::utils::file::idempotent_async;
use crate::utils::parquet::ParquetTableJob;
use crate::utils::parquet::parquet_writer_properties;
use crate::utils::parquet::write_parquet_tables;

/// The parquet-rs default, kept explicit so the shared writer settings do not change the
/// parallelism of existing TPC-H Parquet benchmarks.
const ROW_GROUP_SIZE: usize = 1024 * 1024;

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
}

impl Default for TpchGenOptions {
    fn default() -> Self {
        Self {
            scale_factor: "1.0".to_string(),
            output_dir: "tpch".to_data_path(),
            format: Format::Parquet,
            batch_size: 8192 * 64,
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

    if matches!(options.format, Format::Parquet | Format::OnDiskDuckDB) {
        return generate_parquet_tables(&tables, options).await;
    }

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

            generate_table_file(table_name, *generator, options.clone())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
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

/// Generate every table as Parquet, with all tables' parts sharing one pool of one worker per core.
async fn generate_parquet_tables(
    tables: &[(&str, TableGenerator)],
    options: TpchGenOptions,
) -> Result<()> {
    let output_dir = options.output_dir.join(Format::Parquet.name());
    let scale_factor: f64 = options.scale_factor.parse()?;
    let options = Arc::new(options);
    let jobs = tables
        .iter()
        .map(|&(table_name, generator)| {
            let num_parts = generator.num_parquet_parts(scale_factor);
            info!(
                scale_factor = options.scale_factor,
                table = table_name,
                num_parts,
                "Generating TPC-H table as parquet",
            );
            let options = Arc::clone(&options);
            ParquetTableJob {
                path: output_dir.join(format!("{table_name}.{}", Format::Parquet.ext())),
                num_parts,
                write_part: Box::new(move |part, path| {
                    write_parquet_part(
                        path,
                        create_batch_iterator(generator, &options, part + 1, num_parts)?,
                    )
                }),
            }
        })
        .collect();
    let workers = get_available_parallelism().unwrap_or(1);
    tokio::task::spawn_blocking(move || write_parquet_tables(jobs, workers)).await?
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

/// Rows per Parquet part. Each part is generated and encoded on its own thread.
const ROWS_PER_PART: f64 = (4 * ROW_GROUP_SIZE) as f64;

impl TableGenerator {
    /// Approximate row count at scale factor 1, from the TPC-H specification.
    fn rows_per_scale_factor(self) -> f64 {
        match self {
            Self::Nation | Self::Region => 0.0,
            Self::Part => 200_000.0,
            Self::Supplier => 10_000.0,
            Self::Customer => 150_000.0,
            Self::PartSupp => 800_000.0,
            Self::Orders => 1_500_000.0,
            Self::LineItem => 6_000_000.0,
        }
    }

    /// Number of parts to generate this table's Parquet file in, at most one per core.
    // `as` saturates on overflow, NaN and negatives, and the clamp bounds the result.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn num_parquet_parts(self, scale_factor: f64) -> usize {
        let parts = (self.rows_per_scale_factor() * scale_factor / ROWS_PER_PART).ceil();
        (parts as usize).clamp(1, get_available_parallelism().unwrap_or(1))
    }
}

/// Generate one file for a specific table in streaming fashion
fn generate_table_file(
    table_name: String,
    generator: TableGenerator,
    options: TpchGenOptions,
) -> Result<BoxFuture<'static, Result<()>>> {
    let write_format = match options.format {
        Format::OnDiskVortex => Format::OnDiskVortex,
        Format::VortexCompact => Format::VortexCompact,
        f => {
            anyhow::bail!("{f} format is not supported by tpchgen");
        }
    };

    let output_dir = options.output_dir.join(write_format.name());

    fs::create_dir_all(&output_dir)?;

    let output_file = output_dir.join(format!("{table_name}.{}", write_format.ext()));

    let future = async move {
        let of = output_file.clone();
        idempotent_async(output_file.to_string_lossy().as_ref(), |path| async move {
            info!(
                "Generating {table_name} table as {write_format}, at {}",
                of.to_string_lossy()
            );

            let iter = create_batch_iterator(generator, &options, 1, 1)?;

            // Create generator and process batches in streaming fashion
            let schema = Arc::clone(iter.schema());

            // Create writer based on format
            let mut writer: Box<dyn FileWriter + Send> = match write_format {
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

    Ok(future)
}

/// Create a batch iterator for part `part` (one-based) of `num_parts` of a table.
fn create_batch_iterator(
    generator: TableGenerator,
    options: &TpchGenOptions,
    part: usize,
    num_parts: usize,
) -> Result<Box<dyn RecordBatchIterator>> {
    let scale_factor = options.scale_factor.parse::<f64>()?;
    let part = i32::try_from(part)?;
    let num_parts = i32::try_from(num_parts)?;
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

/// Write one TPC-H table part as a standalone Parquet file.
fn write_parquet_part(path: &Path, batches: Box<dyn RecordBatchIterator>) -> Result<()> {
    let props = parquet_writer_properties()
        .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
        .build();
    let mut writer = ArrowWriter::try_new(
        fs::File::create(path)?,
        Arc::clone(batches.schema()),
        Some(props),
    )?;
    for batch in batches {
        writer.write(&batch)?;
    }
    writer.close()?;
    Ok(())
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
        let dtype = SESSION.arrow().from_arrow_schema(schema.as_ref())?;
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
        let schema = batch.schema();
        let array = SESSION
            .arrow()
            .from_arrow_record_batch(batch.clone(), &schema)?;
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

#[cfg(test)]
mod tests {
    use arrow_select::concat::concat_batches;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use rstest::rstest;

    use super::*;

    fn read_parquet(path: &Path) -> Result<(SchemaRef, usize, RecordBatch)> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(fs::File::open(path)?)?;
        let schema = Arc::clone(builder.schema());
        let row_groups = builder.metadata().num_row_groups();
        let batches = builder.build()?.collect::<Result<Vec<_>, _>>()?;
        Ok((
            Arc::clone(&schema),
            row_groups,
            concat_batches(&schema, &batches)?,
        ))
    }

    #[rstest]
    #[case::customer(TableGenerator::Customer)]
    #[case::partsupp(TableGenerator::PartSupp)]
    #[case::orders(TableGenerator::Orders)]
    #[case::lineitem(TableGenerator::LineItem)]
    fn parquet_parts_match_single_writer(#[case] generator: TableGenerator) -> Result<()> {
        let dir = tempfile::tempdir()?;
        let options = TpchGenOptions::new("0.1".to_string(), dir.path()).with_batch_size(10_000);

        let single = dir.path().join("single.parquet");
        write_parquet_part(&single, create_batch_iterator(generator, &options, 1, 1)?)?;

        let num_parts = 7;
        let parts = dir.path().join("parts.parquet");
        let part_options = options;
        let job = ParquetTableJob {
            path: parts.clone(),
            num_parts,
            write_part: Box::new(move |part, part_path| {
                write_parquet_part(
                    part_path,
                    create_batch_iterator(generator, &part_options, part + 1, num_parts)?,
                )
            }),
        };
        write_parquet_tables(vec![job], 3)?;

        let (single_schema, _, single_rows) = read_parquet(&single)?;
        let (parts_schema, parts_row_groups, parts_rows) = read_parquet(&parts)?;
        // The schema comparison includes the embedded Arrow schema metadata.
        assert_eq!(parts_schema, single_schema);
        assert_eq!(parts_rows, single_rows);
        assert_eq!(parts_row_groups, num_parts);
        // Only the two output files remain: the part files are cleaned up.
        assert_eq!(fs::read_dir(dir.path())?.count(), 2);
        Ok(())
    }
}
