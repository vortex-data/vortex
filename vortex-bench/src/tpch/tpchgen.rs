// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_schema::SchemaRef;
use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use parquet::arrow::ArrowWriter;
use parquet::column::writer::ColumnCloseResult;
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
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
use crate::utils::parquet::parquet_writer_properties;
use crate::utils::split;

/// The parquet-rs default, kept explicit so the shared writer settings do not change the
/// parallelism of existing TPC-H Parquet benchmarks.
const ROW_GROUP_SIZE: usize = 1024 * 1024;

// Compressed size of each TPC-H table at scale factor 1, used to split tables into files of about
// `split::TARGET_FILE_COMPRESSED_BYTES`: the mean of the Parquet and default Vortex sizes, measured at
// scale factor 10. Every table but nation and region scales linearly with the scale factor. At
// this size, a `lineitem` file holds about 7.4M rows.
const PART_BYTES_SF1: u64 = 4_773_797;
const SUPPLIER_BYTES_SF1: u64 = 575_964;
const CUSTOMER_BYTES_SF1: u64 = 8_755_891;
const PARTSUPP_BYTES_SF1: u64 = 27_226_229;
const ORDERS_BYTES_SF1: u64 = 45_329_810;
const LINEITEM_BYTES_SF1: u64 = 169_204_254;

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

    let scale_factor = options.scale_factor.parse::<f64>()?;
    let mut all_futures = Vec::new();
    for generator in TableGenerator::ALL {
        let num_parts = generator.num_parts(scale_factor)?;
        info!(
            scale_factor = options.scale_factor,
            format = %options.format,
            table = generator.name(),
            num_parts,
            "Generating TPC-H table",
        );
        for part in 1..=num_parts {
            all_futures.push(generate_table_file(
                generator,
                part,
                num_parts,
                options.clone(),
            )?);
        }
    }

    let max_concurrent_files = get_available_parallelism().unwrap_or(1);
    let limiter = Arc::new(Semaphore::new(max_concurrent_files));
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
    const ALL: [Self; 8] = [
        Self::Nation,
        Self::Region,
        Self::Part,
        Self::Supplier,
        Self::Customer,
        Self::PartSupp,
        Self::Orders,
        Self::LineItem,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Nation => "nation",
            Self::Region => "region",
            Self::Part => "part",
            Self::Supplier => "supplier",
            Self::Customer => "customer",
            Self::PartSupp => "partsupp",
            Self::Orders => "orders",
            Self::LineItem => "lineitem",
        }
    }

    /// Compressed size of the table at scale factor 1. Nation and region do not grow with the
    /// scale factor and always fit in one file.
    fn compressed_bytes_sf1(self) -> Option<u64> {
        match self {
            Self::Nation | Self::Region => None,
            Self::Part => Some(PART_BYTES_SF1),
            Self::Supplier => Some(SUPPLIER_BYTES_SF1),
            Self::Customer => Some(CUSTOMER_BYTES_SF1),
            Self::PartSupp => Some(PARTSUPP_BYTES_SF1),
            Self::Orders => Some(ORDERS_BYTES_SF1),
            Self::LineItem => Some(LINEITEM_BYTES_SF1),
        }
    }

    fn estimated_rows(self, scale_factor: f64) -> i64 {
        match self {
            Self::Nation => 25,
            Self::Region => 5,
            Self::Part => PartGenerator::calculate_row_count(scale_factor, 1, 1),
            Self::Supplier => SupplierGenerator::calculate_row_count(scale_factor, 1, 1),
            Self::Customer => CustomerGenerator::calculate_row_count(scale_factor, 1, 1),
            Self::PartSupp => PartSuppGenerator::calculate_row_count(scale_factor, 1, 1),
            Self::Orders => OrderGenerator::calculate_row_count(scale_factor, 1, 1),
            Self::LineItem => 4 * OrderGenerator::calculate_row_count(scale_factor, 1, 1),
        }
    }

    /// Number of files to split the table into at `scale_factor`.
    fn num_parts(self, scale_factor: f64) -> Result<usize> {
        let Some(bytes_sf1) = self.compressed_bytes_sf1() else {
            return Ok(1);
        };
        #[expect(
            clippy::cast_precision_loss,
            reason = "table sizes are far below the f64 limit"
        )]
        let bytes = bytes_sf1 as f64 * scale_factor;
        Ok(split::num_files(bytes))
    }
}

/// Generate one file for a specific table in streaming fashion
fn generate_table_file(
    generator: TableGenerator,
    part: usize,
    num_parts: usize,
    options: TpchGenOptions,
) -> Result<BoxFuture<'static, Result<()>>> {
    let write_format = match options.format {
        Format::Parquet | Format::OnDiskDuckDB => Format::Parquet,
        Format::OnDiskVortex => Format::OnDiskVortex,
        Format::VortexCompact => Format::VortexCompact,
        f => {
            anyhow::bail!("{f} format is not supported by tpchgen");
        }
    };

    let output_dir = options.output_dir.join(write_format.name());

    fs::create_dir_all(&output_dir)?;

    let table_name = generator.name();
    let output_file = output_dir.join(split::part_file_name(table_name, part, write_format.ext()));

    let future = async move {
        let of = output_file.clone();
        idempotent_async(output_file.to_string_lossy().as_ref(), |path| async move {
            info!(
                "Generating {table_name} table as {write_format}, at {}",
                of.to_string_lossy()
            );

            if matches!(write_format, Format::Parquet) {
                return write_parquet(generator, &options, part, num_parts, &path).await;
            }

            let iter = create_batch_iterator(generator, &options, part, num_parts)?;
            let schema = iter.schema();

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
                writer.write_batch(&batch?).await?;
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

/// Number of parts generated in parallel for a table written as one Parquet file.
fn table_num_parts(generator: TableGenerator, scale_factor: f64) -> usize {
    let rows = usize::try_from(generator.estimated_rows(scale_factor)).unwrap_or(1);
    get_available_parallelism()
        .unwrap_or(1)
        .min(rows.div_ceil(ROW_GROUP_SIZE))
        .max(1)
}

/// Write one part of a table as Parquet.
///
/// A table split into several files already writes its files in parallel. A table written as one
/// file is generated in parallel parts that are then concatenated.
async fn write_parquet(
    generator: TableGenerator,
    options: &TpchGenOptions,
    part: usize,
    num_parts: usize,
    output: &Path,
) -> Result<()> {
    if num_parts > 1 {
        let options = options.clone();
        let output = output.to_path_buf();
        return tokio::task::spawn_blocking(move || {
            write_parquet_file(generator, &options, part, num_parts, &output)
        })
        .await?;
    }

    let scale_factor = options.scale_factor.parse::<f64>()?;
    let gen_parts = table_num_parts(generator, scale_factor);
    write_parquet_parts(generator, options, gen_parts, output).await
}

async fn write_parquet_parts(
    generator: TableGenerator,
    options: &TpchGenOptions,
    num_parts: usize,
    output: &Path,
) -> Result<()> {
    let mut tasks = Vec::with_capacity(num_parts);
    for part in 1..=num_parts {
        let options = options.clone();
        let part_path = output.with_extension(format!("part{part}"));
        tasks.push(tokio::task::spawn_blocking(move || {
            write_parquet_file(generator, &options, part, num_parts, &part_path)?;
            Ok::<_, anyhow::Error>(part_path)
        }));
    }

    let mut part_paths = Vec::with_capacity(num_parts);
    for task in tasks {
        part_paths.push(task.await??);
    }

    let output = output.to_path_buf();
    tokio::task::spawn_blocking(move || concat_parquet_parts(&part_paths, &output)).await?
}

fn write_parquet_file(
    generator: TableGenerator,
    options: &TpchGenOptions,
    part: usize,
    num_parts: usize,
    output: &Path,
) -> Result<()> {
    let iter = create_batch_iterator(generator, options, part, num_parts)?;
    let schema = iter.schema();
    let props = parquet_writer_properties()
        .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
        .build();
    let mut writer = ArrowWriter::try_new(fs::File::create(output)?, schema, Some(props))?;
    for batch in iter {
        writer.write(&batch?)?;
    }
    writer.close()?;

    Ok(())
}

fn concat_parquet_parts(parts: &[PathBuf], output: &Path) -> Result<()> {
    let inputs = parts
        .iter()
        .map(|part| -> Result<_> {
            let reader = fs::File::open(part)?;
            let metadata = ParquetMetaDataReader::new()
                .with_page_index_policy(PageIndexPolicy::Optional)
                .parse_and_finish(&reader)?;
            Ok((reader, metadata))
        })
        .collect::<Result<Vec<_>>>()?;

    let root_schema = inputs[0].1.file_metadata().schema_descr().root_schema_ptr();
    let mut writer = SerializedFileWriter::new(
        fs::File::create(output)?,
        root_schema,
        Arc::new(WriterProperties::builder().build()),
    )?;

    for (input, metadata) in &inputs {
        let column_indexes = metadata.column_index();
        let offset_indexes = metadata.offset_index();
        for (rg_idx, rg) in metadata.row_groups().iter().enumerate() {
            let rg_column_indexes = column_indexes.and_then(|ci| ci.get(rg_idx));
            let rg_offset_indexes = offset_indexes.and_then(|oi| oi.get(rg_idx));
            let mut rg_out = writer.next_row_group()?;
            for (col_idx, column) in rg.columns().iter().enumerate() {
                let result = ColumnCloseResult {
                    bytes_written: column.compressed_size() as _,
                    rows_written: rg.num_rows() as _,
                    metadata: column.clone(),
                    bloom_filter: None,
                    column_index: rg_column_indexes.and_then(|row| row.get(col_idx)).cloned(),
                    offset_index: rg_offset_indexes.and_then(|row| row.get(col_idx)).cloned(),
                };
                rg_out.append_column(input, result)?;
            }
            rg_out.close()?;
        }
    }
    writer.close()?;

    for part in parts {
        fs::remove_file(part)?;
    }

    Ok(())
}

/// Create a batch iterator for one part (1-based) of a table split into `num_parts` parts.
fn create_batch_iterator(
    generator: TableGenerator,
    options: &TpchGenOptions,
    part: usize,
    num_parts: usize,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    let scale_factor = options.scale_factor.parse::<f64>()?;
    let part = i32::try_from(part)?;
    let num_parts = i32::try_from(num_parts)?;
    let batch_size = options.batch_size;

    let iterator: Box<dyn RecordBatchReader + Send> = match generator {
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
