use std::fs;
use std::fs::create_dir_all;
use std::path::Path;
use std::sync::Arc;

use arrow_array::StructArray as ArrowStructArray;
use arrow_schema::Schema;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::MemTable;
use datafusion::prelude::{CsvReadOptions, ParquetReadOptions, SessionContext};
use tokio::fs::OpenOptions;
use vortex::aliases::hash_map::HashMap;
use vortex::array::{ChunkedArray, StructArray};
use vortex::arrow::FromArrowArray;
use vortex::dtype::DType;
use vortex::file::{VortexFileWriter, VORTEX_FILE_EXTENSION};
use vortex::sampling_compressor::SamplingCompressor;
use vortex::variants::StructArrayTrait;
use vortex::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_datafusion::memory::VortexMemTableOptions;
use vortex_datafusion::persistent::format::VortexFormat;
use vortex_datafusion::SessionContextExt;

use crate::{idempotent_async, Format, CTX, TARGET_BLOCK_BYTESIZE, TARGET_BLOCK_SIZE};

pub mod dbgen;
mod execute;
pub mod schema;

pub use execute::*;

pub const EXPECTED_ROW_COUNTS: [usize; 23] = [
    0, 4, 460, 11620, 5, 5, 1, 4, 2, 175, 37967, 1048, 2, 42, 1, 1, 18314, 1, 57, 1, 186, 411, 7,
];

// Generate table dataset.
pub async fn load_datasets<P: AsRef<Path>>(
    base_dir: P,
    format: Format,
) -> anyhow::Result<SessionContext> {
    let context = SessionContext::new();
    let base_dir = base_dir.as_ref();

    let customer = base_dir.join("customer.tbl");
    let lineitem = base_dir.join("lineitem.tbl");
    let nation = base_dir.join("nation.tbl");
    let orders = base_dir.join("orders.tbl");
    let part = base_dir.join("part.tbl");
    let partsupp = base_dir.join("partsupp.tbl");
    let region = base_dir.join("region.tbl");
    let supplier = base_dir.join("supplier.tbl");

    macro_rules! register_table {
        ($name:ident, $schema:expr) => {
            match format {
                Format::Csv => register_csv(&context, stringify!($name), &$name, $schema).await,
                Format::Arrow => register_arrow(&context, stringify!($name), &$name, $schema).await,
                Format::Parquet => {
                    register_parquet(&context, stringify!($name), &$name, $schema).await
                }
                Format::InMemoryVortex {
                    enable_pushdown, ..
                } => {
                    register_vortex(
                        &context,
                        stringify!($name),
                        &$name,
                        $schema,
                        enable_pushdown,
                    )
                    .await
                }
                Format::OnDiskVortex { enable_compression } => {
                    register_vortex_file(
                        &context,
                        stringify!($name),
                        &$name,
                        $schema,
                        enable_compression,
                    )
                    .await
                }
            }
        };
    }

    register_table!(customer, &schema::CUSTOMER)?;
    register_table!(lineitem, &schema::LINEITEM)?;
    register_table!(nation, &schema::NATION)?;
    register_table!(orders, &schema::ORDERS)?;
    register_table!(part, &schema::PART)?;
    register_table!(partsupp, &schema::PARTSUPP)?;
    register_table!(region, &schema::REGION)?;
    register_table!(supplier, &schema::SUPPLIER)?;

    Ok(context)
}

async fn register_csv(
    session: &SessionContext,
    name: &str,
    file: &Path,
    schema: &Schema,
) -> anyhow::Result<()> {
    session
        .register_csv(
            name,
            file.to_str().unwrap(),
            CsvReadOptions::default()
                .delimiter(b'|')
                .has_header(false)
                .file_extension("tbl")
                .schema(schema),
        )
        .await?;

    Ok(())
}

async fn register_arrow(
    session: &SessionContext,
    name: &str,
    file: &Path,
    schema: &Schema,
) -> anyhow::Result<()> {
    // Read CSV file into a set of Arrow RecordBatch.
    let record_batches = session
        .read_csv(
            file.to_str().unwrap(),
            CsvReadOptions::default()
                .delimiter(b'|')
                .has_header(false)
                .file_extension("tbl")
                .schema(schema),
        )
        .await?
        .collect()
        .await?;

    let mem_table = MemTable::try_new(Arc::new(schema.clone()), vec![record_batches])?;
    session.register_table(name, Arc::new(mem_table))?;

    Ok(())
}

async fn register_parquet(
    session: &SessionContext,
    name: &str,
    file: &Path,
    schema: &Schema,
) -> anyhow::Result<()> {
    let csv_file = file.to_str().unwrap();
    let pq_file = idempotent_async(
        &file.with_extension("").with_extension("parquet"),
        |pq_file| async move {
            let df = session
                .read_csv(
                    csv_file,
                    CsvReadOptions::default()
                        .delimiter(b'|')
                        .has_header(false)
                        .file_extension("tbl")
                        .schema(schema),
                )
                .await?;

            df.write_parquet(
                pq_file.as_path().as_os_str().to_str().unwrap(),
                DataFrameWriteOptions::default(),
                None,
            )
            .await?;

            Ok::<(), anyhow::Error>(())
        },
    )
    .await?;

    Ok(session
        .register_parquet(
            name,
            pq_file.as_os_str().to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await?)
}

async fn register_vortex_file(
    session: &SessionContext,
    table_name: &str,
    file: &Path,
    schema: &Schema,
    enable_compression: bool,
) -> anyhow::Result<()> {
    let vortex_dir = file.parent().unwrap().join(if enable_compression {
        "vortex_compressed"
    } else {
        "vortex_uncompressed"
    });
    create_dir_all(&vortex_dir)?;
    let output_file = &vortex_dir
        .join(file.file_name().unwrap())
        .with_extension(VORTEX_FILE_EXTENSION);
    let vtx_file = idempotent_async(output_file, |vtx_file| async move {
        let record_batches = session
            .read_csv(
                file.to_str().unwrap(),
                CsvReadOptions::default()
                    .delimiter(b'|')
                    .has_header(false)
                    .file_extension("tbl")
                    .schema(schema),
            )
            .await?
            .collect()
            .await?;

        // Create a ChunkedArray from the set of chunks.
        let sts = record_batches
            .into_iter()
            .map(ArrayData::try_from)
            .map(|a| a.unwrap().into_struct().unwrap())
            .collect::<Vec<_>>();

        let mut arrays_map: HashMap<Arc<str>, Vec<ArrayData>> = HashMap::default();
        let mut types_map: HashMap<Arc<str>, DType> = HashMap::default();

        for st in sts.into_iter() {
            let struct_dtype = st.dtype().as_struct().unwrap();
            let names = struct_dtype.names().iter();
            let types = struct_dtype.dtypes().iter();

            for (field_name, field_type) in names.zip(types) {
                let val = arrays_map.entry(field_name.clone()).or_default();
                val.push(st.field_by_name(field_name).unwrap());

                types_map.insert(field_name.clone(), field_type.clone());
            }
        }

        let fields = schema
            .fields()
            .iter()
            .map(|field| {
                let name: Arc<str> = field.name().as_str().into();
                let dtype = types_map[&name].clone();
                let chunks = arrays_map.remove(&name).unwrap();
                let mut chunked_child = ChunkedArray::try_new(chunks, dtype).unwrap();
                if !enable_compression {
                    chunked_child = chunked_child
                        .rechunk(TARGET_BLOCK_BYTESIZE, TARGET_BLOCK_SIZE)
                        .unwrap()
                }

                (name, chunked_child.into_array())
            })
            .collect::<Vec<_>>();

        let data = StructArray::from_fields(&fields)?.into_array();

        let data = if enable_compression {
            let compressor = SamplingCompressor::default();
            compressor.compress(&data, None)?.into_array()
        } else {
            data
        };

        let f = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&vtx_file)
            .await?;

        let mut writer = VortexFileWriter::new(f);
        writer = writer.write_array_columns(data).await?;
        writer.finalize().await?;

        anyhow::Ok(())
    })
    .await?;

    let format = Arc::new(VortexFormat::new(CTX.clone()));
    let table_url = ListingTableUrl::parse(vtx_file.to_str().unwrap())?;
    let config = ListingTableConfig::new(table_url)
        .with_listing_options(ListingOptions::new(format as _))
        .infer_schema(&session.state())
        .await?;

    let listing_table = Arc::new(ListingTable::try_new(config)?);

    session.register_table(table_name, listing_table as _)?;

    Ok(())
}

async fn register_vortex(
    session: &SessionContext,
    name: &str,
    file: &Path,
    schema: &Schema,
    enable_pushdown: bool,
) -> anyhow::Result<()> {
    let record_batches = session
        .read_csv(
            file.to_str().unwrap(),
            CsvReadOptions::default()
                .delimiter(b'|')
                .has_header(false)
                .file_extension("tbl")
                .schema(schema),
        )
        .await?
        .collect()
        .await?;

    // Create a ChunkedArray from the set of chunks.
    let chunks: Vec<ArrayData> = record_batches
        .into_iter()
        .map(ArrowStructArray::from)
        .map(|struct_array| ArrayData::from_arrow(&struct_array, false))
        .collect();

    let dtype = chunks[0].dtype().clone();
    let chunked_array = ChunkedArray::try_new(chunks, dtype)?.into_array();

    session.register_mem_vortex_opts(
        name,
        chunked_array,
        VortexMemTableOptions::default().with_pushdown(enable_pushdown),
    )?;

    Ok(())
}

/// Load a table as an uncompressed Vortex array.
pub async fn load_table(data_dir: impl AsRef<Path>, name: &str, schema: &Schema) -> ArrayData {
    // Create a local session to load the CSV file from the path.
    let path = data_dir
        .as_ref()
        .to_owned()
        .join(format!("{name}.tbl"))
        .to_str()
        .unwrap()
        .to_string();
    let record_batches = SessionContext::new()
        .read_csv(
            &path,
            CsvReadOptions::default()
                .delimiter(b'|')
                .has_header(false)
                .file_extension("tbl")
                .schema(schema),
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let chunks: Vec<ArrayData> = record_batches
        .into_iter()
        .map(ArrowStructArray::from)
        .map(|struct_array| ArrayData::from_arrow(&struct_array, false))
        .collect();

    let dtype = chunks[0].dtype().clone();

    ChunkedArray::try_new(chunks, dtype).unwrap().into_array()
}

pub fn tpch_queries() -> impl Iterator<Item = (usize, Vec<String>)> {
    (1..=22).map(|q| (q, tpch_query(q)))
}

fn tpch_query(query_idx: usize) -> Vec<String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tpch")
        .join(format!("q{}.sql", query_idx));
    fs::read_to_string(manifest_dir)
        .unwrap()
        .split(';')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}
