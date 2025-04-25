use std::borrow::Cow;
use std::fmt::{self, Display};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, LazyLock};

use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use clap::ValueEnum;
use datafusion::datasource::file_format::FileFormat;
use datafusion::datasource::file_format::csv::CsvFormat;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use datafusion_common::{DFSchema, Result, TableReference};
use futures::future::join_all;
use futures::{StreamExt, TryStreamExt, stream};
use humansize::{DECIMAL, format_size};
use regex::Regex;
use tokio::fs::File;
use tokio::process::Command as TokioCommand;
use tokio::runtime::Handle;
use tracing::{debug, info};
use url::Url;
use vortex::aliases::hash_map::HashMap;
use vortex::arrays::ChunkedArray;
use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::file::{VortexOpenOptions, VortexWriteOptions};
use vortex::stream::ArrayStreamExt;
use vortex::{Array, ArrayRef};
use vortex_datafusion::persistent::VortexFormat;

use crate::conversions::parquet_to_vortex;
use crate::datasets::Dataset;
use crate::datasets::data_downloads::{decompress_bz2, download_data};
use crate::{IdempotentPath, idempotent_async, vortex_panic};

pub static PBI_DATASETS: LazyLock<PBIDatasets> = LazyLock::new(|| {
    PBIDatasets::try_new(fetch_schemas_and_queries().expect("failed to fetch public bi queries"))
        .expect("failed to construct PBI Datasets")
});

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, ValueEnum)]
#[clap(rename_all = "PascalCase")]
pub enum PBIDataset {
    Arade,
    Bimbo,
    CMSprovider,
    CityMaxCapita,
    CommonGovernment,
    Corporations,
    Eixo,
    Euro2016,
    Food,
    Generico,
    HashTags,
    Hatred,
    IGlocations1,
    IGlocations2,
    IUBLibrary,
    MLB,
    MedPayment1,
    MedPayment2,
    Medicare1,
    Medicare2,
    Medicare3,
    Motos,
    MulheresMil,
    NYC,
    PanCreactomy1,
    PanCreactomy2,
    Physicians,
    Provider,
    RealEstate1,
    RealEstate2,
    Redfin1,
    Redfin2,
    Redfin3,
    Redfin4,
    Rentabilidad,
    Romance,
    SalariesFrance,
    TableroSistemaPenal,
    Taxpayer,
    Telco,
    TrainsUK1,
    TrainsUK2,
    USCensus,
    Uberlandia,
    Wins,
    YaleLanguages,
}

pub fn fetch_schemas_and_queries() -> VortexResult<PathBuf> {
    let base_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("public_bi");
    let output = Command::new(
        base_dir
            .join("fetch_schemas_and_queries.sh")
            .to_str()
            .unwrap(),
    )
    .output()?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        vortex_bail!("public_bi fetch failed: stdout=\"{stdout}\", stderr=\"{stderr}\"");
    }
    Ok(base_dir)
}

#[derive(Debug)]
pub struct PBIDatasets {
    benchmarks: HashMap<PBIDataset, PBIBenchmark>,
}

impl PBIDatasets {
    pub fn try_new(base_dir: PathBuf) -> VortexResult<Self> {
        let benchmark_dir = base_dir.join("benchmark");
        let benchmarks: HashMap<PBIDataset, _> = fs::read_dir(benchmark_dir)?
            .map(|path| {
                let path = path?;
                let name = path
                    .file_name()
                    .into_string()
                    .map_err(|e| vortex_err!("Not a unicode name: {e:?}"))?;
                Ok((
                    PBIDataset::from_str(name.trim(), true)
                        .map_err(|_e| vortex_err!("unsupported dataset: {} {_e}", &name))?,
                    PBIBenchmark {
                        name,
                        base_path: path.path(),
                    },
                ))
            })
            .collect::<VortexResult<HashMap<_, _>>>()?;
        Ok(Self { benchmarks })
    }

    pub fn get(&self, dataset: PBIDataset) -> &PBIBenchmark {
        self.benchmarks
            .get(&dataset)
            .ok_or_else(|| vortex_err!("{:?} not found", dataset))
            .unwrap()
    }
}

#[derive(Debug)]
pub struct PBIBenchmark {
    pub name: String,
    base_path: PathBuf,
}

pub struct Table {
    create_table_sql: String,
    name: String,
    data_url: Url,
}

impl PBIBenchmark {
    /// Parse the sql files under the queries folder and return their contents with the query idx.
    pub fn queries(&self) -> VortexResult<Vec<(usize, String)>> {
        let mut queries: Vec<_> = fs::read_dir(self.base_path.join("queries"))?
            .map(|sql_file| {
                let sql_file = sql_file?;
                let file_name = sql_file
                    .file_name()
                    .into_string()
                    .map_err(|e| vortex_err!("Not a unicode name: {e:?}"))?;
                let query_idx = file_name
                    .strip_suffix(".sql")
                    .ok_or_else(|| {
                        vortex_err!("found non-sql file under queries folder {file_name}")
                    })?
                    .parse()
                    .map_err(|_| vortex_err!("non numeric filename {file_name}"))?;
                let query = fs::read_to_string(sql_file.path())?;
                Ok((query_idx, query))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        queries.sort();
        Ok(queries)
    }

    /// Return table name and Url pairs. Each Url is pointing to a csv.bz2 file for the table.
    fn tables(&self) -> VortexResult<Vec<Table>> {
        fs::read_to_string(self.base_path.join("data-urls.txt"))?
            .lines()
            .map(|url_str| {
                let url = Url::parse(url_str)?;
                let table_name = url
                    .path_segments()
                    .and_then(|mut path| path.next_back())
                    .and_then(|filename| filename.strip_suffix(".csv.bz2"))
                    .ok_or_else(|| vortex_err!("invalid url {url}"))?;
                let create_table_sql = self.table_sql(table_name)?;
                Ok(Table {
                    create_table_sql,
                    name: table_name.to_string(),
                    data_url: url,
                })
            })
            .collect::<VortexResult<Vec<Table>>>()
            .map_err(|_| vortex_err!("invalid urls in data-urls.txt"))
    }

    fn table_sql(&self, table_name: &str) -> VortexResult<String> {
        Ok(fs::read_to_string(
            self.base_path
                .join("tables")
                .join(table_name)
                .with_extension("table.sql"),
        )?)
    }

    pub fn dataset(&self) -> VortexResult<PBIData> {
        let tables = self.tables()?;
        Ok(PBIData {
            base_path: "PBI".to_data_path().join(&self.name),
            tables,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FileType {
    CsvBzip2,
    Csv,
    Parquet,
    Vortex,
}

impl FileType {
    pub fn name(&self) -> &str {
        match self {
            FileType::CsvBzip2 => "csv_bzip2",
            FileType::Csv => "csv",
            FileType::Parquet => "parquet",
            FileType::Vortex => "vortex",
        }
    }

    pub fn extension(&self) -> &str {
        match self {
            FileType::CsvBzip2 => "csv.bz2",
            FileType::Csv => "csv",
            FileType::Parquet => "parquet",
            FileType::Vortex => "vortex",
        }
    }
}

impl Display for FileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

pub struct PBIData {
    base_path: PathBuf,
    pub tables: Vec<Table>,
}

impl PBIData {
    async fn download_bzips(&self) {
        let download_futures = self.tables.iter().map(|table| {
            download_data(
                self.get_file_path(&table.name, FileType::CsvBzip2),
                table.data_url.as_str(),
            )
        });
        join_all(download_futures).await;
    }

    fn get_file_path(&self, table_name: &str, file_type: FileType) -> PathBuf {
        self.base_path
            .join(file_type.name())
            .join(table_name)
            .with_extension(file_type.extension())
    }

    async fn unzip(&self) {
        let decompress_futures = self.tables.iter().map(|table| {
            let bzipped = self.get_file_path(&table.name, FileType::CsvBzip2);
            let unzipped = self.get_file_path(&table.name, FileType::Csv);
            tokio::task::spawn_blocking(|| {
                decompress_bz2(bzipped, unzipped);
            })
        });
        join_all(decompress_futures).await;
    }

    fn list_files(&self, file_type: FileType) -> Vec<PathBuf> {
        self.tables
            .iter()
            .map(|table| self.get_file_path(&table.name, file_type))
            .collect()
    }

    pub async fn write_as_parquet(&self) {
        self.download_bzips().await;
        self.unzip().await;

        let to_parquet_futures = self.tables.iter().map(|table| {
            let csv = self.get_file_path(&table.name, FileType::Csv);
            let parquet = self.get_file_path(&table.name, FileType::Parquet);
            async move {
                let parquet_file = idempotent_async(&parquet, async |output_path| {
                    info!("Reading schema for {}", csv.to_str().unwrap());
                    info!("Compressing {} to parquet", csv.to_str().unwrap());
                    public_bi_csv_to_parquet_file(table, csv, &output_path).await
                })
                .await
                .vortex_expect("failed to create parquet file");
                let pq_size = parquet_file.metadata().unwrap().size();
                info!(
                    "Parquet size: {}, {}B",
                    format_size(pq_size, DECIMAL),
                    pq_size
                );
            }
        });
        join_all(to_parquet_futures).await;
    }

    pub async fn write_as_vortex(&self) {
        self.write_as_parquet().await;
        let to_vortex_futures = self.tables.iter().map(|table| {
            let parquet = self.get_file_path(&table.name, FileType::Parquet);
            let vortex = self.get_file_path(&table.name, FileType::Vortex);
            async move {
                let vortex_file = idempotent_async(&vortex, async |output_path| {
                    VortexWriteOptions::default()
                        .write(
                            File::create(output_path).await.unwrap(),
                            parquet_to_vortex(parquet).unwrap(),
                        )
                        .await
                })
                .await
                .expect("failed to compress to vortex");
                let vx_size = vortex_file
                    .metadata()
                    .expect("Failed to get metadata")
                    .len();

                debug!(
                    "Vortex size: {}, {}B",
                    format_size(vx_size, DECIMAL),
                    vx_size
                );
            }
        });
        join_all(to_vortex_futures).await;
    }

    pub async fn register_tables(
        &self,
        session: &SessionContext,
        file_type: FileType,
    ) -> Result<()> {
        for table in &self.tables {
            // get schema
            let create_table = &replace_decimals(&table.create_table_sql);
            session.sql(create_table).await?;
            let table_ref = TableReference::bare(&*table.name);
            let df_table = session.table(table_ref.clone()).await?;
            let schema = replace_with_views(df_table.schema())?;

            // drop the temp table after getting the arrow schema.
            session
                .sql(&format!("DROP TABLE '{}';", &table.name))
                .await?;

            let df_format: Arc<dyn FileFormat> = match file_type {
                FileType::Csv => Arc::new(
                    CsvFormat::default()
                        .with_has_header(false)
                        .with_delimiter(b'|'),
                ),
                FileType::Parquet => Arc::new(ParquetFormat::default()),
                FileType::Vortex => Arc::new(VortexFormat::default()),
                _ => vortex_panic!("unsupported file type: {file_type}"),
            };

            let path = self.get_file_path(&table.name, file_type);
            let table_url = ListingTableUrl::parse(path.to_str().expect("unicode"))?;
            let config = ListingTableConfig::new(table_url)
                .with_listing_options(ListingOptions::new(df_format))
                .with_schema(schema.into());

            let listing_table = Arc::new(ListingTable::try_new(config)?);
            session.register_table(table_ref, listing_table)?;
        }
        Ok(())
    }
}

#[async_trait]
impl Dataset for PBIBenchmark {
    fn name(&self) -> &str {
        &self.name
    }

    // TODO(osatici): compress benchmarks use this, but this relies on all
    //                tables in a benchmark to have the same schema.
    //                That is not the case for some PBI datasets.
    async fn to_vortex_array(&self) -> ArrayRef {
        let dataset = self.dataset().expect("failed to parse tables");
        dataset.write_as_vortex().await;

        let arrays = stream::iter(dataset.list_files(FileType::Vortex))
            .map(|f| async move {
                VortexOpenOptions::file()
                    .open(f)
                    .await?
                    .scan()?
                    // TODO(ngates): why do we spawn this on the tokio executor?
                    .with_tokio_executor(Handle::current())
                    .into_array_stream()?
                    .read_all()
                    .await
            })
            .buffered(10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        ChunkedArray::from_iter(arrays).into_array()
    }
}

fn replace_with_views(schema: &DFSchema) -> Result<DFSchema> {
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| match f.data_type() {
            DataType::Binary => {
                Arc::new(<Field as Clone>::clone(f).with_data_type(DataType::BinaryView))
            }
            DataType::Utf8 => {
                Arc::new(<Field as Clone>::clone(f).with_data_type(DataType::Utf8View))
            }
            _ => f.clone(),
        })
        .collect();
    let new_schema = Schema::new(fields);
    let qualifiers = schema
        .iter()
        .map(|(qualifier, _)| qualifier.cloned())
        .collect();
    let df_schema =
        DFSchema::from_field_specific_qualified_schema(qualifiers, &Arc::new(new_schema))?;
    df_schema.with_functional_dependencies(schema.functional_dependencies().clone())
}

pub fn replace_decimals(create_table_sql: &str) -> Cow<'_, str> {
    // replace unsupported decimal types with doubles
    let decimal_regex = Regex::new(r"(?i)DECIMAL\(\s*\d+\s*(?:,\s*\d+\s*)?\)|\bDECIMAL\b").unwrap();
    decimal_regex.replace_all(create_table_sql, "DOUBLE")
}

// not using conversions::csv_to_parquet_file because duckdb does a better job at parsing csv's with the right schema
pub async fn public_bi_csv_to_parquet_file(
    table: &Table,
    csv_path: PathBuf,
    parquet_path: &Path,
) -> VortexResult<()> {
    info!("Compressing {} to parquet", csv_path.to_str().unwrap());
    let table_name = &table.name;
    let csv_path = csv_path.to_str().expect("unicode");
    let parquet_path = parquet_path.to_str().expect("unicode");

    let create_table_with_doubles = replace_decimals(&table.create_table_sql);

    let output = TokioCommand::new("duckdb")
        .arg("-c")
        .arg(format!(
            "
             {create_table_with_doubles};

             COPY {table_name} FROM '{csv_path}' (
              DELIMITER '|', 
              HEADER false, 
              NULL 'null'
             );

             COPY {table_name} TO '{parquet_path}' (FORMAT parquet, COMPRESSION zstd);
             ",
        ))
        .output()
        .await?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        vortex_bail!("duckdb convert failed: stdout=\"{stdout}\", stderr=\"{stderr}\"");
    }
    Ok(())
}
