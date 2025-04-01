use std::fmt::{self, Display};
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

use async_trait::async_trait;
use clap::ValueEnum;
use futures::future::join_all;
use futures::{StreamExt as _, TryStreamExt as _, stream};
use humansize::{DECIMAL, format_size};
use regex::Regex;
use tokio::fs::File;
use tokio::process::Command as TokioCommand;
use tokio::runtime::Handle;
use url::Url;
use vortex::aliases::hash_map::HashMap;
use vortex::arrays::ChunkedArray;
use vortex::error::{VortexExpect, VortexResult, vortex_err};
use vortex::file::{VortexOpenOptions, VortexWriteOptions};
use vortex::io::TokioFile;
use vortex::stream::ArrayStreamExt;
use vortex::{Array as _, ArrayRef};

use crate::conversions::parquet_to_vortex;
use crate::datasets::BenchmarkDataset;
use crate::datasets::data_downloads::{decompress_bz2, download_data};
use crate::{IdempotentPath, idempotent_async};

pub static PBI_DATASETS: LazyLock<PBIDatasets> = LazyLock::new(|| {
    PBIDatasets::try_new(fetch_schemas_and_queries().expect("failed to fetch public bi queries"))
        .expect("failed to construct PBI Datasets")
});

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, ValueEnum)]
#[clap(rename_all = "PascalCase")]
pub enum PBIDataset {
    AirlineSentiment,
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
    Command::new(
        base_dir
            .join("fetch_schemas_and_queries.sh")
            .to_str()
            .unwrap(),
    )
    .status()?
    .exit_ok()
    .map_err(|e| vortex_err!("Failed to fetch public bi queries: {}", e))?;
    Ok(base_dir)
}

#[derive(Debug)]
pub struct PBIDatasets {
    benchmarks: HashMap<PBIDataset, PBIBenchmark>,
    base_dir: PathBuf,
}

impl PBIDatasets {
    pub fn try_new(base_dir: PathBuf) -> VortexResult<Self> {
        let benchmark_dir = base_dir.join("benchmark");
        let benchmarks: HashMap<PBIDataset, _> = std::fs::read_dir(benchmark_dir)?
            .map(|path| {
                let path = path?;
                let name = path.file_name().into_string().expect("unicode");
                Ok((
                    PBIDataset::from_str(&name.trim(), true)
                        .map_err(|_e| vortex_err!("unsupported dataset: {} {_e}", &name))?,
                    PBIBenchmark {
                        name,
                        base_path: path.path(),
                    },
                ))
            })
            .collect::<VortexResult<HashMap<_, _>>>()?;
        Ok(Self {
            benchmarks,
            base_dir,
        })
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
    // TODO: maybe does not need a name
    name: String,
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
        let mut queries: Vec<_> = std::fs::read_dir(self.base_path.join("queries"))?
            .map(|sql_file| {
                let sql_file = sql_file?;
                let file_name = sql_file.file_name().into_string().expect("unicode");
                let query_idx = file_name
                    .strip_suffix(".sql")
                    .ok_or_else(|| {
                        vortex_err!("found non-sql file under queries folder {file_name}")
                    })?
                    .parse()
                    .map_err(|_| vortex_err!("non numeric filename {file_name}"))?;
                let query = std::fs::read_to_string(sql_file.path())?;
                Ok((query_idx, query))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        queries.sort();
        Ok(queries)
    }

    /// Return table name and Url pairs. Each Url is pointing to a csv.bz2 file for the table.
    fn tables(&self) -> VortexResult<Vec<Table>> {
        std::fs::read_to_string(self.base_path.join("data-urls.txt"))?
            .lines()
            .map(|url_str| {
                let url = Url::parse(url_str)?;
                let table_name = url
                    .path_segments()
                    .and_then(|path| path.last())
                    .and_then(|filename| filename.strip_suffix(".csv.bz2"))
                    .ok_or_else(|| vortex_err!("invalid url {url}"))?;
                let create_table_sql = self.table_sql(&table_name)?;
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
        Ok(std::fs::read_to_string(
            self.base_path
                .join("tables")
                .join(table_name)
                .with_extension("table.sql"),
        )?)
    }

    pub fn dataset(&self) -> VortexResult<PBIData> {
        let tables = self.tables()?;
        Ok(PBIData {
            name: self.name.clone(),
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
    name: String,
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
                    tracing::info!("Reading schema for {}", csv.to_str().unwrap());
                    tracing::info!("Compressing {} to parquet", csv.to_str().unwrap());
                    public_bi_csv_to_parquet_file(&table, csv, &output_path).await
                })
                .await
                .vortex_expect("failed to create parquet file");
                let pq_size = parquet_file.metadata().unwrap().size();
                tracing::info!(
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
                    .len() as usize;

                tracing::debug!(
                    "Vortex size: {}, {}B",
                    format_size(vx_size as u64, DECIMAL),
                    vx_size
                );
            }
        });
        join_all(to_vortex_futures).await;
    }
}

#[async_trait]
impl BenchmarkDataset for PBIBenchmark {
    fn name(&self) -> &str {
        &self.name
    }

    async fn to_vortex_array(&self) -> ArrayRef {
        let dataset = self.dataset().expect("failed to parse tables");
        dataset.write_as_vortex().await;

        let arrays = stream::iter(dataset.list_files(FileType::Vortex))
            .map(|f| async move {
                VortexOpenOptions::file()
                    .open(TokioFile::open(f)?)
                    .await?
                    .scan()?
                    .spawn_tokio(Handle::current())
                    .unwrap()
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

// not using conversions::csv_to_parquet_file because duckdb does a better job at parsing csv's with the right schema
pub async fn public_bi_csv_to_parquet_file(
    table: &Table,
    csv_path: PathBuf,
    parquet_path: &Path,
) -> VortexResult<()> {
    tracing::info!("Compressing {} to parquet", csv_path.to_str().unwrap());
    let table_name = &table.name;
    let csv_path = csv_path.to_str().expect("unicode");
    let parquet_path = parquet_path.to_str().expect("unicode");

    // replace unsupported decimal types with doubles
    let decimal_regex = Regex::new(r"(?i)DECIMAL\(\s*\d+\s*(?:,\s*\d+\s*)?\)|\bDECIMAL\b").unwrap();
    let create_table_with_doubles = decimal_regex.replace_all(&table.create_table_sql, "DOUBLE");

    TokioCommand::new("duckdb")
        .arg("-c")
        .arg(format!(
            "
             {create_table_with_doubles};

             COPY {table_name} FROM '{csv_path}' (
              DELIMITER '|', 
              HEADER false, 
              NULL 'null'
             );

             COPY {table_name} TO '{parquet_path}' (COMPRESSION ZSTD);
             ",
        ))
        .status()
        .await?
        .exit_ok()
        .map_err(|e| vortex_err!("Failed to convert csv to parquet: {}", e))
}
