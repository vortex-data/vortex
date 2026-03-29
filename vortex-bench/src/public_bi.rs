// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt::Display;
use std::fmt::{self};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::LazyLock;

use anyhow::Context;
use anyhow::anyhow;
use anyhow::bail;
use async_trait::async_trait;
use clap::ValueEnum;
use futures::future::join_all;
use futures::future::try_join_all;
use humansize::DECIMAL;
use humansize::format_size;
use regex::Regex;
use tokio::fs::File;
use tokio::process::Command as TokioCommand;
use tracing::info;
use tracing::trace;
use url::Url;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::SESSION;
use crate::TableSpec;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;
use crate::datasets::data_downloads::decompress_bz2;
use crate::datasets::data_downloads::download_data;
use crate::idempotent_async;
use crate::workspace_root;

pub static PBI_DATASETS: LazyLock<PBIDatasets> = LazyLock::new(|| {
    PBIDatasets::try_new(fetch_schemas_and_queries().expect("failed to fetch public bi queries"))
        .expect("failed to construct PBI Datasets")
});

use std::str::FromStr;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, ValueEnum)]
#[clap(rename_all = "LowerCase")]
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

impl FromStr for PBIDataset {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Use clap's ValueEnum parsing
        <Self as ValueEnum>::from_str(s, true)
            .map_err(|e| anyhow!("invalid PBI dataset '{}': {}", s, e))
    }
}

pub fn fetch_schemas_and_queries() -> anyhow::Result<PathBuf> {
    let scripts_dir = workspace_root().join("vortex-bench").join("scripts");
    let output = Command::new(
        scripts_dir
            .join("fetch_public_bi_schemas_and_queries.sh")
            .to_str()
            .unwrap(),
    )
    .output()?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("public_bi fetch failed: stdout=\"{stdout}\", stderr=\"{stderr}\"");
    }

    // Return the public_bi directory where the git repo is initialized.
    Ok(Path::new(env!("CARGO_MANIFEST_DIR")).join("public_bi"))
}

#[derive(Debug)]
pub struct PBIDatasets {
    benchmarks: HashMap<PBIDataset, PBIBenchmark>,
}

impl PBIDatasets {
    pub fn try_new(base_dir: PathBuf) -> anyhow::Result<Self> {
        let benchmark_dir = base_dir.join("benchmark");
        let benchmarks: HashMap<PBIDataset, _> = fs::read_dir(benchmark_dir)?
            .map(|path| {
                let path = path?;
                let name = path
                    .file_name()
                    .into_string()
                    .map_err(|e| vortex_err!("Not a unicode name: {e:?}"))?;
                Ok((
                    <PBIDataset as ValueEnum>::from_str(name.trim(), true)
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
    pub fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
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
    fn tables(&self) -> anyhow::Result<Vec<Table>> {
        fs::read_to_string(self.base_path.join("data-urls.txt"))?
            .lines()
            .map(|url_str| {
                let url = Url::parse(url_str)?;
                let table_name = url
                    .path_segments()
                    .and_then(|mut path| path.next_back())
                    .and_then(|filename| filename.strip_suffix(".csv.bz2"))
                    .ok_or_else(|| anyhow!("invalid url {url}"))?;
                let create_table_sql = self.table_sql(table_name)?;
                Ok(Table {
                    create_table_sql,
                    name: table_name.to_string(),
                    data_url: url,
                })
            })
            .collect::<anyhow::Result<Vec<Table>>>()
            .map_err(|_| anyhow!("invalid urls in data-urls.txt"))
    }

    fn table_sql(&self, table_name: &str) -> anyhow::Result<String> {
        Ok(fs::read_to_string(
            self.base_path
                .join("tables")
                .join(table_name)
                .with_extension("table.sql"),
        )?)
    }

    pub fn dataset(&self) -> anyhow::Result<PBIData> {
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
    async fn download_bzips(&self) -> anyhow::Result<()> {
        let download_futures = self.tables.iter().map(|table| {
            download_data(
                self.get_file_path(&table.name, FileType::CsvBzip2),
                table.data_url.as_str(),
            )
        });
        let results = join_all(download_futures).await;
        for result in results {
            result?;
        }
        Ok(())
    }

    fn get_file_path(&self, table_name: &str, file_type: FileType) -> PathBuf {
        self.base_path
            .join(file_type.name())
            .join(table_name)
            .with_extension(file_type.extension())
    }

    async fn unzip(&self) -> anyhow::Result<()> {
        let decompress_futures = self.tables.iter().map(|table| {
            let bzipped = self.get_file_path(&table.name, FileType::CsvBzip2);
            let unzipped = self.get_file_path(&table.name, FileType::Csv);
            tokio::task::spawn_blocking(move || decompress_bz2(bzipped, unzipped))
        });
        let results = join_all(decompress_futures).await;
        for result in results {
            result.map_err(|e| anyhow::anyhow!("Failed to spawn decompression task: {}", e))??;
        }
        Ok(())
    }

    fn list_files(&self, file_type: FileType) -> Vec<PathBuf> {
        self.tables
            .iter()
            .map(|table| self.get_file_path(&table.name, file_type))
            .collect()
    }

    pub async fn write_as_parquet(&self) -> anyhow::Result<()> {
        self.download_bzips().await?;
        self.unzip().await?;

        let to_parquet_futures = self.tables.iter().map(|table| {
            let csv = self.get_file_path(&table.name, FileType::Csv);
            let parquet = self.get_file_path(&table.name, FileType::Parquet);
            async move {
                let parquet_file = idempotent_async(&parquet, async |output_path| {
                    info!("Reading schema for {}", csv.to_str().unwrap());
                    info!("Compressing {} to parquet", csv.to_str().unwrap());
                    public_bi_csv_to_parquet_file(table, csv, &output_path).await
                })
                .await?;
                let pq_size = parquet_file.metadata().unwrap().len();
                info!(
                    "Parquet size: {}, {}B",
                    format_size(pq_size, DECIMAL),
                    pq_size
                );
                Ok::<_, anyhow::Error>(())
            }
        });
        try_join_all(to_parquet_futures).await?;
        Ok(())
    }

    pub async fn write_as_vortex(&self) -> anyhow::Result<()> {
        self.write_as_parquet().await?;
        let to_vortex_futures = self.tables.iter().map(|table| {
            let parquet = self.get_file_path(&table.name, FileType::Parquet);
            let vortex = self.get_file_path(&table.name, FileType::Vortex);

            async move {
                let data = parquet_to_vortex_chunks(parquet).await?;
                let vortex_file =
                    idempotent_async(&vortex, async |output_path| -> anyhow::Result<()> {
                        SESSION
                            .write_options()
                            .write(
                                &mut File::create(output_path)
                                    .await
                                    .map_err(|e| anyhow::anyhow!("Failed to create file: {}", e))?,
                                data.into_array().to_array_stream(),
                            )
                            .await
                            .map_err(|e| anyhow::anyhow!("Failed to write vortex file: {}", e))?;
                        Ok(())
                    })
                    .await?;
                let vx_size = vortex_file.metadata()?.len();

                trace!(
                    "Vortex size: {}, {}B",
                    format_size(vx_size, DECIMAL),
                    vx_size
                );

                Ok::<_, anyhow::Error>(())
            }
        });
        try_join_all(to_vortex_futures).await?;
        Ok(())
    }
}

fn replace_decimals(create_table_sql: &str) -> Cow<'_, str> {
    // replace unsupported decimal types with doubles
    let decimal_regex = Regex::new(r"(?i)DECIMAL\(\s*\d+\s*(?:,\s*\d+\s*)?\)|\bDECIMAL\b").unwrap();
    decimal_regex.replace_all(create_table_sql, "DOUBLE")
}

// not using conversions::csv_to_parquet_file because duckdb does a better job at parsing csv's with the right schema
pub async fn public_bi_csv_to_parquet_file(
    table: &Table,
    csv_path: PathBuf,
    parquet_path: &Path,
) -> anyhow::Result<()> {
    info!(
        "Compressing {} to parquet",
        csv_path
            .to_str()
            .context("Failed to convert CSV path to string")?
    );
    let table_name = &table.name;
    let csv_path = csv_path
        .to_str()
        .context("Failed to convert CSV path to unicode string")?;
    let parquet_path = parquet_path
        .to_str()
        .context("Failed to convert Parquet path to unicode string")?;

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
        bail!("duckdb convert failed: stdout=\"{stdout}\", stderr=\"{stderr}\"");
    }
    Ok(())
}

#[async_trait]
impl Dataset for PBIBenchmark {
    fn name(&self) -> &str {
        &self.name
    }

    async fn to_vortex_array(&self) -> anyhow::Result<ArrayRef> {
        let dataset = self.dataset()?;
        dataset.write_as_vortex().await?;
        // reading only the first table, each table in a PBI benchmark
        // has its own schema.
        let path = dataset
            .list_files(FileType::Vortex)
            .first()
            .ok_or_else(|| anyhow!("must have at least one table"))?
            .clone();

        Ok(SESSION
            .open_options()
            .open_path(path.as_path())
            .await?
            .scan()?
            .into_array_stream()?
            .read_all()
            .await?)
    }

    async fn to_parquet_path(&self) -> anyhow::Result<PathBuf> {
        let dataset = self.dataset()?;
        dataset.write_as_parquet().await?;
        dataset
            .list_files(FileType::Parquet)
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("must have at least one parquet file"))
    }
}

/// Public BI benchmark implementation that conforms to the `Benchmark` trait.
pub struct PublicBiBenchmark {
    pub dataset: PBIDataset,
    pub data_url: Url,
    /// Cached table names from the dataset
    table_names: Vec<String>,
}

impl PublicBiBenchmark {
    pub fn new(dataset: PBIDataset) -> anyhow::Result<Self> {
        let pbi_benchmark = PBI_DATASETS.get(dataset);
        let pbi_data = pbi_benchmark.dataset()?;
        let table_names: Vec<String> = pbi_data.tables.iter().map(|t| t.name.clone()).collect();

        let data_url = Url::parse(&format!(
            "file:{}/",
            pbi_data
                .base_path
                .to_str()
                .ok_or_else(|| anyhow!("path not utf8"))?
        ))?;

        Ok(Self {
            dataset,
            data_url,
            table_names,
        })
    }

    fn pbi_benchmark(&self) -> &PBIBenchmark {
        PBI_DATASETS.get(self.dataset)
    }
}

#[async_trait]
impl Benchmark for PublicBiBenchmark {
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        self.pbi_benchmark().queries()
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        let pbi_data = self.pbi_benchmark().dataset()?;
        pbi_data.write_as_parquet().await
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::PublicBi {
            name: self.pbi_benchmark().name.clone(),
        }
    }

    fn dataset_name(&self) -> &str {
        "public-bi"
    }

    fn dataset_display(&self) -> String {
        format!("public-bi({})", self.pbi_benchmark().name)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        // Public BI datasets have dynamic schemas parsed from SQL files at runtime,
        // so we return table specs without static Arrow schemas.
        // The schema will be inferred from the data files.
        self.table_names
            .iter()
            .map(|name| {
                // Leak the string to get a &'static str - this is fine since benchmarks
                // are long-lived and we only create a small number of them.
                let static_name: &'static str = Box::leak(name.clone().into_boxed_str());
                TableSpec::new(static_name, None)
            })
            .collect()
    }

    fn pattern(&self, table_name: &str, format: Format) -> Option<glob::Pattern> {
        // Each table is a single file named {table_name}.{ext}
        let pattern_str = format!("{}.{}", table_name, format.ext());
        glob::Pattern::new(&pattern_str).ok()
    }
}
