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
use vortex::array::IntoArray;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::SESSION;
use crate::SetupCtx;
use crate::TableSpec;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;
use crate::datasets::data_downloads::decompress_bz2;
use crate::datasets::data_downloads::download_many;
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

/// Every Public BI dataset and its tables, vendored from the upstream
/// `benchmark/<dataset>/data-urls.txt` files.
///
/// Upstream stores one URL per table, but all 206 of them are
/// `{DATA_URL_PREFIX}/{dataset}/{table}.csv.bz2`, so only the table names are
/// recorded here and [`data_urls`] rebuilds the URL.
///
/// Derived from the Public BI benchmark (<https://github.com/cwida/public_bi_benchmark>),
/// MIT licensed, Copyright (c) 2019 CWI Database Architectures Group. The underlying table
/// dumps are anonymized Tableau Public workbooks and carry their own provenance.
const DATASET_TABLES: &[(&str, &[&str])] = &[
    ("Arade", &["Arade_1"]),
    ("Bimbo", &["Bimbo_1"]),
    ("CMSprovider", &["CMSprovider_1", "CMSprovider_2"]),
    ("CityMaxCapita", &["CityMaxCapita_1"]),
    (
        "CommonGovernment",
        &[
            "CommonGovernment_10",
            "CommonGovernment_11",
            "CommonGovernment_12",
            "CommonGovernment_13",
            "CommonGovernment_1",
            "CommonGovernment_2",
            "CommonGovernment_3",
            "CommonGovernment_4",
            "CommonGovernment_5",
            "CommonGovernment_6",
            "CommonGovernment_7",
            "CommonGovernment_8",
            "CommonGovernment_9",
        ],
    ),
    ("Corporations", &["Corporations_1"]),
    ("Eixo", &["Eixo_1"]),
    ("Euro2016", &["Euro2016_1"]),
    ("Food", &["Food_1"]),
    (
        "Generico",
        &[
            "Generico_1",
            "Generico_2",
            "Generico_3",
            "Generico_4",
            "Generico_5",
        ],
    ),
    ("HashTags", &["HashTags_1"]),
    ("Hatred", &["Hatred_1"]),
    ("IGlocations1", &["IGlocations1_1"]),
    ("IGlocations2", &["IGlocations2_1", "IGlocations2_2"]),
    ("IUBLibrary", &["IUBLibrary_1"]),
    (
        "MLB",
        &[
            "MLB_10", "MLB_11", "MLB_12", "MLB_13", "MLB_14", "MLB_15", "MLB_16", "MLB_17",
            "MLB_18", "MLB_19", "MLB_1", "MLB_20", "MLB_21", "MLB_22", "MLB_23", "MLB_24",
            "MLB_25", "MLB_26", "MLB_27", "MLB_28", "MLB_29", "MLB_2", "MLB_30", "MLB_31",
            "MLB_32", "MLB_33", "MLB_34", "MLB_35", "MLB_36", "MLB_37", "MLB_38", "MLB_39",
            "MLB_3", "MLB_40", "MLB_41", "MLB_42", "MLB_43", "MLB_44", "MLB_45", "MLB_46",
            "MLB_47", "MLB_48", "MLB_49", "MLB_4", "MLB_50", "MLB_51", "MLB_52", "MLB_53",
            "MLB_54", "MLB_55", "MLB_56", "MLB_57", "MLB_58", "MLB_59", "MLB_5", "MLB_60",
            "MLB_61", "MLB_62", "MLB_63", "MLB_64", "MLB_65", "MLB_66", "MLB_67", "MLB_68",
            "MLB_6", "MLB_7", "MLB_8", "MLB_9",
        ],
    ),
    ("MedPayment1", &["MedPayment1_1"]),
    ("MedPayment2", &["MedPayment2_1"]),
    ("Medicare1", &["Medicare1_1", "Medicare1_2"]),
    ("Medicare2", &["Medicare2_1", "Medicare2_2"]),
    ("Medicare3", &["Medicare3_1"]),
    ("Motos", &["Motos_1", "Motos_2"]),
    ("MulheresMil", &["MulheresMil_1"]),
    ("NYC", &["NYC_1", "NYC_2"]),
    ("PanCreactomy1", &["PanCreactomy1_1"]),
    ("PanCreactomy2", &["PanCreactomy2_1", "PanCreactomy2_2"]),
    ("Physicians", &["Physicians_1"]),
    (
        "Provider",
        &[
            "Provider_1",
            "Provider_2",
            "Provider_3",
            "Provider_4",
            "Provider_5",
            "Provider_6",
            "Provider_7",
            "Provider_8",
        ],
    ),
    ("RealEstate1", &["RealEstate1_1", "RealEstate1_2"]),
    (
        "RealEstate2",
        &[
            "RealEstate2_1",
            "RealEstate2_2",
            "RealEstate2_3",
            "RealEstate2_4",
            "RealEstate2_5",
            "RealEstate2_6",
            "RealEstate2_7",
        ],
    ),
    (
        "Redfin1",
        &["Redfin1_1", "Redfin1_2", "Redfin1_3", "Redfin1_4"],
    ),
    ("Redfin2", &["Redfin2_1", "Redfin2_2", "Redfin2_3"]),
    ("Redfin3", &["Redfin3_1", "Redfin3_2"]),
    ("Redfin4", &["Redfin4_1"]),
    (
        "Rentabilidad",
        &[
            "Rentabilidad_1",
            "Rentabilidad_2",
            "Rentabilidad_3",
            "Rentabilidad_4",
            "Rentabilidad_5",
            "Rentabilidad_6",
            "Rentabilidad_7",
            "Rentabilidad_8",
            "Rentabilidad_9",
        ],
    ),
    ("Romance", &["Romance_1", "Romance_2"]),
    (
        "SalariesFrance",
        &[
            "SalariesFrance_10",
            "SalariesFrance_11",
            "SalariesFrance_12",
            "SalariesFrance_13",
            "SalariesFrance_1",
            "SalariesFrance_2",
            "SalariesFrance_3",
            "SalariesFrance_4",
            "SalariesFrance_5",
            "SalariesFrance_6",
            "SalariesFrance_7",
            "SalariesFrance_8",
            "SalariesFrance_9",
        ],
    ),
    (
        "TableroSistemaPenal",
        &[
            "TableroSistemaPenal_1",
            "TableroSistemaPenal_2",
            "TableroSistemaPenal_3",
            "TableroSistemaPenal_4",
            "TableroSistemaPenal_5",
            "TableroSistemaPenal_6",
            "TableroSistemaPenal_7",
            "TableroSistemaPenal_8",
        ],
    ),
    (
        "Taxpayer",
        &[
            "Taxpayer_10",
            "Taxpayer_1",
            "Taxpayer_2",
            "Taxpayer_3",
            "Taxpayer_4",
            "Taxpayer_5",
            "Taxpayer_6",
            "Taxpayer_7",
            "Taxpayer_8",
            "Taxpayer_9",
        ],
    ),
    ("Telco", &["Telco_1"]),
    (
        "TrainsUK1",
        &["TrainsUK1_1", "TrainsUK1_2", "TrainsUK1_3", "TrainsUK1_4"],
    ),
    ("TrainsUK2", &["TrainsUK2_1", "TrainsUK2_2"]),
    ("USCensus", &["USCensus_1", "USCensus_2", "USCensus_3"]),
    ("Uberlandia", &["Uberlandia_1"]),
    ("Wins", &["Wins_1", "Wins_2", "Wins_3", "Wins_4"]),
    (
        "YaleLanguages",
        &[
            "YaleLanguages_1",
            "YaleLanguages_2",
            "YaleLanguages_3",
            "YaleLanguages_4",
            "YaleLanguages_5",
        ],
    ),
];

/// Bucket holding every Public BI table dump.
const DATA_URL_PREFIX: &str = "https://pub-334c2a12c9bf46f3b8464a8718df8cae.r2.dev";

/// `(table name, source URL)` for every table in `dataset`.
///
/// Vendored rather than parsed from the upstream `data-urls.txt` so the download list is
/// known at compile time, and a fetch of the upstream repo cannot change what we download.
fn data_urls(dataset: &str) -> anyhow::Result<Vec<(String, Url)>> {
    let tables = DATASET_TABLES
        .iter()
        .find(|(name, _)| *name == dataset)
        .map(|(_, tables)| *tables)
        .ok_or_else(|| anyhow!("unknown Public BI dataset {dataset}"))?;

    tables
        .iter()
        .map(|table| {
            let url = Url::parse(&format!("{DATA_URL_PREFIX}/{dataset}/{table}.csv.bz2"))?;
            Ok(((*table).to_owned(), url))
        })
        .collect()
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

    /// Table name and source URL pairs, from the vendored [`DATASET_TABLES`] listing.
    fn tables(&self) -> anyhow::Result<Vec<Table>> {
        data_urls(&self.name)?
            .into_iter()
            .map(|(name, data_url)| {
                Ok(Table {
                    create_table_sql: self.table_sql(&name)?,
                    name,
                    data_url,
                })
            })
            .collect()
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
        let downloads = self.tables.iter().map(|table| {
            (
                self.get_file_path(&table.name, FileType::CsvBzip2),
                table.data_url.as_str().to_owned(),
            )
        });
        download_many(downloads).await?;
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

    fn v3_dataset_dims(&self) -> (&str, Option<&str>) {
        // Match the v2 → v3 migrate classifier, which emits PBI compression
        // records as `dataset = <lowercased pbi name>, dataset_variant = NULL`.
        // The case-folding is applied by `compression_time_record` /
        // `compression_size_record`; this method just surfaces the raw PBI
        // name as the dataset.
        (&self.name, None)
    }

    async fn download(&self) -> anyhow::Result<()> {
        self.dataset()?.download_bzips().await
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
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/public-bi.md"
    }

    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        self.pbi_benchmark().queries()
    }

    async fn setup(&self, ctx: &SetupCtx, _format: Format) -> anyhow::Result<()> {
        let pbi_data = self.pbi_benchmark().dataset()?;
        pbi_data.write_as_parquet().await?;
        for (table, path) in pbi_data
            .tables
            .iter()
            .zip(pbi_data.list_files(FileType::Parquet))
        {
            ctx.emit(&table.name, path);
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pbi_v3_dataset_dims_uses_pbi_name_as_dataset_with_no_variant() {
        // The v2 → v3 migrate classifier emits PBI compression records as
        // `dataset = <lowercased pbi name>, dataset_variant = NULL` (it never
        // carried a `public-bi` parent in v2 chart names). The live emitter
        // must mirror that shape so live ingests merge with migrated history
        // into a single per-PBI-dataset chart group instead of forking off a
        // sibling group keyed on `public-bi/<name>`. Lowercasing happens in
        // `compression_time_record`/`compression_size_record`, so this trait
        // method just needs to surface the raw PBI name as the dataset.
        let bench = PBIBenchmark {
            name: "Arade".to_string(),
            base_path: PathBuf::new(),
        };
        assert_eq!(bench.v3_dataset_dims(), ("Arade", None));

        let bench = PBIBenchmark {
            name: "CMSprovider".to_string(),
            base_path: PathBuf::new(),
        };
        assert_eq!(bench.v3_dataset_dims(), ("CMSprovider", None));
    }

    /// Every `PBIDataset` variant must appear in the vendored listing, or `data_urls` fails
    /// at runtime for that dataset.
    #[test]
    fn every_dataset_variant_has_vendored_tables() -> anyhow::Result<()> {
        for dataset in PBIDataset::value_variants() {
            let name = format!("{dataset:?}");
            let urls =
                data_urls(&name).with_context(|| format!("{name} missing from DATASET_TABLES"))?;
            assert!(!urls.is_empty(), "{name} has no tables");
        }
        Ok(())
    }

    #[test]
    fn data_urls_rebuild_the_upstream_layout() -> anyhow::Result<()> {
        let urls = data_urls("Arade")?;
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].0, "Arade_1");
        assert_eq!(
            urls[0].1.as_str(),
            "https://pub-334c2a12c9bf46f3b8464a8718df8cae.r2.dev/Arade/Arade_1.csv.bz2",
        );
        Ok(())
    }

    #[test]
    fn unknown_dataset_is_rejected() {
        assert!(data_urls("NotADataset").is_err());
    }

    /// 206 tables across 46 datasets, matching the upstream `data-urls.txt` corpus this was
    /// vendored from. A drift here means upstream added or removed a table dump.
    #[test]
    fn vendored_corpus_size_is_pinned() {
        assert_eq!(DATASET_TABLES.len(), 46);
        let tables: usize = DATASET_TABLES.iter().map(|(_, t)| t.len()).sum();
        assert_eq!(tables, 206);
    }
}
