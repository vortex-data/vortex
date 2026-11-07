// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Nested TPC-H: TPC-H data and queries in the nested shape of Apache DataFusion Comet's
//! [`benchmarks/nested`][comet].
//!
//! Each table keeps its join keys and one partition column at the top level and folds every
//! other column into a one-element `<table>_data` list of structs (see [`crate::tpch::nest`]);
//! each query unnests the lists it reads. The lists hold exactly one element, so the queries
//! return the flat TPC-H answers and share the flat suite's expected row counts.
//!
//! Data comes from the flat suite's generator with [`TableShape::ListOfStruct`], so it has the
//! flat suite's types, row groups and file splits rather than Comet's Spark output. Queries live
//! in `vortex-bench/sql/tpch-nested/<shape>/`.
//!
//! [comet]: https://github.com/apache/datafusion-comet/tree/main/benchmarks/nested

use std::fmt::Display;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use arrow_schema::Schema;
use clap::ValueEnum;
use glob::Pattern;
use serde::Deserialize;
use serde::Serialize;
use tracing::warn;
use url::Url;
use vortex::error::VortexExpect;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::TableSpec;
use crate::tpch::EXPECTED_ROW_COUNTS_SF1;
use crate::tpch::EXPECTED_ROW_COUNTS_SF10;
use crate::tpch::nest::nested_schema;
use crate::tpch::schema::CUSTOMER;
use crate::tpch::schema::LINEITEM;
use crate::tpch::schema::NATION;
use crate::tpch::schema::ORDERS;
use crate::tpch::schema::PART;
use crate::tpch::schema::PARTSUPP;
use crate::tpch::schema::REGION;
use crate::tpch::schema::SUPPLIER;
use crate::tpch::tpchgen;
use crate::tpch::tpchgen::TableShape;
use crate::tpch::tpchgen::TpchGenOptions;
use crate::utils::split;

/// How each table's non-key columns are nested.
#[derive(ValueEnum, Default, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Shape {
    /// A one-element list of structs, Comet's layout.
    #[default]
    List,
}

impl Shape {
    pub fn name(self) -> &'static str {
        match self {
            Shape::List => "list",
        }
    }

    fn table_shape(self) -> TableShape {
        match self {
            Shape::List => TableShape::ListOfStruct,
        }
    }
}

impl FromStr for Shape {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "list" => Ok(Shape::List),
            _ => anyhow::bail!("Failed to parse {s}, the only valid shape value is `list`"),
        }
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// The flat schemas the nested tables derive from.
const TABLES: [(&str, &std::sync::LazyLock<Schema>); 8] = [
    ("customer", &CUSTOMER),
    ("lineitem", &LINEITEM),
    ("nation", &NATION),
    ("orders", &ORDERS),
    ("part", &PART),
    ("partsupp", &PARTSUPP),
    ("region", &REGION),
    ("supplier", &SUPPLIER),
];

/// The 22 nested queries of `shape`, from `sql/tpch-nested/<shape>/q{n}.sql`.
pub fn tpch_nested_queries(shape: Shape) -> impl Iterator<Item = (usize, String)> {
    (1..=22).map(move |query| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("sql")
            .join("tpch-nested")
            .join(shape.name())
            .join(format!("q{query}"))
            .with_extension("sql");
        let sql =
            fs::read_to_string(path).vortex_expect("cannot load nested TPC-H query from file");
        (query, sql)
    })
}

/// Nested TPC-H benchmark implementation.
pub struct TpcHNestedBenchmark {
    pub scale_factor: String,
    pub shape: Shape,
    data_url: Url,
}

impl TpcHNestedBenchmark {
    pub fn new(
        scale_factor: String,
        shape: Shape,
        remote_data_dir: Option<String>,
    ) -> anyhow::Result<Self> {
        let data_url = match remote_data_dir {
            None => {
                let data_dir = "tpch-nested"
                    .to_data_path()
                    .join(&scale_factor)
                    .join(shape.name());
                Url::from_directory_path(&data_dir).map_err(|_| {
                    anyhow::anyhow!("Failed to create URL from directory path: {data_dir:?}")
                })?
            }
            Some(remote_data_dir) => {
                if !remote_data_dir.ends_with('/') {
                    warn!(
                        "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/tpch-nested/{scale_factor}/{shape}/"
                    );
                }
                Url::parse(&remote_data_dir)?
            }
        };
        Ok(Self {
            scale_factor,
            shape,
            data_url,
        })
    }
}

#[async_trait::async_trait]
impl Benchmark for TpcHNestedBenchmark {
    fn doc_path(&self) -> &'static str {
        "vortex-bench/sql/tpch-nested/README.md"
    }

    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        Ok(tpch_nested_queries(self.shape).collect())
    }

    async fn generate_base_data(&self) -> anyhow::Result<()> {
        if self.data_url.scheme() != "file" {
            return Ok(());
        }

        let base_data_dir = self
            .data_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", self.data_url.as_str()))?;

        let options = TpchGenOptions::new(self.scale_factor.clone(), base_data_dir)
            .with_shape(self.shape.table_shape());

        tpchgen::generate_tpch_tables(options).await?;

        Ok(())
    }

    /// The flat suite's row counts: every list holds one element, so unnesting changes nothing.
    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        match self.scale_factor.as_str() {
            "1.0" => Some(EXPECTED_ROW_COUNTS_SF1.to_vec()),
            "10.0" => Some(EXPECTED_ROW_COUNTS_SF10.to_vec()),
            _ => None,
        }
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::TpcHNested {
            scale_factor: self.scale_factor.clone(),
            shape: self.shape,
        }
    }

    fn dataset_name(&self) -> &str {
        "tpch-nested"
    }

    fn dataset_display(&self) -> String {
        format!(
            "tpch-nested(sf={}, shape={})",
            self.scale_factor, self.shape
        )
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        TABLES
            .iter()
            .map(|(name, flat)| TableSpec::new(name, Some(nested_schema(name, flat))))
            .collect()
    }

    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        Some(split::part_pattern(table_name, format.ext()))
    }
}
