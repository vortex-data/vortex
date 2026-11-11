// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The three largest TPC-H tables at scale factor 1, each as a whole-table compression dataset.
//!
//! `lineitem`, `orders`, and `partsupp` hold almost all of the TPC-H data. The other tables are
//! left out of the compression suite.

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;

use crate::Format;
use crate::IdempotentPath;
use crate::conversions::parquet_to_vortex_chunks;
use crate::datasets::Dataset;
use crate::tpch::tpchgen::TpchGenOptions;
use crate::tpch::tpchgen::generate_tpch_tables;
use crate::utils::split;

/// One TPC-H table at scale factor 1, read in full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpchSf1Table {
    LineItem,
    Orders,
    PartSupp,
}

impl TpchSf1Table {
    /// Every TPC-H table this suite measures.
    pub const ALL: [Self; 3] = [Self::LineItem, Self::Orders, Self::PartSupp];

    /// The TPC-H table name, as the generator writes it.
    pub fn table_name(self) -> &'static str {
        match self {
            Self::LineItem => "lineitem",
            Self::Orders => "orders",
            Self::PartSupp => "partsupp",
        }
    }

    /// Path of this table in `format`, generating the TPC-H SF=1 tables if it is missing.
    ///
    /// At SF=1 every table fits in a single file (see [`split::TARGET_FILE_COMPRESSED_BYTES`]), so
    /// this reads the TPC-H benchmark's own data.
    pub async fn path(self, format: Format) -> Result<PathBuf> {
        let table = self.table_name();
        let scale_factor_dir = "tpch".to_data_path().join("1.0");
        let format_dir = scale_factor_dir.join(format.name());
        let path = format_dir.join(split::part_file_name(table, 1, format.ext()));
        if !path.exists() {
            let options =
                TpchGenOptions::new("1.0".to_string(), scale_factor_dir).with_format(format);
            generate_tpch_tables(options).await?;
        }
        anyhow::ensure!(path.exists(), "No {table} {format} file found");
        anyhow::ensure!(
            !format_dir
                .join(split::part_file_name(table, 2, format.ext()))
                .exists(),
            "TPC-H SF=1 {table} is split into several {format} files"
        );
        Ok(path)
    }
}

#[async_trait]
impl Dataset for TpchSf1Table {
    fn name(&self) -> &str {
        match self {
            Self::LineItem => "TPC-H SF1 lineitem",
            Self::Orders => "TPC-H SF1 orders",
            Self::PartSupp => "TPC-H SF1 partsupp",
        }
    }

    async fn to_vortex_array(&self, _ctx: &mut ExecutionCtx) -> Result<ArrayRef> {
        let parquet = self.to_parquet_path().await?;
        Ok(parquet_to_vortex_chunks(parquet).await?.into_array())
    }

    async fn to_parquet_path(&self) -> Result<PathBuf> {
        self.path(Format::Parquet).await
    }
}
