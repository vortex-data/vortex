// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-DS data generation using the pure-Rust `tpcdsgen` crate.

use std::fs;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_cast::CastOptions;
use arrow_cast::cast_with_options;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use parquet::arrow::ArrowWriter;
use tpcdsgen::config::CompatMode;
use tpcdsgen::config::Session;
use tpcdsgen::config::SessionBuilder;
use tpcdsgen_arrow::CallCenterArrow;
use tpcdsgen_arrow::CatalogPageArrow;
use tpcdsgen_arrow::CatalogReturnsArrow;
use tpcdsgen_arrow::CatalogSalesArrow;
use tpcdsgen_arrow::CustomerAddressArrow;
use tpcdsgen_arrow::CustomerArrow;
use tpcdsgen_arrow::CustomerDemographicsArrow;
use tpcdsgen_arrow::DateDimArrow;
use tpcdsgen_arrow::HouseholdDemographicsArrow;
use tpcdsgen_arrow::IncomeBandArrow;
use tpcdsgen_arrow::InventoryArrow;
use tpcdsgen_arrow::ItemArrow;
use tpcdsgen_arrow::PromotionArrow;
use tpcdsgen_arrow::ReasonArrow;
use tpcdsgen_arrow::RecordBatchIterator;
use tpcdsgen_arrow::ShipModeArrow;
use tpcdsgen_arrow::StoreArrow;
use tpcdsgen_arrow::StoreReturnsArrow;
use tpcdsgen_arrow::StoreSalesArrow;
use tpcdsgen_arrow::TimeDimArrow;
use tpcdsgen_arrow::WarehouseArrow;
use tpcdsgen_arrow::WebPageArrow;
use tpcdsgen_arrow::WebReturnsArrow;
use tpcdsgen_arrow::WebSalesArrow;
use tpcdsgen_arrow::WebSiteArrow;
use tracing::info;
use vortex::utils::parallelism::get_available_parallelism;

use crate::Format;
use crate::utils::parquet::ParquetTableJob;
use crate::utils::parquet::parquet_writer_properties;
use crate::utils::parquet::write_parquet_tables;

const BATCH_SIZE: usize = 8192 * 16;

/// Matches the row group size of DuckDB's Parquet writer, which previously generated this data.
const ROW_GROUP_SIZE: usize = 122_880;

/// Generate every TPC-DS table as a single Parquet file per table under `base_dir/parquet`.
///
/// The files use the column names and types of the TPC-DS specification (and of DuckDB's
/// `dsdgen`). Tables are generated on a shared pool of one worker per core. Already generated
/// tables are skipped.
///
/// Each table is one unit of work: the released `tpcdsgen-arrow` always starts generating at a
/// table's first row, so a table cannot yet be split into parts.
pub async fn generate_tpcds(base_dir: PathBuf, scale_factor: String) -> Result<PathBuf> {
    let output_dir = base_dir.join(Format::Parquet.name());
    fs::create_dir_all(&output_dir)?;

    let scale_factor: f64 = scale_factor.parse()?;
    // Follow the C `dsdgen` reference rather than the Trino Java port. The generated values
    // still differ from DuckDB's bundled `dsdgen`, which is a different version of the C tool.
    let session = SessionBuilder::new()
        .with_scale_factor(scale_factor)
        .with_compat_mode(CompatMode::C)
        .build()?;

    let jobs = TABLES
        .map(|table| {
            let session = session.clone();
            ParquetTableJob {
                path: output_dir.join(format!("{table}.{}", Format::Parquet.ext())),
                num_parts: 1,
                write_part: Box::new(move |_, path| {
                    info!(table, scale_factor, "Generating TPC-DS table");
                    write_parquet(path, table_batches(table, session.clone()))
                }),
            }
        })
        .into_iter()
        .collect();
    let workers = get_available_parallelism().unwrap_or(1);
    tokio::task::spawn_blocking(move || write_parquet_tables(jobs, workers)).await??;

    Ok(output_dir)
}

const TABLES: [&str; 24] = [
    "call_center",
    "catalog_page",
    "catalog_returns",
    "catalog_sales",
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "inventory",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "store_returns",
    "store_sales",
    "time_dim",
    "warehouse",
    "web_page",
    "web_returns",
    "web_sales",
    "web_site",
];

fn table_batches(table: &str, session: Session) -> Box<dyn RecordBatchIterator> {
    match table {
        "call_center" => Box::new(CallCenterArrow::new(session).with_batch_size(BATCH_SIZE)),
        "catalog_page" => Box::new(CatalogPageArrow::new(session).with_batch_size(BATCH_SIZE)),
        "catalog_returns" => {
            Box::new(CatalogReturnsArrow::new(session).with_batch_size(BATCH_SIZE))
        }
        "catalog_sales" => Box::new(CatalogSalesArrow::new(session).with_batch_size(BATCH_SIZE)),
        "customer" => Box::new(CustomerArrow::new(session).with_batch_size(BATCH_SIZE)),
        "customer_address" => {
            Box::new(CustomerAddressArrow::new(session).with_batch_size(BATCH_SIZE))
        }
        "customer_demographics" => {
            Box::new(CustomerDemographicsArrow::new(session).with_batch_size(BATCH_SIZE))
        }
        "date_dim" => Box::new(DateDimArrow::new(session).with_batch_size(BATCH_SIZE)),
        "household_demographics" => {
            Box::new(HouseholdDemographicsArrow::new(session).with_batch_size(BATCH_SIZE))
        }
        "income_band" => Box::new(IncomeBandArrow::new(session).with_batch_size(BATCH_SIZE)),
        "inventory" => Box::new(InventoryArrow::new(session).with_batch_size(BATCH_SIZE)),
        "item" => Box::new(ItemArrow::new(session).with_batch_size(BATCH_SIZE)),
        "promotion" => Box::new(PromotionArrow::new(session).with_batch_size(BATCH_SIZE)),
        "reason" => Box::new(ReasonArrow::new(session).with_batch_size(BATCH_SIZE)),
        "ship_mode" => Box::new(ShipModeArrow::new(session).with_batch_size(BATCH_SIZE)),
        "store" => Box::new(StoreArrow::new(session).with_batch_size(BATCH_SIZE)),
        "store_returns" => Box::new(StoreReturnsArrow::new(session).with_batch_size(BATCH_SIZE)),
        "store_sales" => Box::new(StoreSalesArrow::new(session).with_batch_size(BATCH_SIZE)),
        "time_dim" => Box::new(TimeDimArrow::new(session).with_batch_size(BATCH_SIZE)),
        "warehouse" => Box::new(WarehouseArrow::new(session).with_batch_size(BATCH_SIZE)),
        "web_page" => Box::new(WebPageArrow::new(session).with_batch_size(BATCH_SIZE)),
        "web_returns" => Box::new(WebReturnsArrow::new(session).with_batch_size(BATCH_SIZE)),
        "web_sales" => Box::new(WebSalesArrow::new(session).with_batch_size(BATCH_SIZE)),
        "web_site" => Box::new(WebSiteArrow::new(session).with_batch_size(BATCH_SIZE)),
        _ => unreachable!("unknown TPC-DS table {table}"),
    }
}

fn write_parquet(path: &Path, batches: Box<dyn RecordBatchIterator>) -> Result<()> {
    let schema = spec_schema(batches.schema());
    let props = parquet_writer_properties()
        .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(path)?, Arc::clone(&schema), Some(props))?;
    // Fail on values that do not fit the narrower spec types instead of silently nulling them.
    let options = CastOptions {
        safe: false,
        ..Default::default()
    };
    for batch in batches {
        let columns = batch
            .columns()
            .iter()
            .zip(schema.fields())
            .map(|(column, field)| cast_with_options(column, field.data_type(), &options))
            .collect::<Result<Vec<_>, _>>()?;
        writer.write(&RecordBatch::try_new(Arc::clone(&schema), columns)?)?;
    }
    writer.close()?;
    Ok(())
}

/// Columns that `tpcdsgen-arrow` names after the Trino Java port instead of the specification.
const RENAMED_COLUMNS: [(&str, &str); 5] = [
    ("cr_return_amount_inc_tax", "cr_return_amt_inc_tax"),
    ("ib_income_band_id", "ib_income_band_sk"),
    ("r_reason_description", "r_reason_desc"),
    ("s_tax_precentage", "s_tax_percentage"),
    ("wr_store_credit", "wr_account_credit"),
];

/// Integer columns that DuckDB declares as `INTEGER`. All other integers are `BIGINT`.
const INT32_COLUMNS: [&str; 6] = [
    "c_last_review_date_sk",
    "cd_dep_college_count",
    "hd_vehicle_count",
    "ib_upper_bound",
    "inv_quantity_on_hand",
    "wp_max_ad_count",
];

/// Map a `tpcdsgen-arrow` schema onto the column names and types of the TPC-DS specification.
fn spec_schema(schema: &SchemaRef) -> SchemaRef {
    let fields = schema.fields().iter().map(|field| {
        let name = RENAMED_COLUMNS
            .iter()
            .find(|(from, _)| *from == field.name())
            .map_or_else(|| field.name().as_str(), |(_, to)| to);
        let data_type = match field.data_type() {
            _ if name.ends_with("_street_number") => DataType::Utf8,
            _ if name.ends_with("_gmt_offset") || name.ends_with("_tax_percentage") => {
                DataType::Decimal128(5, 2)
            }
            DataType::Decimal128(..) if name == "p_cost" => DataType::Decimal128(15, 2),
            DataType::Decimal128(..) => DataType::Decimal128(7, 2),
            DataType::Int32 if !INT32_COLUMNS.contains(&name) => DataType::Int64,
            data_type => data_type.clone(),
        };
        Field::new(name, data_type, true)
    });
    Arc::new(Schema::new(fields.collect::<Vec<_>>()))
}
