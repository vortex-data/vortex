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
use futures::StreamExt;
use futures::stream;
use parquet::arrow::ArrowWriter;
use tpcdsgen::config::CompatMode;
use tpcdsgen::config::Session;
use tpcdsgen::config::SessionBuilder;
use tpcdsgen::config::Table;
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
use crate::utils::file::temp_download_filepath;
use crate::utils::parquet::parquet_writer_properties;
use crate::utils::split;

const BATCH_SIZE: usize = 8192 * 16;

/// Matches the row group size of DuckDB's Parquet writer, which previously generated this data.
const ROW_GROUP_SIZE: usize = 122_880;

/// Generate every TPC-DS table as Parquet under `base_dir/parquet`.
///
/// Large tables are split into files of about [`split::TARGET_FILE_COMPRESSED_BYTES`], named by
/// [`split::part_file_name`]; every other table is a single `{table}_0001.parquet`. The files use
/// the column names and types of the TPC-DS specification (and of DuckDB's `dsdgen`). Each table
/// is generated on its own blocking thread. Already generated tables are skipped.
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

    let parallelism = get_available_parallelism().unwrap_or(1);
    let mut tasks = stream::iter(TABLES.map(|table| {
        let session = session.clone();
        let output_dir = output_dir.clone();
        tokio::task::spawn_blocking(move || {
            let first_file =
                output_dir.join(split::part_file_name(table, 1, Format::Parquet.ext()));
            if first_file.exists() {
                return Ok(());
            }
            let layout = FileLayout::new(table, &session);
            info!(
                table,
                scale_factor,
                files = layout.num_files,
                "Generating TPC-DS table"
            );
            write_parquet_files(&output_dir, table, table_batches(table, session), layout)
        })
    }))
    .buffer_unordered(parallelism);

    while let Some(result) = tasks.next().await {
        result??;
    }
    drop(tasks);

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

/// How a TPC-DS table's size scales, used to split it into files.
struct TableSize {
    /// The table whose `tpcdsgen` row count drives generation. Returns tables are generated
    /// from their sales tickets.
    source: Table,
    /// Generated rows per source row, such as line items per ticket.
    rows_per_source_row: f64,
    /// Compressed bytes per row: the mean of the Parquet and default Vortex sizes.
    compressed_bytes_per_row: f64,
}

/// Sizes of the tables that can outgrow one file, measured at scale factor 10. The other tables
/// stay well below [`split::TARGET_FILE_COMPRESSED_BYTES`] even at scale factor 1000.
fn table_size(table: &str) -> Option<TableSize> {
    let (source, rows_per_source_row, compressed_bytes_per_row) = match table {
        "store_sales" => (Table::StoreSales, 12.0004, 37.283),
        "store_returns" => (Table::StoreSales, 1.1981, 43.643),
        "catalog_sales" => (Table::CatalogSales, 9.0008, 54.241),
        "catalog_returns" => (Table::CatalogSales, 0.8998, 56.303),
        "web_sales" => (Table::WebSales, 11.9959, 52.890),
        "web_returns" => (Table::WebSales, 1.1987, 55.432),
        "inventory" => (Table::Inventory, 1.0, 2.762),
        "customer" => (Table::Customer, 1.0, 41.164),
        "customer_address" => (Table::CustomerAddress, 1.0, 15.311),
        "item" => (Table::Item, 1.0, 74.525),
        _ => return None,
    };
    Some(TableSize {
        source,
        rows_per_source_row,
        compressed_bytes_per_row,
    })
}

/// How the rows of one table are spread over its files.
#[derive(Clone, Copy)]
struct FileLayout {
    num_files: usize,
    /// Rows in every file but the last, which also takes any rows beyond the estimate.
    rows_per_file: usize,
}

impl FileLayout {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        reason = "row counts are far below the f64 and usize limits"
    )]
    fn new(table: &str, session: &Session) -> Self {
        let Some(size) = table_size(table) else {
            return Self {
                num_files: 1,
                rows_per_file: usize::MAX,
            };
        };
        let rows =
            session.get_scaling().get_row_count(size.source) as f64 * size.rows_per_source_row;
        let num_files = split::num_files(rows * size.compressed_bytes_per_row);
        Self {
            num_files,
            rows_per_file: (rows / num_files as f64).ceil().max(1.0) as usize,
        }
    }
}

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

/// Write `batches` as the Parquet files of `table` in `output_dir`, split by `layout`.
///
/// Files are written to temporary paths and renamed into place last part first, so the first
/// part only exists once the whole table does.
fn write_parquet_files(
    output_dir: &Path,
    table: &str,
    batches: Box<dyn RecordBatchIterator>,
    layout: FileLayout,
) -> Result<()> {
    let schema = spec_schema(batches.schema());
    let props = parquet_writer_properties()
        .set_max_row_group_row_count(Some(ROW_GROUP_SIZE))
        .build();
    // Fail on values that do not fit the narrower spec types instead of silently nulling them.
    let options = CastOptions {
        safe: false,
        ..Default::default()
    };

    let mut files = Vec::with_capacity(layout.num_files);
    let mut writer: Option<ArrowWriter<File>> = None;
    let mut rows_in_file = 0;
    for batch in batches {
        let columns = batch
            .columns()
            .iter()
            .zip(schema.fields())
            .map(|(column, field)| cast_with_options(column, field.data_type(), &options))
            .collect::<Result<Vec<_>, _>>()?;
        let mut batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
        while batch.num_rows() > 0 {
            let current = match writer.as_mut() {
                Some(current) => current,
                None => {
                    let path = temp_download_filepath();
                    let file = File::create(&path)?;
                    files.push(path);
                    rows_in_file = 0;
                    writer.insert(ArrowWriter::try_new(
                        file,
                        Arc::clone(&schema),
                        Some(props.clone()),
                    )?)
                }
            };
            let is_last_file = files.len() == layout.num_files;
            let len = if is_last_file {
                batch.num_rows()
            } else {
                batch.num_rows().min(layout.rows_per_file - rows_in_file)
            };
            current.write(&batch.slice(0, len))?;
            rows_in_file += len;
            batch = batch.slice(len, batch.num_rows() - len);
            if !is_last_file
                && rows_in_file == layout.rows_per_file
                && let Some(full) = writer.take()
            {
                full.close()?;
            }
        }
    }
    match writer {
        Some(writer) => {
            writer.close()?;
        }
        // An empty table still gets one file carrying its schema.
        None if files.is_empty() => {
            let path = temp_download_filepath();
            ArrowWriter::try_new(File::create(&path)?, schema, Some(props))?.close()?;
            files.push(path);
        }
        None => {}
    }

    for (idx, path) in files.iter().enumerate().rev() {
        let name = split::part_file_name(table, idx + 1, Format::Parquet.ext());
        fs::rename(path, output_dir.join(name))?;
    }
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
