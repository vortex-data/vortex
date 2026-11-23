// SPDX-FileCopyrightText: 2025-2026 Copyright The Apache Software Foundation
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileComment: Derived from spatialbench-arrow/src at tag sedona-spatialbench-0.2.0 at https://github.com/apache/sedona-spatialbench
// SPDX-FileNotice: https://github.com/apache/sedona-spatialbench/blob/sedona-spatialbench-0.2.0/NOTICE

//! Arrow record batches for the SpatialBench tables that are generated in-process.
//!
//! This is the `spatialbench-arrow` crate's conversion of the `building`, `customer`, and `trip`
//! generators, built on the workspace's Arrow. The `spatialbench` generators do not depend on
//! Arrow, but `spatialbench-arrow` does, and its releases trail Arrow by several major versions.
//! With the conversion here, a SpatialBench release never holds back an Arrow, Parquet, or
//! DataFusion upgrade.
//!
//! The schemas and values must match `spatialbench-arrow` at the `spatialbench` version in use, so
//! that the data is the same as upstream's. Compare this file with upstream when `spatialbench`
//! changes version.

use std::fmt::Display;
use std::fmt::Write;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::ArrayRef;
use arrow_array::BinaryArray;
use arrow_array::Decimal128Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringViewArray;
use arrow_array::TimestampMillisecondArray;
use arrow_array::builder::StringViewBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use arrow_schema::TimeUnit;
use geo::Geometry;
use geozero::CoordDimensions;
use geozero::ToWkb;
use spatialbench::dates::TPCHDate;
use spatialbench::decimal::TPCHDecimal;
use spatialbench::generators::Building;
use spatialbench::generators::BuildingGenerator;
use spatialbench::generators::Customer;
use spatialbench::generators::CustomerGenerator;
use spatialbench::generators::Trip;
use spatialbench::generators::TripGenerator;

/// Record batches of one table, and the schema they share.
pub trait RecordBatches: Iterator<Item = Result<RecordBatch>> + Send {
    /// The schema of every batch.
    fn schema(&self) -> &SchemaRef;
}

/// Converts the rows of a `spatialbench` generator into record batches of up to `batch_size`
/// rows.
struct GeneratorBatches<I: Iterator> {
    rows: I,
    batch_size: usize,
    schema: SchemaRef,
    columns: fn(&[I::Item]) -> Result<Vec<ArrayRef>>,
}

impl<I: Iterator> Iterator for GeneratorBatches<I> {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        let rows: Vec<_> = self.rows.by_ref().take(self.batch_size).collect();
        if rows.is_empty() {
            return None;
        }
        Some(
            (self.columns)(&rows)
                .and_then(|columns| Ok(RecordBatch::try_new(Arc::clone(&self.schema), columns)?)),
        )
    }
}

impl<I: Iterator + Send> RecordBatches for GeneratorBatches<I> {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }
}

/// Batches of the `building` table.
pub fn building(generator: BuildingGenerator<'static>, batch_size: usize) -> impl RecordBatches {
    GeneratorBatches {
        rows: generator.iter(),
        batch_size,
        schema: Arc::new(Schema::new(vec![
            Field::new("b_buildingkey", DataType::Int64, false),
            Field::new("b_name", DataType::Utf8View, false),
            Field::new("b_boundary", DataType::Binary, false),
        ])),
        columns: building_columns,
    }
}

fn building_columns(rows: &[Building<'static>]) -> Result<Vec<ArrayRef>> {
    let boundaries = rows
        .iter()
        .map(|r| Geometry::Polygon(r.b_boundary.clone()).to_wkb(CoordDimensions::xy()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(vec![
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.b_buildingkey),
        )),
        Arc::new(string_view_array(rows.iter().map(|r| &r.b_name))?),
        Arc::new(BinaryArray::from_iter_values(boundaries)),
    ])
}

/// Batches of the `customer` table.
pub fn customer(generator: CustomerGenerator<'static>, batch_size: usize) -> impl RecordBatches {
    GeneratorBatches {
        rows: generator.iter(),
        batch_size,
        schema: Arc::new(Schema::new(vec![
            Field::new("c_custkey", DataType::Int64, false),
            Field::new("c_name", DataType::Utf8View, false),
            Field::new("c_address", DataType::Utf8View, false),
            Field::new("c_region", DataType::Utf8View, false),
            Field::new("c_nation", DataType::Utf8View, false),
            Field::new("c_phone", DataType::Utf8View, false),
        ])),
        columns: customer_columns,
    }
}

fn customer_columns(rows: &[Customer<'static>]) -> Result<Vec<ArrayRef>> {
    Ok(vec![
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.c_custkey),
        )),
        Arc::new(string_view_array(rows.iter().map(|r| r.c_name))?),
        Arc::new(string_view_array(rows.iter().map(|r| &r.c_address))?),
        Arc::new(StringViewArray::from_iter_values(
            rows.iter().map(|r| r.c_region),
        )),
        Arc::new(StringViewArray::from_iter_values(
            rows.iter().map(|r| r.c_nation),
        )),
        Arc::new(string_view_array(rows.iter().map(|r| &r.c_phone))?),
    ])
}

/// Batches of the `trip` table.
pub fn trip(generator: TripGenerator, batch_size: usize) -> impl RecordBatches {
    let timestamp = DataType::Timestamp(TimeUnit::Millisecond, None);
    let decimal = DataType::Decimal128(DECIMAL_PRECISION, DECIMAL_SCALE);
    GeneratorBatches {
        rows: generator.iter(),
        batch_size,
        schema: Arc::new(Schema::new(vec![
            Field::new("t_tripkey", DataType::Int64, false),
            Field::new("t_custkey", DataType::Int64, false),
            Field::new("t_driverkey", DataType::Int64, false),
            Field::new("t_vehiclekey", DataType::Int64, false),
            Field::new("t_pickuptime", timestamp.clone(), false),
            Field::new("t_dropofftime", timestamp, false),
            Field::new("t_fare", decimal.clone(), false),
            Field::new("t_tip", decimal.clone(), false),
            Field::new("t_totalamount", decimal.clone(), false),
            Field::new("t_distance", decimal, false),
            Field::new("t_pickuploc", DataType::Binary, false),
            Field::new("t_dropoffloc", DataType::Binary, false),
        ])),
        columns: trip_columns,
    }
}

fn trip_columns(rows: &[Trip]) -> Result<Vec<ArrayRef>> {
    let pickup_locations = rows
        .iter()
        .map(|r| Geometry::Point(r.t_pickuploc).to_wkb(CoordDimensions::xy()))
        .collect::<Result<Vec<_>, _>>()?;
    let dropoff_locations = rows
        .iter()
        .map(|r| Geometry::Point(r.t_dropoffloc).to_wkb(CoordDimensions::xy()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(vec![
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.t_tripkey),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.t_custkey),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.t_driverkey),
        )),
        Arc::new(Int64Array::from_iter_values(
            rows.iter().map(|r| r.t_vehiclekey),
        )),
        Arc::new(timestamp_millis_array(rows.iter().map(|r| r.t_pickuptime))),
        Arc::new(timestamp_millis_array(rows.iter().map(|r| r.t_dropofftime))),
        Arc::new(decimal_array(rows.iter().map(|r| r.t_fare))?),
        Arc::new(decimal_array(rows.iter().map(|r| r.t_tip))?),
        Arc::new(decimal_array(rows.iter().map(|r| r.t_totalamount))?),
        Arc::new(decimal_array(rows.iter().map(|r| r.t_distance))?),
        Arc::new(BinaryArray::from_iter_values(pickup_locations)),
        Arc::new(BinaryArray::from_iter_values(dropoff_locations)),
    ])
}

/// The type in which upstream writes each `TPCHDecimal`'s inner integer: `Decimal128(15, 5)`.
const DECIMAL_PRECISION: u8 = 15;
const DECIMAL_SCALE: i8 = 5;

fn decimal_array(values: impl Iterator<Item = TPCHDecimal>) -> Result<Decimal128Array> {
    Ok(
        Decimal128Array::from_iter_values(values.map(|v| i128::from(v.into_inner())))
            .with_precision_and_scale(DECIMAL_PRECISION, DECIMAL_SCALE)?,
    )
}

fn timestamp_millis_array(values: impl Iterator<Item = TPCHDate>) -> TimestampMillisecondArray {
    TimestampMillisecondArray::from_iter_values(values.map(|v| v.to_unix_epoch_seconds() * 1000))
}

/// Builds a string view array from the `Display` form of each value.
fn string_view_array(values: impl Iterator<Item: Display>) -> Result<StringViewArray> {
    let mut buffer = String::new();
    let mut builder = StringViewBuilder::with_capacity(values.size_hint().0);
    for value in values {
        buffer.clear();
        write!(buffer, "{value}")?;
        builder.append_value(&buffer);
    }
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use spatialbench::dates::MIN_GENERATE_DATE;

    use super::*;

    #[test]
    fn timestamps_are_unix_epoch_millis() {
        let millis = |offset| {
            timestamp_millis_array(std::iter::once(TPCHDate::new(
                MIN_GENERATE_DATE + offset,
                0,
                0,
                0,
            )))
            .value(0)
        };
        assert_eq!(millis(0), 694_224_000_000);
        assert_eq!(millis(100), 702_864_000_000);
        assert_eq!(millis(1234), 800_841_600_000);
    }
}
