// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow schemas and `RecordBatch` streaming for the SSB tables.
//!
//! Types match the reference DDL: every numeric column is a 32-bit `INTEGER` (at SF 100 the
//! widest values are `lo_orderkey` at 6e8 and `lo_ordtotalprice` at ~5e7, both comfortably inside
//! `i32`), everything else is a string. `lo_shippriority` is a string because it is one in the
//! reference schema, even though the generator only ever emits `0`.

use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_array::Int32Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use vortex::error::VortexExpect;

use crate::ssb::ssbgen::Customer;
use crate::ssb::ssbgen::CustomerGenerator;
use crate::ssb::ssbgen::Dwdate;
use crate::ssb::ssbgen::DwdateGenerator;
use crate::ssb::ssbgen::Lineorder;
use crate::ssb::ssbgen::LineorderGenerator;
use crate::ssb::ssbgen::Part;
use crate::ssb::ssbgen::PartGenerator;
use crate::ssb::ssbgen::Supplier;
use crate::ssb::ssbgen::SupplierGenerator;

/// One SSB table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsbTable {
    Customer,
    Supplier,
    Part,
    /// The date dimension. Named `dwdate` because `date` is a reserved word in both engines'
    /// parsers, matching the reference load scripts.
    Dwdate,
    Lineorder,
}

impl SsbTable {
    /// Every table, in the order the reference generator emits them.
    ///
    /// The order is cosmetic: each iterator advances its own streams to the right starting point
    /// on construction (`customer` skips past `supplier`'s draws, `lineorder` past the calendar's),
    /// so tables can be generated in any order or in parallel.
    pub const ALL: [SsbTable; 5] = [
        SsbTable::Part,
        SsbTable::Supplier,
        SsbTable::Customer,
        SsbTable::Dwdate,
        SsbTable::Lineorder,
    ];

    /// Name this table registers under, and its Parquet file stem.
    pub fn name(&self) -> &'static str {
        match self {
            SsbTable::Customer => "customer",
            SsbTable::Supplier => "supplier",
            SsbTable::Part => "part",
            SsbTable::Dwdate => "dwdate",
            SsbTable::Lineorder => "lineorder",
        }
    }

    /// Arrow schema of this table.
    pub fn schema(&self) -> SchemaRef {
        let fields: Vec<Field> = match self {
            SsbTable::Customer => vec![
                int("c_custkey"),
                utf8("c_name"),
                utf8("c_address"),
                utf8("c_city"),
                utf8("c_nation"),
                utf8("c_region"),
                utf8("c_phone"),
                utf8("c_mktsegment"),
            ],
            SsbTable::Supplier => vec![
                int("s_suppkey"),
                utf8("s_name"),
                utf8("s_address"),
                utf8("s_city"),
                utf8("s_nation"),
                utf8("s_region"),
                utf8("s_phone"),
            ],
            SsbTable::Part => vec![
                int("p_partkey"),
                utf8("p_name"),
                utf8("p_mfgr"),
                utf8("p_category"),
                utf8("p_brand1"),
                utf8("p_color"),
                utf8("p_type"),
                int("p_size"),
                utf8("p_container"),
            ],
            SsbTable::Dwdate => vec![
                int("d_datekey"),
                utf8("d_date"),
                utf8("d_dayofweek"),
                utf8("d_month"),
                int("d_year"),
                int("d_yearmonthnum"),
                utf8("d_yearmonth"),
                int("d_daynuminweek"),
                int("d_daynuminmonth"),
                int("d_daynuminyear"),
                int("d_monthnuminyear"),
                int("d_weeknuminyear"),
                utf8("d_sellingseason"),
                int("d_lastdayinweekfl"),
                int("d_lastdayinmonthfl"),
                int("d_holidayfl"),
                int("d_weekdayfl"),
            ],
            SsbTable::Lineorder => vec![
                int("lo_orderkey"),
                int("lo_linenumber"),
                int("lo_custkey"),
                int("lo_partkey"),
                int("lo_suppkey"),
                int("lo_orderdate"),
                utf8("lo_orderpriority"),
                int("lo_shippriority"),
                int("lo_quantity"),
                int("lo_extendedprice"),
                int("lo_ordtotalprice"),
                int("lo_discount"),
                int("lo_revenue"),
                int("lo_supplycost"),
                int("lo_tax"),
                int("lo_commitdate"),
                utf8("lo_shipmode"),
            ],
        };
        Arc::new(Schema::new(fields))
    }

    /// Stream this table as `RecordBatch`es of at most `batch_size` rows.
    ///
    /// `batch_size` is a [`NonZeroUsize`] because zero would otherwise yield an empty iterator for
    /// every table, silently reporting a full dataset as empty.
    pub fn batches(
        &self,
        scale_factor: f64,
        batch_size: NonZeroUsize,
    ) -> Box<dyn Iterator<Item = RecordBatch> + Send> {
        let schema = self.schema();
        match self {
            SsbTable::Customer => Box::new(Batched::new(
                CustomerGenerator::new(scale_factor).iter(),
                batch_size,
                schema,
                customer_batch,
            )),
            SsbTable::Supplier => Box::new(Batched::new(
                SupplierGenerator::new(scale_factor).iter(),
                batch_size,
                schema,
                supplier_batch,
            )),
            SsbTable::Part => Box::new(Batched::new(
                PartGenerator::new(scale_factor).iter(),
                batch_size,
                schema,
                part_batch,
            )),
            SsbTable::Dwdate => Box::new(Batched::new(
                DwdateGenerator::new().iter(),
                batch_size,
                schema,
                dwdate_batch,
            )),
            SsbTable::Lineorder => Box::new(Batched::new(
                LineorderGenerator::new(scale_factor).iter(),
                batch_size,
                schema,
                lineorder_batch,
            )),
        }
    }
}

/// Generated columns are never null, so the schema says so — matching the sibling TPC-H schemas.
fn int(name: &str) -> Field {
    Field::new(name, DataType::Int32, false)
}

fn utf8(name: &str) -> Field {
    Field::new(name, DataType::Utf8, false)
}

/// Chunks a row iterator into `RecordBatch`es using a per-table builder.
struct Batched<I, F> {
    rows: I,
    batch_size: NonZeroUsize,
    schema: SchemaRef,
    build: F,
}

impl<I, F> Batched<I, F> {
    fn new(rows: I, batch_size: NonZeroUsize, schema: SchemaRef, build: F) -> Self {
        Self {
            rows,
            batch_size,
            schema,
            build,
        }
    }
}

impl<I, F> Iterator for Batched<I, F>
where
    I: Iterator,
    F: Fn(&[I::Item], &SchemaRef) -> RecordBatch,
{
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let rows: Vec<I::Item> = self.rows.by_ref().take(self.batch_size.get()).collect();
        if rows.is_empty() {
            return None;
        }
        Some((self.build)(&rows, &self.schema))
    }
}

/// Assemble a batch, treating a schema mismatch as the programming error it is.
fn batch(schema: &SchemaRef, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::clone(schema), columns)
        .vortex_expect("ssb: generated columns do not match the table schema")
}

fn ints(values: impl IntoIterator<Item = i32>) -> ArrayRef {
    Arc::new(Int32Array::from_iter_values(values))
}

fn strings<'a>(values: impl IntoIterator<Item = &'a str>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(values))
}

/// Narrow a 64-bit key or price to the `INTEGER` the reference schema declares.
///
/// Unreachable in practice: [`validate_scale_factor`](super::validate_scale_factor) rejects any
/// scale factor whose largest `lo_orderkey` would not fit, and every other column is far smaller.
fn narrow(value: i64) -> i32 {
    i32::try_from(value).vortex_expect(
        "ssb: value exceeds the schema's INTEGER range; validate_scale_factor should have \
         rejected this scale factor",
    )
}

fn customer_batch(rows: &[Customer], schema: &SchemaRef) -> RecordBatch {
    let phones: Vec<String> = rows.iter().map(|r| r.c_phone.to_string()).collect();
    batch(
        schema,
        vec![
            ints(rows.iter().map(|r| narrow(r.c_custkey))),
            strings(rows.iter().map(|r| r.c_name.as_str())),
            strings(rows.iter().map(|r| r.c_address.as_str())),
            strings(rows.iter().map(|r| r.c_city.as_str())),
            strings(rows.iter().map(|r| r.c_nation)),
            strings(rows.iter().map(|r| r.c_region)),
            strings(phones.iter().map(String::as_str)),
            strings(rows.iter().map(|r| r.c_mktsegment)),
        ],
    )
}

fn supplier_batch(rows: &[Supplier], schema: &SchemaRef) -> RecordBatch {
    let phones: Vec<String> = rows.iter().map(|r| r.s_phone.to_string()).collect();
    batch(
        schema,
        vec![
            ints(rows.iter().map(|r| narrow(r.s_suppkey))),
            strings(rows.iter().map(|r| r.s_name.as_str())),
            strings(rows.iter().map(|r| r.s_address.as_str())),
            strings(rows.iter().map(|r| r.s_city.as_str())),
            strings(rows.iter().map(|r| r.s_nation)),
            strings(rows.iter().map(|r| r.s_region)),
            strings(phones.iter().map(String::as_str)),
        ],
    )
}

fn part_batch(rows: &[Part], schema: &SchemaRef) -> RecordBatch {
    batch(
        schema,
        vec![
            ints(rows.iter().map(|r| narrow(r.p_partkey))),
            strings(rows.iter().map(|r| r.p_name.as_str())),
            strings(rows.iter().map(|r| r.p_mfgr.as_str())),
            strings(rows.iter().map(|r| r.p_category.as_str())),
            strings(rows.iter().map(|r| r.p_brand1.as_str())),
            strings(rows.iter().map(|r| r.p_color)),
            strings(rows.iter().map(|r| r.p_type)),
            ints(rows.iter().map(|r| r.p_size)),
            strings(rows.iter().map(|r| r.p_container)),
        ],
    )
}

fn dwdate_batch(rows: &[Dwdate], schema: &SchemaRef) -> RecordBatch {
    batch(
        schema,
        vec![
            ints(rows.iter().map(|r| r.d_datekey)),
            strings(rows.iter().map(|r| r.d_date.as_str())),
            strings(rows.iter().map(|r| r.d_dayofweek)),
            strings(rows.iter().map(|r| r.d_month)),
            ints(rows.iter().map(|r| r.d_year)),
            ints(rows.iter().map(|r| r.d_yearmonthnum)),
            strings(rows.iter().map(|r| r.d_yearmonth.as_str())),
            ints(rows.iter().map(|r| r.d_daynuminweek)),
            ints(rows.iter().map(|r| r.d_daynuminmonth)),
            ints(rows.iter().map(|r| r.d_daynuminyear)),
            ints(rows.iter().map(|r| r.d_monthnuminyear)),
            ints(rows.iter().map(|r| r.d_weeknuminyear)),
            strings(rows.iter().map(|r| r.d_sellingseason)),
            ints(rows.iter().map(|r| r.d_lastdayinweekfl)),
            ints(rows.iter().map(|r| r.d_lastdayinmonthfl)),
            ints(rows.iter().map(|r| r.d_holidayfl)),
            ints(rows.iter().map(|r| r.d_weekdayfl)),
        ],
    )
}

fn lineorder_batch(rows: &[Lineorder], schema: &SchemaRef) -> RecordBatch {
    batch(
        schema,
        vec![
            ints(rows.iter().map(|r| narrow(r.lo_orderkey))),
            ints(rows.iter().map(|r| r.lo_linenumber)),
            ints(rows.iter().map(|r| narrow(r.lo_custkey))),
            ints(rows.iter().map(|r| narrow(r.lo_partkey))),
            ints(rows.iter().map(|r| narrow(r.lo_suppkey))),
            ints(rows.iter().map(|r| r.lo_orderdate)),
            strings(rows.iter().map(|r| r.lo_orderpriority)),
            ints(rows.iter().map(|r| r.lo_shippriority)),
            ints(rows.iter().map(|r| r.lo_quantity)),
            ints(rows.iter().map(|r| narrow(r.lo_extendedprice))),
            ints(rows.iter().map(|r| narrow(r.lo_ordtotalprice))),
            ints(rows.iter().map(|r| r.lo_discount)),
            ints(rows.iter().map(|r| narrow(r.lo_revenue))),
            ints(rows.iter().map(|r| narrow(r.lo_supplycost))),
            ints(rows.iter().map(|r| r.lo_tax)),
            ints(rows.iter().map(|r| r.lo_commitdate)),
            strings(rows.iter().map(|r| r.lo_shipmode)),
        ],
    )
}
