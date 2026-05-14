// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real-data column loaders for the P7 layered-pco-bench.
//!
//! Each function returns a vector of `(column_name, dataset_label, PrimitiveArray)`
//! tuples. Failures to load (e.g. missing fixtures) yield an empty vector plus a
//! one-line note printed to stderr. The bench main loop concatenates these onto
//! the synthetic columns from P6 and runs the existing 5-variant measurement.

use arrow_array::cast::AsArray;
use arrow_array::types::Date32Type;
use arrow_array::types::Decimal128Type;
use arrow_array::types::Int64Type;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::OrderArrow;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

/// A single column under test.
pub struct DatasetColumn {
    /// Dataset label, e.g. "tpch_sf0p1_lineitem".
    pub dataset: &'static str,
    /// Column label, e.g. "l_orderkey".
    pub column: &'static str,
    /// Encoded as i64 for uniform handling by the variant runners.
    pub array: PrimitiveArray,
}

/// Maximum rows to feed each variant. Limits the per-column runtime.
pub const MAX_ROWS: usize = 1_000_000;

/// Build all the TPC-H columns. Returns empty + stderr note if generation fails.
pub fn tpch_columns() -> Vec<DatasetColumn> {
    match build_tpch_inner() {
        Ok(cols) => cols,
        Err(e) => {
            eprintln!("tpch: skipped ({e})");
            Vec::new()
        }
    }
}

fn build_tpch_inner() -> VortexResult<Vec<DatasetColumn>> {
    // SF=0.1 gives ~600k lineitem rows and ~150k orders rows.
    let scale_factor = 0.1;
    let lineitem_label: &'static str = "tpch_sf0p1_lineitem";
    let orders_label: &'static str = "tpch_sf0p1_orders";

    let mut out = Vec::new();

    // ---- LineItem ----
    {
        let generator = LineItemGenerator::new(scale_factor, 1, 1);
        let arrow_iter = LineItemArrow::new(generator).with_batch_size(65_536);
        let mut l_orderkey = BufferMut::<i64>::with_capacity(MAX_ROWS);
        let mut l_extendedprice_cents = BufferMut::<i64>::with_capacity(MAX_ROWS);
        let mut l_shipdate = BufferMut::<i32>::with_capacity(MAX_ROWS);
        let mut l_quantity_cents = BufferMut::<i64>::with_capacity(MAX_ROWS);

        for batch in arrow_iter {
            if l_orderkey.len() >= MAX_ROWS {
                break;
            }
            let remaining = MAX_ROWS - l_orderkey.len();
            let take = batch.num_rows().min(remaining);

            let ok = batch
                .column_by_name("l_orderkey")
                .ok_or_else(|| vortex_error::vortex_err!("missing l_orderkey"))?
                .as_primitive::<Int64Type>();
            extend_i64(&mut l_orderkey, ok, take);

            let ep = batch
                .column_by_name("l_extendedprice")
                .ok_or_else(|| vortex_error::vortex_err!("missing l_extendedprice"))?
                .as_primitive::<Decimal128Type>();
            extend_decimal128_as_i64(&mut l_extendedprice_cents, ep, take);

            let sd = batch
                .column_by_name("l_shipdate")
                .ok_or_else(|| vortex_error::vortex_err!("missing l_shipdate"))?
                .as_primitive::<Date32Type>();
            extend_i32(&mut l_shipdate, sd, take);

            let q = batch
                .column_by_name("l_quantity")
                .ok_or_else(|| vortex_error::vortex_err!("missing l_quantity"))?
                .as_primitive::<Decimal128Type>();
            extend_decimal128_as_i64(&mut l_quantity_cents, q, take);
        }

        out.push(DatasetColumn {
            dataset: lineitem_label,
            column: "l_orderkey",
            array: PrimitiveArray::new(l_orderkey.freeze(), Validity::NonNullable),
        });
        out.push(DatasetColumn {
            dataset: lineitem_label,
            column: "l_extendedprice_cents",
            array: PrimitiveArray::new(l_extendedprice_cents.freeze(), Validity::NonNullable),
        });
        out.push(DatasetColumn {
            dataset: lineitem_label,
            column: "l_shipdate_i64",
            array: PrimitiveArray::new(i32_buf_to_i64(l_shipdate).freeze(), Validity::NonNullable),
        });
        out.push(DatasetColumn {
            dataset: lineitem_label,
            column: "l_quantity_cents",
            array: PrimitiveArray::new(l_quantity_cents.freeze(), Validity::NonNullable),
        });
    }

    // ---- Orders ----
    {
        let generator = OrderGenerator::new(scale_factor, 1, 1);
        let arrow_iter = OrderArrow::new(generator).with_batch_size(65_536);
        let mut o_orderkey = BufferMut::<i64>::with_capacity(MAX_ROWS);
        for batch in arrow_iter {
            if o_orderkey.len() >= MAX_ROWS {
                break;
            }
            let remaining = MAX_ROWS - o_orderkey.len();
            let take = batch.num_rows().min(remaining);
            let ok = batch
                .column_by_name("o_orderkey")
                .ok_or_else(|| vortex_error::vortex_err!("missing o_orderkey"))?
                .as_primitive::<Int64Type>();
            extend_i64(&mut o_orderkey, ok, take);
        }
        out.push(DatasetColumn {
            dataset: orders_label,
            column: "o_orderkey",
            array: PrimitiveArray::new(o_orderkey.freeze(), Validity::NonNullable),
        });
    }

    Ok(out)
}

fn extend_i64(buf: &mut BufferMut<i64>, arr: &arrow_array::PrimitiveArray<Int64Type>, take: usize) {
    let values: &[i64] = &arr.values()[..take.min(arr.len())];
    buf.extend_from_slice(values);
}

fn extend_i32(
    buf: &mut BufferMut<i32>,
    arr: &arrow_array::PrimitiveArray<Date32Type>,
    take: usize,
) {
    let values: &[i32] = &arr.values()[..take.min(arr.len())];
    buf.extend_from_slice(values);
}

/// Cast a Decimal128 column to i64 by narrowing each i128 value. TPC-H
/// decimals at scale (15, 2) are bounded well below `i64::MAX`, so the cast
/// is lossless in practice. Out-of-range values would saturate; in the
/// unlikely event of saturation we still produce a column the variants can
/// round-trip.
#[allow(
    clippy::cast_possible_truncation,
    reason = "TPC-H Decimal128(15,2) values fit in i64; saturation is acceptable"
)]
fn extend_decimal128_as_i64(
    buf: &mut BufferMut<i64>,
    arr: &arrow_array::PrimitiveArray<Decimal128Type>,
    take: usize,
) {
    let n = take.min(arr.len());
    let values: &[i128] = &arr.values()[..n];
    buf.reserve(n);
    for &v in values {
        // The Decimal128(15, 2) range is roughly ±10^15, far inside i64.
        buf.push(v as i64);
    }
}

/// Widen an `i32` buffer to `i64` so all dataset columns share a single
/// element type and the variant runner can dispatch uniformly.
fn i32_buf_to_i64(src: BufferMut<i32>) -> BufferMut<i64> {
    let mut out = BufferMut::<i64>::with_capacity(src.len());
    for v in src.as_slice() {
        out.push(*v as i64);
    }
    out
}
