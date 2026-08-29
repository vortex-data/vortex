// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real TPC-H data and real TPC-H scan queries.
//!
//! The earlier workloads were *shaped like* TPC-H; this module is the real thing. Data comes from
//! `tpchgen` — the same generator `vortex-bench` uses, already a workspace dependency, so no
//! download is needed — at a caller-chosen scale factor, with dbgen's real schema, real value
//! distributions and real correlations. Queries are the scan portion of the TPC-H queries in
//! `vortex-bench/sql/tpch/`: the projection a scan must produce and the filter that pushes into
//! it, transcribed predicate for predicate.
//!
//! ## What "the scan portion" means, precisely
//!
//! A scan executor produces the rows an engine's aggregation, join and sort operators consume. It
//! does not aggregate, join or sort. So for Q6 —
//!
//! ```sql
//! select sum(l_extendedprice * l_discount) from lineitem
//! where l_shipdate >= date '1994-01-01' and l_shipdate < date '1995-01-01'
//!   and l_discount between 0.05 and 0.07 and l_quantity < 24;
//! ```
//!
//! — the scan's job is exactly `select l_extendedprice, l_discount` under all four predicates, and
//! that is what is benchmarked. The `sum` above it is identical work for either executor and is
//! excluded rather than double-counted. Queries whose scan portion is a bare full-table read of a
//! few columns (Q1) are included precisely because they are the case where an executor has the
//! least room to differ.
//!
//! ## What is deliberately not exercised
//!
//! The fixture is written through a real compressing pipeline (btrblocks: ALP, FSST, RLE,
//! bit-packing, ...) so decode cost — the denominator of every ratio — is real. But it writes
//! **struct-of-chunked-flat only**: no zone maps and no dictionary *layout*. Both are supported by
//! the V1 reader and neither is in P1's scope, so enabling them would compare a pruning executor
//! against a non-pruning one rather than comparing executors. That gap is a real capability
//! difference and is reported as one, not hidden inside a ratio.

use std::sync::Arc;
use std::sync::Arc as StdArc;

use arrow_schema::Schema;
use tpchgen::generators::LineItemGenerator;
use tpchgen_arrow::LineItemArrow;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::and;
use vortex_array::expr::get_item;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::lt_eq;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::LayoutStrategy;
use vortex_session::VortexSession;

use crate::fixtures::Column;
use crate::harness::Query;

/// One generated TPC-H table: its rows as Vortex struct arrays, one per generator batch.
pub struct Table {
    /// The table name.
    pub name: &'static str,
    /// Row batches in generation order.
    pub batches: Vec<ArrayRef>,
    /// Total rows.
    pub row_count: u64,
}

/// Generate `lineitem` at the given scale factor.
///
/// `batch_rows` is the generator's batch size, which becomes the natural chunk granularity before
/// the write pipeline repartitions it. Batches are converted through the session's own Arrow
/// import path — the same one `vortex-bench` uses to build its TPC-H files — so extension types
/// such as `l_shipdate`'s date are imported exactly as a real conversion would import them.
pub fn lineitem(
    session: &VortexSession,
    scale_factor: f64,
    batch_rows: usize,
) -> VortexResult<Table> {
    use tpchgen_arrow::RecordBatchIterator;
    use vortex_arrow::ArrowSessionExt;

    let iter =
        LineItemArrow::new(LineItemGenerator::new(scale_factor, 1, 1)).with_batch_size(batch_rows);
    let schema: StdArc<Schema> = StdArc::clone(iter.schema());

    let mut batches = Vec::new();
    let mut row_count = 0u64;
    for batch in iter {
        row_count += batch.num_rows() as u64;
        batches.push(session.arrow().from_arrow_record_batch(batch, &schema)?);
    }

    Ok(Table {
        name: "lineitem",
        batches,
        row_count,
    })
}

/// The number of days from the Unix epoch to a `yyyy-mm-dd` date, for `Date32` literals.
///
/// TPC-H predicates are all date literals against `l_shipdate`/`l_commitdate`/`l_receiptdate`,
/// which `tpchgen-arrow` emits as `Date32`. Comparing them needs a literal of the same type, so
/// this converts the calendar dates in the query text into the physical representation.
fn date32(year: i32, month: u32, day: u32) -> i32 {
    // Days from civil algorithm (Howard Hinnant), exact for the proleptic Gregorian calendar.
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let mp = (month as i32 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// A `Date32` literal for a calendar date.
fn date_lit(dtype: &DType, year: i32, month: u32, day: u32) -> VortexResult<Expression> {
    Ok(lit(Scalar::primitive_value(
        date32(year, month, day).into(),
        PType::I32,
        dtype.nullability(),
    )))
}

/// The scan portion of the TPC-H queries that push a filter into `lineitem`, plus the two
/// full-scan shapes.
///
/// Each entry names the TPC-H query it comes from and transcribes that query's `lineitem`
/// predicates and the `lineitem` columns its outer operators consume.
pub fn lineitem_queries(dtype: &DType) -> VortexResult<Vec<Query>> {
    let shipdate = || get_item("l_shipdate", root());
    let discount = || get_item("l_discount", root());
    let quantity = || get_item("l_quantity", root());

    let shipdate_dtype = field_dtype(dtype, "l_shipdate")?;
    let decimal_dtype = field_dtype(dtype, "l_discount")?;
    let quantity_dtype = field_dtype(dtype, "l_quantity")?;

    // TPC-H decimals are DECIMAL(15,2); tpchgen-arrow emits them as Decimal128(15, 2), so a
    // literal must carry the same precision and scale to compare without a cast.
    let dec = |value: f64| -> VortexResult<Expression> { decimal_lit(&decimal_dtype, value) };
    let qty = |value: f64| -> VortexResult<Expression> { decimal_lit(&quantity_dtype, value) };

    Ok(vec![
        // Q6: sum(l_extendedprice * l_discount) with four pushed predicates.
        Query {
            name: "Q6",
            projection: select(vec!["l_extendedprice", "l_discount"], root()),
            filter: Some(and(
                and(
                    gt_eq(shipdate(), date_lit(&shipdate_dtype, 1994, 1, 1)?),
                    lt(shipdate(), date_lit(&shipdate_dtype, 1995, 1, 1)?),
                ),
                and(
                    and(gt_eq(discount(), dec(0.05)?), lt_eq(discount(), dec(0.07)?)),
                    lt(quantity(), qty(24.0)?),
                ),
            )),
        },
        // Q1: one pushed date predicate, then a group-by over six lineitem columns.
        Query {
            name: "Q1",
            projection: select(
                vec![
                    "l_returnflag",
                    "l_linestatus",
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                    "l_tax",
                ],
                root(),
            ),
            filter: Some(lt_eq(shipdate(), date_lit(&shipdate_dtype, 1998, 9, 2)?)),
        },
        // Q14: promo revenue over one shipdate month, joined to part.
        Query {
            name: "Q14",
            projection: select(vec!["l_partkey", "l_extendedprice", "l_discount"], root()),
            filter: Some(and(
                gt_eq(shipdate(), date_lit(&shipdate_dtype, 1995, 9, 1)?),
                lt(shipdate(), date_lit(&shipdate_dtype, 1995, 10, 1)?),
            )),
        },
        // Q15: revenue by supplier over one quarter.
        Query {
            name: "Q15",
            projection: select(vec!["l_suppkey", "l_extendedprice", "l_discount"], root()),
            filter: Some(and(
                gt_eq(shipdate(), date_lit(&shipdate_dtype, 1996, 1, 1)?),
                lt(shipdate(), date_lit(&shipdate_dtype, 1996, 4, 1)?),
            )),
        },
        // Q12: the two shipmodes plus the commit/receipt ordering predicates.
        Query {
            name: "Q12",
            projection: select(
                vec!["l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate"],
                root(),
            ),
            filter: Some(and(
                and(
                    lt(
                        get_item("l_commitdate", root()),
                        get_item("l_receiptdate", root()),
                    ),
                    lt(shipdate(), get_item("l_commitdate", root())),
                ),
                and(
                    gt_eq(
                        get_item("l_receiptdate", root()),
                        date_lit(&shipdate_dtype, 1994, 1, 1)?,
                    ),
                    lt(
                        get_item("l_receiptdate", root()),
                        date_lit(&shipdate_dtype, 1995, 1, 1)?,
                    ),
                ),
            )),
        },
        // Q19: the quantity band shared by all three disjuncts, projecting what the join needs.
        Query {
            name: "Q19",
            projection: select(
                vec![
                    "l_partkey",
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                    "l_shipmode",
                    "l_shipinstruct",
                ],
                root(),
            ),
            filter: Some(and(
                gt_eq(quantity(), qty(1.0)?),
                lt_eq(quantity(), qty(30.0)?),
            )),
        },
        // A bare projection with no filter: the case with the least room for an executor to differ.
        Query {
            name: "scan-6col",
            projection: select(
                vec![
                    "l_orderkey",
                    "l_partkey",
                    "l_suppkey",
                    "l_quantity",
                    "l_extendedprice",
                    "l_discount",
                ],
                root(),
            ),
            filter: None,
        },
        // A highly selective point-ish filter: most morsels seal empty.
        Query {
            name: "selective",
            projection: select(vec!["l_orderkey", "l_extendedprice"], root()),
            filter: Some(and(
                and(
                    gt_eq(shipdate(), date_lit(&shipdate_dtype, 1994, 6, 1)?),
                    lt(shipdate(), date_lit(&shipdate_dtype, 1994, 6, 8)?),
                ),
                and(gt_eq(discount(), dec(0.09)?), lt(quantity(), qty(5.0)?)),
            )),
        },
    ])
}

fn field_dtype(dtype: &DType, name: &str) -> VortexResult<DType> {
    dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("lineitem dtype must be a struct"))?
        .field(name)
        .ok_or_else(|| vortex_err!("lineitem has no field {name}"))
}

/// A decimal literal matching the column's precision and scale.
fn decimal_lit(dtype: &DType, value: f64) -> VortexResult<Expression> {
    let DType::Decimal(decimal, _) = dtype else {
        // tpchgen may emit these as floats depending on version; fall back to a float literal.
        return Ok(lit(value));
    };
    let scale = decimal.scale();
    let scaled = (value * 10f64.powi(i32::from(scale))).round();
    // TPC-H literals are small and exactly representable at DECIMAL(15,2); the guard keeps a
    // typo in a query from silently wrapping rather than failing.
    if !scaled.is_finite() || scaled.abs() > i128::MAX as f64 {
        vortex_bail!("decimal literal {value} is out of range for {dtype}");
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the range check above proves the value fits an i128"
    )]
    let scaled = scaled as i128;
    Ok(lit(Scalar::decimal(
        DecimalValue::I128(scaled),
        *decimal,
        dtype.nullability(),
    )))
}

/// Chunk a table's generated batches into the chunk sizes a column should be written at.
///
/// The write pipeline repartitions anyway, so this only sets the pre-write granularity; keeping
/// it a parameter lets the eval show that the executors agree independent of it.
pub fn rechunk(table: &Table, target_rows: usize) -> VortexResult<Vec<ArrayRef>> {
    if target_rows == 0 {
        return Ok(table.batches.clone());
    }
    let mut out = Vec::new();
    for batch in &table.batches {
        let mut offset = 0usize;
        while offset < batch.len() {
            let end = (offset + target_rows).min(batch.len());
            out.push(batch.slice(offset..end)?);
            offset = end;
        }
    }
    Ok(out)
}

/// The columns a `lineitem` fixture needs, as `(name, per-column chunk row count)`.
///
/// Real Vortex files repartition every column onto the same row blocks, so the misalignment the
/// earlier synthetic fixtures forced does not occur here. That is the honest configuration and it
/// removes the effect the leased cells exploit — which is itself worth measuring.
pub fn aligned_chunking(rows_per_chunk: usize) -> usize {
    rows_per_chunk
}

/// Wrap the generated table into the [`crate::fixtures::Column`] form, one column per field.
pub fn columns(
    table: &Table,
    chunk_rows: usize,
    session: &VortexSession,
) -> VortexResult<Vec<Column>> {
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::struct_::StructArrayExt;

    let chunks = rechunk(table, chunk_rows)?;
    let first = chunks
        .first()
        .ok_or_else(|| vortex_err!("lineitem generated no rows"))?;
    let fields = first
        .dtype()
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("lineitem must be a struct"))?
        .clone();

    let mut ctx = session.create_execution_ctx();
    let mut per_field: Vec<Vec<ArrayRef>> =
        vec![Vec::with_capacity(chunks.len()); fields.nfields()];
    for chunk in &chunks {
        let structs = chunk.clone().execute::<StructArray>(&mut ctx)?;
        for (idx, slot) in per_field.iter_mut().enumerate() {
            slot.push(
                structs
                    .unmasked_field_opt(idx)
                    .cloned()
                    .ok_or_else(|| vortex_err!("missing field {idx}"))?,
            );
        }
    }

    Ok(fields
        .names()
        .iter()
        .cloned()
        .zip(per_field)
        .map(|(name, chunks)| Column { name, chunks })
        .collect())
}

/// The write strategy used for TPC-H fixtures: the production compression pipeline restricted to
/// the layouts P1 supports.
///
/// This is `WriteStrategyBuilder`'s stack with the zoned-statistics and dictionary-*layout* stages
/// removed — repartition, coalesce, compress, buffer, chunk, flat — so segments carry real
/// btrblocks encodings while the layout tree stays struct-of-chunked-flat.
pub fn write_strategy(row_block_size: usize, block_target_bytes: u64) -> Arc<dyn LayoutStrategy> {
    use vortex_btrblocks::BtrBlocksCompressorBuilder;
    use vortex_layout::layouts::buffered::BufferedStrategy;
    use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
    use vortex_layout::layouts::compressed::CompressingStrategy;
    use vortex_layout::layouts::compressed::CompressorPlugin;
    use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
    use vortex_layout::layouts::repartition::RepartitionStrategy;
    use vortex_layout::layouts::repartition::RepartitionWriterOptions;

    let compressor: Arc<dyn CompressorPlugin> =
        Arc::new(BtrBlocksCompressorBuilder::default().build());

    let flat = FlatLayoutStrategy::default();
    let chunked = ChunkedLayoutStrategy::new(flat);
    let buffered = BufferedStrategy::new(chunked, 2 * (1 << 20));
    let compressing = CompressingStrategy::new(buffered, compressor);
    let coalescing = RepartitionStrategy::new(
        compressing,
        RepartitionWriterOptions {
            block_size_minimum: block_target_bytes,
            block_len_multiple: row_block_size,
            block_size_target: Some(block_target_bytes),
            canonicalize: true,
        },
    );
    let repartition = RepartitionStrategy::new(
        coalescing,
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: row_block_size,
            block_size_target: None,
            canonicalize: false,
        },
    );
    Arc::new(repartition)
}
