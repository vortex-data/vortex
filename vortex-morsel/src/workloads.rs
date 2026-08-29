// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shape-matched workloads for the evaluation.
//!
//! The named suites in the prototype plan (FineWeb, TPC-H SF10, ClickBench) need multi-gigabyte
//! downloads that this environment cannot fetch, and the harness holds segments in memory. What
//! these workloads reproduce instead is the *shape* the plan says those suites lower to:
//! struct-of-chunked-flat columns with per-column chunk boundaries that do not agree, scanned
//! under conjunctive filters of varying selectivity with narrow and wide projections.
//!
//! Two things follow from that and are stated here rather than buried in the numbers: the
//! absolute wall times are not comparable to the recorded suite numbers, and any effect that
//! depends on a specific encoding's decode cost (FSST, ALP-RD, dictionary) is not exercised.
//! What *is* exercised is the executor's own overhead — per-morsel scheduling, planning,
//! cutting, decode reuse — which is what gate E1 measures.

use std::fs::File;
use std::path::Path;

use arrow_array::Array;
use arrow_array::Int64Array;
use arrow_cast::cast;
use arrow_schema::DataType;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::Nullability;
use vortex_array::expr::and;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::fixtures::Column;
use crate::harness::Query;

/// A named workload: how to build its columns, and the queries to run against it.
pub struct Workload {
    /// Short name for reporting.
    pub name: &'static str,
    /// One line on what shape this reproduces.
    pub shape: &'static str,
    /// The columns, already chunked.
    pub columns: Vec<Column>,
    /// The queries.
    pub queries: Vec<Query>,
}

const REAL_CLICKBENCH_COLUMNS: [&str; 8] = [
    "CounterID",
    "AdvEngineID",
    "EventDate",
    "IsRefresh",
    "DontCountHits",
    "URLHash",
    "WindowClientWidth",
    "WindowClientHeight",
];

/// Load exact scan inputs for zero-based ClickBench Q1 and Q41 from an official Parquet shard.
///
/// The prototype executes the query's scan/filter/projection portion. Its aggregation, grouping,
/// ordering, limit, and offset remain outside the executor's supported operator set.
pub fn real_clickbench(path: &Path) -> VortexResult<Workload> {
    let file = File::open(path)
        .map_err(|err| vortex_err!("failed to open ClickBench shard {}: {err}", path.display()))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| vortex_err!("failed to inspect ClickBench shard: {err}"))?;
    let mut projection = REAL_CLICKBENCH_COLUMNS
        .iter()
        .map(|name| {
            builder
                .schema()
                .index_of(name)
                .map_err(|err| vortex_err!("ClickBench column {name}: {err}"))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    projection.sort_unstable();
    let projection = ProjectionMask::roots(builder.parquet_schema(), projection);
    builder = builder.with_projection(projection).with_batch_size(131_072);

    let mut chunks = (0..REAL_CLICKBENCH_COLUMNS.len())
        .map(|_| Vec::<ArrayRef>::new())
        .collect::<Vec<_>>();
    for batch in builder
        .build()
        .map_err(|err| vortex_err!("failed to create ClickBench reader: {err}"))?
    {
        let batch = batch.map_err(|err| vortex_err!("failed to decode ClickBench batch: {err}"))?;
        for (column_chunks, name) in chunks.iter_mut().zip(REAL_CLICKBENCH_COLUMNS) {
            let source = batch
                .column_by_name(name)
                .ok_or_else(|| vortex_err!("ClickBench batch is missing {name}"))?;
            let values = cast(source, &DataType::Int64)
                .map_err(|err| vortex_err!("cannot cast ClickBench {name} to i64: {err}"))?;
            let values = values
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| vortex_err!("ClickBench {name} cast did not produce i64"))?;
            if values.null_count() != 0 {
                return Err(vortex_err!("ClickBench {name} contains nulls"));
            }
            column_chunks.push(
                PrimitiveArray::new(Buffer::copy_from(values.values()), Validity::NonNullable)
                    .into_array(),
            );
        }
    }

    let columns = REAL_CLICKBENCH_COLUMNS
        .into_iter()
        .zip(chunks)
        .map(|(name, chunks)| Column::new(name, chunks))
        .collect();
    let july_2013 = and(
        gt(get_item("EventDate", root()), lit(15_886i64)),
        lt(get_item("EventDate", root()), lit(15_918i64)),
    );
    let filter = and(
        and(eq(get_item("CounterID", root()), lit(62i64)), july_2013),
        and(
            and(
                eq(get_item("IsRefresh", root()), lit(0i64)),
                eq(get_item("DontCountHits", root()), lit(0i64)),
            ),
            eq(
                get_item("URLHash", root()),
                lit(2_868_770_270_353_813_622i64),
            ),
        ),
    );

    Ok(Workload {
        name: "clickbench-real",
        shape: "official ClickBench shard; exact Q1/Q41 scan/filter/projection inputs",
        columns,
        queries: vec![
            Query {
                name: "ClickBench Q1 scan",
                projection: select(vec!["AdvEngineID"], root()),
                // AdvEngineID is unsigned in the source, so `> 0` is exactly `<> 0`.
                filter: Some(gt(get_item("AdvEngineID", root()), lit(0i64))),
            },
            Query {
                name: "ClickBench Q41 scan",
                projection: select(vec!["WindowClientWidth", "WindowClientHeight"], root()),
                filter: Some(filter),
            },
        ],
    })
}

/// A cheap deterministic pseudo-random sequence, so every run sees identical data.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_u32(&mut self, bound: u32) -> u32 {
        (self.next() >> 33) as u32 % bound
    }
}

/// Cut a generated column into chunks of `chunk_rows`, the last one short.
fn chunk_i32(values: &[i32], chunk_rows: usize) -> Vec<ArrayRef> {
    values
        .chunks(chunk_rows)
        .map(|slice| {
            PrimitiveArray::new(Buffer::copy_from(slice), Validity::NonNullable).into_array()
        })
        .collect()
}

fn chunk_f32(values: &[f32], chunk_rows: usize) -> Vec<ArrayRef> {
    values
        .chunks(chunk_rows)
        .map(|slice| {
            PrimitiveArray::new(Buffer::copy_from(slice), Validity::NonNullable).into_array()
        })
        .collect()
}

fn chunk_str(values: &[String], chunk_rows: usize) -> Vec<ArrayRef> {
    values
        .chunks(chunk_rows)
        .map(|slice| VarBinViewArray::from_iter_str(slice.iter().cloned()).into_array())
        .collect()
}

/// A string-heavy workload in the shape of the FineWeb scan: a wide text column, a URL column,
/// a low-cardinality language column and a float score, each chunked differently.
pub fn string_heavy(rows: usize) -> Workload {
    let mut rng = Rng::new(0x5EED_1234);
    let languages = ["en", "de", "fr", "es", "it", "pt", "nl", "pl"];

    let mut url = Vec::with_capacity(rows);
    let mut text = Vec::with_capacity(rows);
    let mut language = Vec::with_capacity(rows);
    let mut score = Vec::with_capacity(rows);
    let mut tokens = Vec::with_capacity(rows);

    for idx in 0..rows {
        let host = rng.next_u32(5000);
        url.push(format!("https://host{host:05}.example.com/page/{idx}"));
        // Short by FineWeb standards, but long enough that the text column dominates the bytes.
        let words = 12 + rng.next_u32(20) as usize;
        let mut body = String::with_capacity(words * 7);
        for w in 0..words {
            body.push_str(match rng.next_u32(8) {
                0 => "vortex ",
                1 => "google ",
                2 => "search ",
                3 => "index ",
                4 => "column ",
                5 => "layout ",
                6 => "segment ",
                _ => "data ",
            });
            if w % 5 == 4 {
                body.push_str("- ");
            }
        }
        text.push(body);
        language.push(
            languages[rng.next_u32(u32::try_from(languages.len()).unwrap_or(u32::MAX)) as usize]
                .to_string(),
        );
        score.push(rng.next_u32(1000) as f32 / 1000.0);
        tokens.push(rng.next_u32(2048) as i32);
    }

    // Deliberately disagreeing chunk boundaries: the text column is written in small chunks
    // because its rows are big, the scalar columns in large ones.
    let columns = vec![
        Column::new("url", chunk_str(&url, 8_192)),
        Column::new("text", chunk_str(&text, 4_096)),
        Column::new("language", chunk_str(&language, 32_768)),
        Column::new("language_score", chunk_f32(&score, 16_384)),
        Column::new("token_count", chunk_i32(&tokens, 65_536)),
    ];

    let queries = vec![
        Query {
            name: "SH1 select-all",
            projection: select(
                vec!["url", "text", "language", "language_score", "token_count"],
                root(),
            ),
            filter: None,
        },
        Query {
            name: "SH2 lowcard-eq",
            projection: select(vec!["url", "language_score"], root()),
            filter: Some(eq(get_item("language", root()), lit("en"))),
        },
        Query {
            name: "SH3 two-conjuncts",
            projection: select(vec!["url", "text"], root()),
            filter: Some(and(
                eq(get_item("language", root()), lit("en")),
                gt(get_item("language_score", root()), lit(0.92f32)),
            )),
        },
        Query {
            name: "SH4 selective",
            projection: select(vec!["url", "text", "token_count"], root()),
            filter: Some(and(
                gt(get_item("language_score", root()), lit(0.995f32)),
                lt(get_item("token_count", root()), lit(64i32)),
            )),
        },
        Query {
            name: "SH5 empty",
            projection: select(vec!["url", "text"], root()),
            filter: Some(gt(get_item("token_count", root()), lit(1_000_000i32))),
        },
        Query {
            name: "SH6 narrow-project",
            projection: select(vec!["token_count"], root()),
            filter: Some(gt(get_item("language_score", root()), lit(0.5f32))),
        },
    ];

    Workload {
        name: "string-heavy",
        shape: "FineWeb-shaped: wide text plus scalars, five disagreeing chunkings",
        columns,
        queries,
    }
}

/// A wide numeric workload in the shape of a ClickBench scan: many narrow integer columns with
/// point and range predicates over a few of them.
pub fn wide_numeric(rows: usize) -> Workload {
    let mut rng = Rng::new(0xC1CB_BE7C);
    let ncolumns = 20;

    let mut data: Vec<Vec<i32>> = (0..ncolumns).map(|_| Vec::with_capacity(rows)).collect();
    for _ in 0..rows {
        for (idx, column) in data.iter_mut().enumerate() {
            // A mix of cardinalities: low-cardinality flags, mid-cardinality ids, wide values.
            let value = match idx % 4 {
                0 => rng.next_u32(8),
                1 => rng.next_u32(1024),
                2 => rng.next_u32(1_000_000),
                _ => rng.next_u32(64),
            };
            column.push(value as i32);
        }
    }

    // Chunk sizes chosen so no two adjacent columns agree, and no boundary set divides another.
    let chunk_sizes = [16_384usize, 12_288, 20_480, 32_768, 9_216];
    let columns = data
        .into_iter()
        .enumerate()
        .map(|(idx, values)| {
            Column::new(
                format!("c{idx:02}"),
                chunk_i32(&values, chunk_sizes[idx % chunk_sizes.len()]),
            )
        })
        .collect();

    let all: Vec<String> = (0..ncolumns).map(|idx| format!("c{idx:02}")).collect();
    let all_refs: Vec<&str> = all.iter().map(String::as_str).collect();

    let queries = vec![
        Query {
            name: "WN1 select-all",
            projection: select(all_refs.clone(), root()),
            filter: None,
        },
        Query {
            name: "WN2 point-filter",
            projection: select(vec!["c00", "c01", "c02"], root()),
            filter: Some(eq(get_item("c02", root()), lit(12345i32))),
        },
        Query {
            name: "WN3 dashboard",
            projection: select(vec!["c00", "c01", "c04", "c05", "c08", "c09"], root()),
            filter: Some(gt(get_item("c00", root()), lit(0i32))),
        },
        Query {
            name: "WN4 two-conjuncts",
            projection: select(all_refs.clone(), root()),
            filter: Some(and(
                gt(get_item("c00", root()), lit(5i32)),
                lt(get_item("c03", root()), lit(4i32)),
            )),
        },
        Query {
            name: "WN5 selective-wide",
            projection: select(all_refs, root()),
            filter: Some(and(
                gt(get_item("c02", root()), lit(999_000i32)),
                lt(get_item("c01", root()), lit(8i32)),
            )),
        },
        Query {
            name: "WN6 packed",
            projection: pack(
                vec![
                    ("a", get_item("c00", root())),
                    ("b", get_item("c06", root())),
                ],
                Nullability::NonNullable,
            ),
            filter: Some(lt(get_item("c07", root()), lit(16i32))),
        },
    ];

    Workload {
        name: "wide-numeric",
        shape: "ClickBench-shaped: 20 narrow integer columns, five disagreeing chunkings",
        columns,
        queries,
    }
}

/// A narrow analytic workload in the shape of TPC-H Q6: a conjunctive range filter over three
/// columns, projecting two.
pub fn narrow_analytic(rows: usize) -> Workload {
    let mut rng = Rng::new(0x79C4_0006);

    let mut quantity = Vec::with_capacity(rows);
    let mut discount = Vec::with_capacity(rows);
    let mut price = Vec::with_capacity(rows);
    let mut shipdate = Vec::with_capacity(rows);
    for _ in 0..rows {
        quantity.push(1 + rng.next_u32(50) as i32);
        discount.push(rng.next_u32(11) as f32 / 100.0);
        price.push(rng.next_u32(100_000) as f32 / 100.0);
        shipdate.push(1992 * 365 + rng.next_u32(7 * 365) as i32);
    }

    let columns = vec![
        Column::new("l_quantity", chunk_i32(&quantity, 65_536)),
        Column::new("l_discount", chunk_f32(&discount, 49_152)),
        Column::new("l_extendedprice", chunk_f32(&price, 65_536)),
        Column::new("l_shipdate", chunk_i32(&shipdate, 40_960)),
    ];

    let queries = vec![
        Query {
            name: "NA1 q6-shape",
            projection: select(vec!["l_extendedprice", "l_discount"], root()),
            filter: Some(and(
                and(
                    gt(get_item("l_shipdate", root()), lit(1994 * 365i32)),
                    lt(get_item("l_shipdate", root()), lit(1995 * 365i32)),
                ),
                and(
                    lt(get_item("l_quantity", root()), lit(24i32)),
                    gt(get_item("l_discount", root()), lit(0.05f32)),
                ),
            )),
        },
        Query {
            name: "NA2 q1-shape",
            projection: select(vec!["l_quantity", "l_extendedprice", "l_discount"], root()),
            filter: Some(lt(get_item("l_shipdate", root()), lit(1998 * 365i32))),
        },
        Query {
            name: "NA3 scan-all",
            projection: select(
                vec!["l_quantity", "l_discount", "l_extendedprice", "l_shipdate"],
                root(),
            ),
            filter: None,
        },
    ];

    Workload {
        name: "narrow-analytic",
        shape: "TPC-H Q6/Q1-shaped: conjunctive range filter, narrow projection",
        columns,
        queries,
    }
}
