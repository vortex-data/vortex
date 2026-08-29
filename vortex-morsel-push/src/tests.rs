// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Correctness suites for the morsel executor.
//!
//! Every suite is differential: the V1 `LayoutReader` is the oracle, and a run passes only when
//! it emits the same rows in the same order. The properties the design document lists are each
//! expressed as a variation the output must be invariant under — thread count, morsel size,
//! conjunct policy, decode-cache budget, chunk alignment.

// Fixture generation counts rows into `i32` columns at sizes that trivially fit; the cast lint
// only makes the generators harder to read.
#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::task::Poll;
use std::task::Waker;
use std::time::Duration;

use futures::FutureExt;
use futures::future::poll_fn;
use parking_lot::Mutex;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::and;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::lt_eq;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutRef;
use vortex_layout::segments::ReadAtNowait;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use crate::DemandHintDelivery;
use crate::ExecutionMode;
use crate::fixtures::Column;
use crate::fixtures::Fixture;
use crate::fixtures::write_fixture;
use crate::harness::MorselConfig;
use crate::harness::Query;
use crate::harness::RunOutcome;
use crate::harness::assert_same_rows;
use crate::harness::run_morsel;
use crate::harness::run_v1;
use crate::nodes::ConjunctMode;

fn session() -> VortexSession {
    array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
}

fn i32_chunks(values: &[i32], boundaries: &[usize]) -> Vec<ArrayRef> {
    cut(values, boundaries)
        .into_iter()
        .map(|slice| {
            PrimitiveArray::new(Buffer::copy_from(slice), Validity::NonNullable).into_array()
        })
        .collect()
}

fn utf8_chunks(values: &[i32], boundaries: &[usize]) -> Vec<ArrayRef> {
    cut(values, boundaries)
        .into_iter()
        .map(|slice| {
            VarBinViewArray::from_iter_str(slice.iter().map(|v| format!("row-{v:06}"))).into_array()
        })
        .collect()
}

/// Split `values` at `boundaries`, which are exclusive ends in ascending order.
fn cut<'a>(values: &'a [i32], boundaries: &[usize]) -> Vec<&'a [i32]> {
    let mut out = Vec::with_capacity(boundaries.len());
    let mut start = 0;
    for &end in boundaries {
        out.push(&values[start..end]);
        start = end;
    }
    assert_eq!(start, values.len(), "boundaries must cover every value");
    out
}

/// The canonical misaligned fixture: three columns cut on three different boundary sets.
fn misaligned_fixture(session: &VortexSession, rows: usize) -> VortexResult<Fixture> {
    let col_a: Vec<i32> = (0..rows as i32).collect();
    let col_b: Vec<i32> = (0..rows as i32).map(|v| (v * 7) % 101).collect();
    let col_c: Vec<i32> = (0..rows as i32).map(|v| (v * 13) % 17).collect();

    let thirds = boundaries(rows, 3);
    let fifths = boundaries(rows, 5);
    let sevenths = boundaries(rows, 7);

    block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&col_a, &thirds)),
                Column::new("b", i32_chunks(&col_b, &fifths)),
                Column::new("c", utf8_chunks(&col_c, &sevenths)),
            ],
            session,
        )
        .await
    })
}

/// The same data with every column cut on the same boundaries — the aligned reference.
fn aligned_fixture(session: &VortexSession, rows: usize) -> VortexResult<Fixture> {
    let col_a: Vec<i32> = (0..rows as i32).collect();
    let col_b: Vec<i32> = (0..rows as i32).map(|v| (v * 7) % 101).collect();
    let col_c: Vec<i32> = (0..rows as i32).map(|v| (v * 13) % 17).collect();
    let single = vec![rows];

    block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&col_a, &single)),
                Column::new("b", i32_chunks(&col_b, &single)),
                Column::new("c", utf8_chunks(&col_c, &single)),
            ],
            session,
        )
        .await
    })
}

fn boundaries(rows: usize, parts: usize) -> Vec<usize> {
    let step = rows.div_ceil(parts);
    let mut out = Vec::with_capacity(parts);
    let mut end = step;
    while end < rows {
        out.push(end);
        end += step;
    }
    out.push(rows);
    out
}

fn queries() -> Vec<Query> {
    vec![
        Query {
            name: "select-all",
            projection: select(vec!["a", "b", "c"], root()),
            filter: None,
        },
        Query {
            name: "project-two",
            projection: select(vec!["a", "c"], root()),
            filter: None,
        },
        Query {
            name: "one-conjunct",
            projection: select(vec!["a", "b"], root()),
            filter: Some(gt(get_item("a", root()), lit(400i32))),
        },
        Query {
            name: "two-conjuncts",
            projection: select(vec!["a", "b", "c"], root()),
            filter: Some(and(
                gt(get_item("a", root()), lit(100i32)),
                lt(get_item("b", root()), lit(50i32)),
            )),
        },
        Query {
            name: "selective",
            projection: select(vec!["a", "c"], root()),
            filter: Some(and(
                gt(get_item("a", root()), lit(900i32)),
                lt(get_item("b", root()), lit(10i32)),
            )),
        },
        Query {
            name: "empty-result",
            projection: select(vec!["a"], root()),
            filter: Some(gt(get_item("a", root()), lit(1_000_000i32))),
        },
        Query {
            name: "filter-on-unprojected",
            projection: select(vec!["c"], root()),
            filter: Some(lt(get_item("b", root()), lit(30i32))),
        },
        Query {
            name: "packed-projection",
            projection: pack(
                vec![("x", get_item("a", root())), ("y", get_item("b", root()))],
                Nullability::NonNullable,
            ),
            filter: Some(gt(get_item("a", root()), lit(200i32))),
        },
    ]
}

const ROWS: usize = 1000;

#[test]
fn q6_ranges_build_three_predicate_sources_and_match_pull_push() -> VortexResult<()> {
    let session = session();
    let fixture = aligned_fixture(&session, ROWS)?;
    let projection = select(vec!["a", "b"], root());
    let a = || get_item("a", root());
    let b = || get_item("b", root());
    let filter = and(
        and(gt_eq(a(), lit(100i32)), lt(a(), lit(900i32))),
        and(
            and(gt_eq(b(), lit(10i32)), lt_eq(b(), lit(80i32))),
            lt(a(), lit(850i32)),
        ),
    );
    let plan = crate::build_plan(
        &fixture.layout,
        &projection,
        Some(&filter),
        ConjunctMode::Cascade,
    )?;
    let predicate_slots = plan
        .sources()
        .iter()
        .filter_map(|source| match source.role {
            crate::SourceRole::Predicate { slot, .. } => Some(slot),
            crate::SourceRole::Projection => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(predicate_slots, [0, 1, 2]);

    let query = Query {
        name: "q6-range-fusion",
        projection,
        filter: Some(filter),
    };
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);
    let oracle = run_v1(&session, &fixture.layout, &segments, &query)?;
    for execution_mode in [ExecutionMode::Pull, ExecutionMode::Push] {
        let actual = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                threads: 2,
                morsel_rows: 128,
                execution_mode,
                ..Default::default()
            },
        )?;
        assert_same_rows(
            &session,
            &v1_dtype(&fixture.layout, &query)?,
            &oracle,
            &actual,
        )?;
        if execution_mode == ExecutionMode::Push {
            let stats = actual.stats.as_ref().expect("morsel runs report stats");
            assert!(stats.push_inline_gates > 0);
            assert_eq!(stats.push_cold_frame_spills, 0);
        }
    }
    Ok(())
}

#[test]
fn null_range_bound_preserves_pull_push_semantics() -> VortexResult<()> {
    let session = session();
    let fixture = aligned_fixture(&session, ROWS)?;
    let null = lit(Scalar::null(DType::Primitive(
        PType::I32,
        Nullability::Nullable,
    )));
    let query = Query {
        name: "null-range-bound",
        projection: select(vec!["a"], root()),
        filter: Some(and(
            gt_eq(get_item("a", root()), null),
            lt_eq(get_item("a", root()), lit(500i32)),
        )),
    };
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);
    let oracle = run_v1(&session, &fixture.layout, &segments, &query)?;
    for execution_mode in [ExecutionMode::Pull, ExecutionMode::Push] {
        let actual = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                threads: 2,
                morsel_rows: 128,
                execution_mode,
                ..Default::default()
            },
        )?;
        assert_same_rows(
            &session,
            &v1_dtype(&fixture.layout, &query)?,
            &oracle,
            &actual,
        )?;
    }
    Ok(())
}

/// Property: the executor agrees with V1 on every query, over misaligned chunks.
#[rstest]
fn matches_v1_oracle(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
    #[values(1, 2, 4)] threads: usize,
) -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let v1 = run_v1(&session, &fixture.layout, &segments, &query)?;
        let morsel = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                threads,
                execution_mode,
                ..Default::default()
            },
        )
        .map_err(|err| err.with_context(format!("query {}", query.name)))?;
        assert_same_rows(&session, &v1_dtype(&fixture.layout, &query)?, &v1, &morsel)
            .map_err(|err| err.with_context(format!("query {}", query.name)))?;
    }
    Ok(())
}

#[rstest]
fn demand_hint_delivery_is_not_observable(
    #[values(
        DemandHintDelivery::Immediate,
        DemandHintDelivery::Disabled,
        DemandHintDelivery::Delayed(usize::MAX)
    )]
    demand_hints: DemandHintDelivery,
) -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);
    for query in queries() {
        let v1 = run_v1(&session, &fixture.layout, &segments, &query)?;
        let morsel = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                threads: 2,
                execution_mode: ExecutionMode::Push,
                demand_hints,
                ..Default::default()
            },
        )?;
        assert_same_rows(&session, &v1_dtype(&fixture.layout, &query)?, &v1, &morsel)?;
        if query.filter.is_some() {
            let stats = morsel.stats.as_ref().expect("morsel runs report stats");
            assert!(stats.demand_hints_emitted > 0);
            assert!(stats.demand_state_live_max <= ROWS as u64);
            match demand_hints {
                DemandHintDelivery::Immediate => assert!(stats.demand_hints_observed > 0),
                DemandHintDelivery::Disabled => {
                    assert_eq!(stats.demand_hints_observed, 0);
                    assert!(stats.demand_hints_dropped > 0);
                }
                DemandHintDelivery::Delayed(_) => assert!(stats.demand_hints_dropped > 0),
            }
        }
    }
    Ok(())
}

#[test]
fn leaf_batch_crosses_multiple_parent_edges_inline() -> VortexResult<()> {
    let session = session();
    let values: Vec<i32> = (0..32).collect();
    let fixture = block_on(|_handle| async {
        write_fixture(vec![Column::new("a", i32_chunks(&values, &[32]))], &session).await
    })?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);
    let query = Query {
        name: "inline-parent-chain",
        projection: select(vec!["a"], root()),
        filter: None,
    };
    let v1 = run_v1(&session, &fixture.layout, &segments, &query)?;
    let morsel = run_morsel(
        &session,
        &fixture.layout,
        &segments,
        &query,
        MorselConfig {
            execution_mode: ExecutionMode::Push,
            ..Default::default()
        },
    )?;
    assert_same_rows(&session, &v1_dtype(&fixture.layout, &query)?, &v1, &morsel)?;
    let stats = morsel.stats.as_ref().expect("morsel runs report stats");
    assert!(stats.push_inline_transfers >= 2);
    assert_eq!(
        stats.push_pipeline_stage_calls, 3,
        "one terminal leaf batch should cross the unary chain without a final root credit"
    );
    assert!(stats.push_fast_stage_transfers >= 2);
    assert_eq!(
        stats.push_fast_stage_transfers, stats.push_inline_transfers,
        "every payload edge in the unary/cross-boundary chain should stay on the fast path"
    );
    assert_eq!(
        stats.push_cold_frame_spills, 0,
        "a terminal unary chain must not enter the cold frame dispatcher"
    );
    assert_eq!(
        stats.push_runtime_mask_clones, 0,
        "typed routing must move selection with the batch instead of cloning it"
    );
    assert_eq!(stats.push_dispatch_spills, 0);
    Ok(())
}

#[rstest]
fn bounded_stream_resumes_in_order_after_consumer_stall(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
) -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);
    let query = Query {
        name: "bounded-stream",
        projection: select(vec!["a", "b", "c"], root()),
        filter: None,
    };
    let v1 = run_v1(&session, &fixture.layout, &segments, &query)?;
    let plan = Arc::new(crate::build_plan(
        &fixture.layout,
        &query.projection,
        None,
        ConjunctMode::Cascade,
    )?);
    let cut = crate::driver::morsels(&plan, 0);
    let mut stream = crate::MorselScan::new(plan, segments, session.clone())
        .with_threads(4)
        .with_morsels(cut)
        .with_execution_mode(execution_mode)
        .with_share_decodes(false)
        .with_output_capacity(1, 1)
        .into_stream()?;

    std::thread::sleep(Duration::from_millis(20));
    let mut batches = Vec::new();
    for batch in stream.by_ref() {
        batches.push(batch?);
    }
    let (stats, wall) = stream.finish()?;
    let streamed = RunOutcome {
        rows: batches.iter().map(|batch| batch.len()).sum(),
        batches,
        wall,
        time_to_first_batch: stats.time_to_first_batch,
        stats: Some(stats.clone()),
        source_io_requests: None,
        source_io_bytes: None,
    };
    assert_same_rows(
        &session,
        &v1_dtype(&fixture.layout, &query)?,
        &v1,
        &streamed,
    )?;
    assert!(stats.output_credit_blocks > 0);
    assert!(stats.output_rows_max > 1, "one oversized batch must escape");
    if execution_mode == ExecutionMode::Push {
        assert!(stats.push_inline_transfers > 0);
    }
    Ok(())
}

#[rstest]
fn dropping_stream_cancels_stalled_scan(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
) -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);
    let query = Query {
        name: "cancel-stream",
        projection: select(vec!["a", "b", "c"], root()),
        filter: None,
    };
    let plan = Arc::new(crate::build_plan(
        &fixture.layout,
        &query.projection,
        None,
        ConjunctMode::Cascade,
    )?);
    let cut = crate::driver::morsels(&plan, 0);
    let mut stream = crate::MorselScan::new(plan, segments, session)
        .with_threads(4)
        .with_morsels(cut)
        .with_execution_mode(execution_mode)
        .with_share_decodes(false)
        .with_output_capacity(1, 1)
        .into_stream()?;
    drop(stream.next().transpose()?);
    drop(stream);
    Ok(())
}

struct NeverReadySource;

impl SegmentSource for NeverReadySource {
    fn request(&self, _id: SegmentId) -> SegmentFuture {
        futures::future::pending().boxed()
    }

    fn prefers_background_reads(&self) -> bool {
        true
    }
}

#[test]
fn dropping_stream_cancels_never_ready_io() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let query = Query {
        name: "cancel-never-ready",
        projection: select(vec!["a", "b", "c"], root()),
        filter: None,
    };
    let plan = Arc::new(crate::build_plan(
        &fixture.layout,
        &query.projection,
        None,
        ConjunctMode::Cascade,
    )?);
    let cut = crate::driver::morsels(&plan, 0);
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        let stream = crate::MorselScan::new(plan, Arc::new(NeverReadySource), session)
            .with_threads(2)
            .with_morsels(cut)
            .with_execution_mode(ExecutionMode::Push)
            .into_stream();
        match stream {
            Ok(stream) => {
                std::thread::sleep(Duration::from_millis(20));
                drop(stream);
                drop(done_tx.send(Ok(())));
            }
            Err(err) => drop(done_tx.send(Err(err))),
        }
    });
    done_rx
        .recv_timeout(Duration::from_secs(1))
        .map_err(|_| vortex_err!("dropping a stream did not cancel never-ready IO"))??;
    Ok(())
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn rejects_invalid_morsel_cuts_before_starting() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let query = Query {
        name: "invalid-cut",
        projection: select(vec!["a"], root()),
        filter: None,
    };
    let plan = Arc::new(crate::build_plan(
        &fixture.layout,
        &query.projection,
        None,
        ConjunctMode::Cascade,
    )?);
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);
    let row_count = plan.row_count();
    let invalid = [
        vec![],
        vec![0..0, 0..row_count],
        vec![1..row_count],
        vec![0..10, 11..row_count],
        vec![0..20, 10..row_count],
        vec![10..row_count, 0..10],
        vec![0..row_count + 1],
        vec![0..row_count - 1],
    ];
    for cut in invalid {
        let result =
            crate::MorselScan::new(Arc::clone(&plan), Arc::clone(&segments), session.clone())
                .with_morsels(cut)
                .into_stream();
        assert!(result.is_err());
    }
    Ok(())
}

/// Property: misaligned chunking is invisible. The same logical table stored with three
/// different per-column chunkings must produce byte-identical output to the single-chunk
/// reference.
#[rstest]
fn misaligned_chunks_match_aligned_reference() -> VortexResult<()> {
    let session = session();
    let misaligned = misaligned_fixture(&session, ROWS)?;
    let aligned = aligned_fixture(&session, ROWS)?;
    let misaligned_segments: Arc<dyn SegmentSource> = Arc::clone(&misaligned.segments);
    let aligned_segments: Arc<dyn SegmentSource> = Arc::clone(&aligned.segments);

    for query in queries() {
        let left = run_morsel(
            &session,
            &misaligned.layout,
            &misaligned_segments,
            &query,
            MorselConfig::default(),
        )
        .map_err(|err| err.with_context(format!("query {}", query.name)))?;
        let right = run_morsel(
            &session,
            &aligned.layout,
            &aligned_segments,
            &query,
            MorselConfig::default(),
        )?;
        assert_same_rows(
            &session,
            &v1_dtype(&misaligned.layout, &query)?,
            &left,
            &right,
        )
        .map_err(|err| err.with_context(format!("query {}", query.name)))?;
    }
    Ok(())
}

/// The document's specific misaligned-chunk case: fields chunked `[0,3,10)` against `[0,6,10)`.
#[rstest]
fn document_misalignment_case() -> VortexResult<()> {
    let session = session();
    let values: Vec<i32> = (0..10).collect();
    let fixture = block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&values, &[3, 10])),
                Column::new("b", i32_chunks(&values, &[6, 10])),
            ],
            &session,
        )
        .await
    })?;
    let reference = block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&values, &[10])),
                Column::new("b", i32_chunks(&values, &[10])),
            ],
            &session,
        )
        .await
    })?;

    let query = Query {
        name: "doc-case",
        projection: select(vec!["a", "b"], root()),
        filter: Some(gt(get_item("a", root()), lit(2i32))),
    };
    let dtype = v1_dtype(&fixture.layout, &query)?;

    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);
    let reference_segments: Arc<dyn SegmentSource> = Arc::clone(&reference.segments);

    let left = run_morsel(
        &session,
        &fixture.layout,
        &segments,
        &query,
        MorselConfig::default(),
    )?;
    let right = run_morsel(
        &session,
        &reference.layout,
        &reference_segments,
        &query,
        MorselConfig::default(),
    )?;
    let v1 = run_v1(&session, &fixture.layout, &segments, &query)?;

    assert_same_rows(&session, &dtype, &left, &right)?;
    assert_same_rows(&session, &dtype, &left, &v1)?;

    // The morsel cut must be the union of both columns' boundaries.
    let plan = crate::build_plan(
        &fixture.layout,
        &query.projection,
        query.filter.as_ref(),
        ConjunctMode::Cascade,
    )?;
    assert_eq!(plan.natural_splits(), &[3, 6, 10]);
    Ok(())
}

/// Property: the result does not depend on how the scan is cut into morsels.
#[rstest]
fn independent_of_morsel_size(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
    #[values(0, 1, 7, 128, 4096)] morsel_rows: u64,
) -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let dtype = v1_dtype(&fixture.layout, &query)?;
        let v1 = run_v1(&session, &fixture.layout, &segments, &query)?;
        let morsel = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                morsel_rows,
                execution_mode,
                ..Default::default()
            },
        )
        .map_err(|err| err.with_context(format!("query {}", query.name)))?;
        assert_same_rows(&session, &dtype, &v1, &morsel)
            .map_err(|err| err.with_context(format!("query {}", query.name)))?;
    }
    Ok(())
}

/// Property: cascade and parallel conjunct policies are observationally identical.
#[rstest]
fn conjunct_policy_is_not_observable(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
) -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let dtype = v1_dtype(&fixture.layout, &query)?;
        let cascade = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                mode: ConjunctMode::Cascade,
                execution_mode,
                ..Default::default()
            },
        )?;
        let parallel = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                mode: ConjunctMode::Parallel,
                execution_mode,
                ..Default::default()
            },
        )?;
        assert_same_rows(&session, &dtype, &cascade, &parallel)
            .map_err(|err| err.with_context(format!("query {}", query.name)))?;
    }
    Ok(())
}

/// Property: the leased shared cells are an optimisation only. Disabling them must not change
/// a single row, at any thread count — the chaos check for the decode-reuse mechanism.
#[rstest]
fn shared_cells_are_not_observable(#[values(1, 4)] threads: usize) -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let dtype = v1_dtype(&fixture.layout, &query)?;
        let shared = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                threads,
                ..Default::default()
            },
        )?;
        let unshared = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                threads,
                share_decodes: false,
                ..Default::default()
            },
        )?;
        assert_same_rows(&session, &dtype, &shared, &unshared)
            .map_err(|err| err.with_context(format!("query {}", query.name)))?;

        let shared_stats = shared.stats.as_ref().expect("morsel runs report stats");
        let unshared_stats = unshared.stats.as_ref().expect("morsel runs report stats");
        assert_eq!(unshared_stats.decode_reuses, 0);
        assert_eq!(
            shared_stats.decodes + shared_stats.decode_reuses,
            unshared_stats.decodes,
            "query {}: every skipped decode must be accounted for by a reuse",
            query.name
        );
    }
    Ok(())
}

/// Property: on the misaligned fixture, sharing actually fires — a chunk overlapped by several
/// per-split morsels is decoded once and reused for the rest.
#[rstest]
fn shared_cells_reuse_straddled_chunks() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    let query = Query {
        name: "reuse",
        projection: select(vec!["a", "b", "c"], root()),
        filter: None,
    };
    let run = run_morsel(
        &session,
        &fixture.layout,
        &segments,
        &query,
        MorselConfig::default(),
    )?;
    let stats = run.stats.as_ref().expect("morsel runs report stats");
    assert!(
        stats.decode_reuses > 0,
        "expected cross-morsel decode reuse on a misaligned fixture, got none"
    );
    // Each of the 15 chunks (3 + 5 + 7) is decoded exactly once across the whole scan.
    assert_eq!(stats.decodes, 15);
    Ok(())
}

struct CountingSegmentSource {
    inner: Arc<dyn SegmentSource>,
    requests: Arc<AtomicUsize>,
}

struct BackgroundCountingSource {
    inner: Arc<dyn SegmentSource>,
    requests: Arc<AtomicUsize>,
}

impl SegmentSource for BackgroundCountingSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.inner.request(id)
    }

    fn prefers_background_reads(&self) -> bool {
        true
    }
}

struct NowaitSegmentSource {
    buffers: Arc<[ByteBuffer]>,
    attempts: Arc<AtomicUsize>,
    fallbacks: Arc<AtomicUsize>,
    hit: bool,
}

impl SegmentSource for NowaitSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.fallbacks.fetch_add(1, Ordering::Relaxed);
        let buffer = self.buffers.get(*id as usize).cloned();
        async move {
            buffer
                .map(BufferHandle::new_host)
                .ok_or_else(|| vortex_err!("missing segment {id}"))
        }
        .boxed()
    }

    fn request_nowait(&self, id: SegmentId) -> VortexResult<ReadAtNowait> {
        self.attempts.fetch_add(1, Ordering::Relaxed);
        if !self.hit {
            return Ok(ReadAtNowait::WouldBlock);
        }
        self.buffers
            .get(*id as usize)
            .cloned()
            .map(BufferHandle::new_host)
            .map(ReadAtNowait::Ready)
            .ok_or_else(|| vortex_err!("missing segment {id}"))
    }
}

#[rstest]
fn inline_nowait_hit_never_creates_a_background_future(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
) -> VortexResult<()> {
    let session = session();
    let fixture = aligned_fixture(&session, 64)?;
    let attempts = Arc::new(AtomicUsize::new(0));
    let fallbacks = Arc::new(AtomicUsize::new(0));
    let source: Arc<dyn SegmentSource> = Arc::new(NowaitSegmentSource {
        buffers: Arc::from(fixture.segment_buffers.clone()),
        attempts: Arc::clone(&attempts),
        fallbacks: Arc::clone(&fallbacks),
        hit: true,
    });
    let query = Query {
        name: "nowait-hit",
        projection: select(vec!["a"], root()),
        filter: None,
    };
    let v1 = run_v1(&session, &fixture.layout, &fixture.segments, &query)?;
    let morsel = run_morsel(
        &session,
        &fixture.layout,
        &source,
        &query,
        MorselConfig {
            execution_mode,
            ..Default::default()
        },
    )?;

    assert_same_rows(&session, &v1_dtype(&fixture.layout, &query)?, &v1, &morsel)?;
    assert_eq!(attempts.load(Ordering::Relaxed), 1);
    assert_eq!(fallbacks.load(Ordering::Relaxed), 0);
    let stats = morsel.stats.as_ref().expect("morsel runs report stats");
    assert_eq!(stats.nowait_attempts, 1);
    assert_eq!(stats.nowait_hits, 1);
    assert_eq!(stats.nowait_misses, 0);
    assert_eq!(stats.execute_io_blocks, 0);
    assert_eq!(stats.io_waits, 0);
    Ok(())
}

#[rstest]
fn inline_nowait_miss_falls_back_once(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
) -> VortexResult<()> {
    let session = session();
    let fixture = aligned_fixture(&session, 64)?;
    let attempts = Arc::new(AtomicUsize::new(0));
    let fallbacks = Arc::new(AtomicUsize::new(0));
    let source: Arc<dyn SegmentSource> = Arc::new(NowaitSegmentSource {
        buffers: Arc::from(fixture.segment_buffers.clone()),
        attempts: Arc::clone(&attempts),
        fallbacks: Arc::clone(&fallbacks),
        hit: false,
    });
    let query = Query {
        name: "nowait-miss",
        projection: select(vec!["a"], root()),
        filter: None,
    };
    let v1 = run_v1(&session, &fixture.layout, &fixture.segments, &query)?;
    let morsel = run_morsel(
        &session,
        &fixture.layout,
        &source,
        &query,
        MorselConfig {
            execution_mode,
            ..Default::default()
        },
    )?;

    assert_same_rows(&session, &v1_dtype(&fixture.layout, &query)?, &v1, &morsel)?;
    assert_eq!(attempts.load(Ordering::Relaxed), 1);
    assert_eq!(fallbacks.load(Ordering::Relaxed), 1);
    let stats = morsel.stats.as_ref().expect("morsel runs report stats");
    assert_eq!(stats.nowait_attempts, 1);
    assert_eq!(stats.nowait_hits, 0);
    assert_eq!(stats.nowait_misses, 1);
    assert_eq!(stats.nowait_unsupported, 0);
    assert!(stats.execute_io_blocks > 0);
    Ok(())
}

impl SegmentSource for CountingSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.inner.request(id)
    }
}

/// Raw request cells are shared scan-wide even when decoded-array sharing is disabled.
#[rstest]
fn scan_wide_io_cells_deduplicate_straddled_chunks() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let requests = Arc::new(AtomicUsize::new(0));
    let source: Arc<dyn SegmentSource> = Arc::new(CountingSegmentSource {
        inner: Arc::clone(&fixture.segments),
        requests: Arc::clone(&requests),
    });
    let query = Query {
        name: "scan-wide-io",
        projection: select(vec!["a", "b", "c"], root()),
        filter: None,
    };

    let run = run_morsel(
        &session,
        &fixture.layout,
        &source,
        &query,
        MorselConfig {
            threads: 4,
            share_decodes: false,
            ..Default::default()
        },
    )?;
    let stats = run.stats.as_ref().expect("morsel runs report stats");

    assert_eq!(requests.load(Ordering::Relaxed), 15);
    assert_eq!(stats.io_requests, 15);
    assert!(stats.io_uses > stats.io_requests);
    Ok(())
}

#[rstest]
fn filtered_lookahead_refills_from_retired_frontier() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let requests = Arc::new(AtomicUsize::new(0));
    let source: Arc<dyn SegmentSource> = Arc::new(BackgroundCountingSource {
        inner: Arc::clone(&fixture.segments),
        requests: Arc::clone(&requests),
    });
    let query = Query {
        name: "sliding-lookahead",
        projection: select(vec!["a", "c"], root()),
        filter: Some(gt(get_item("a", root()), lit(400i32))),
    };
    let run = run_morsel(
        &session,
        &fixture.layout,
        &source,
        &query,
        MorselConfig {
            threads: 1,
            execution_mode: ExecutionMode::Push,
            lookahead_morsels: 1,
            ..Default::default()
        },
    )?;
    let stats = run.stats.as_ref().expect("morsel runs report stats");
    assert!(stats.lookahead_refills > 0);
    assert!(stats.demand_io_promotions > 0);
    assert_eq!(
        stats.io_requests,
        u64::try_from(requests.load(Ordering::Relaxed)).unwrap_or(u64::MAX)
    );
    assert!(requests.load(Ordering::Relaxed) > 0);
    Ok(())
}

/// Property: every read a node waits on was named by its own planning stream, so the number of
/// distinct segments read never exceeds the number of uses named.
#[rstest]
fn every_read_was_planned() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let run = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig::default(),
        )?;
        let stats = run.stats.as_ref().expect("morsel runs report stats");
        assert!(
            stats.io_requests <= stats.io_uses,
            "query {}: {} requests exceeds {} named uses",
            query.name,
            stats.io_requests,
            stats.io_uses
        );
    }
    Ok(())
}

/// Property: an all-false filter emits nothing and does not decode its projection columns.
#[rstest]
fn empty_filter_emits_nothing() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    let query = Query {
        name: "empty",
        projection: select(vec!["a", "b", "c"], root()),
        filter: Some(gt(get_item("a", root()), lit(i32::MAX - 1))),
    };
    let run = run_morsel(
        &session,
        &fixture.layout,
        &segments,
        &query,
        MorselConfig::default(),
    )?;
    assert_eq!(run.rows, 0);
    assert!(run.batches.is_empty());
    let stats = run.stats.as_ref().expect("morsel runs report stats");
    assert_eq!(stats.morsels_empty, stats.morsels);
    Ok(())
}

#[derive(Default)]
struct PairedGate {
    polled: [bool; 2],
    wakers: [Option<Waker>; 2],
    watchdog_fired: bool,
}

struct PairedPendingSource {
    buffers: Arc<[ByteBuffer]>,
    gate: Arc<Mutex<PairedGate>>,
}

impl SegmentSource for PairedPendingSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let index = *id as usize;
        let buffer = self.buffers.get(index).cloned();
        let gate = Arc::clone(&self.gate);
        poll_fn(move |cx| {
            let Some(buffer) = buffer.as_ref() else {
                return Poll::Ready(Err(vortex_error::vortex_err!(
                    "missing gated segment {index}"
                )));
            };
            if index >= 2 {
                return Poll::Ready(Ok(BufferHandle::new_host(buffer.clone())));
            }

            let other = 1 - index;
            let mut gate = gate.lock();
            gate.polled[index] = true;
            if gate.polled[other] {
                if let Some(waker) = gate.wakers[other].take() {
                    waker.wake();
                }
                Poll::Ready(Ok(BufferHandle::new_host(buffer.clone())))
            } else {
                gate.wakers[index] = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .boxed()
    }
}

/// One CPU worker must submit every planned read before waiting for either one. Each of this
/// source's first two futures remains pending until the other has been polled, so the old inline
/// `block_on` driver reaches the watchdog while the continuation scheduler completes immediately.
#[rstest]
fn planned_reads_progress_together_without_parking_a_worker(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
) -> VortexResult<()> {
    let session = session();
    let values: Vec<i32> = (0..32).collect();
    let fixture = block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&values, &[32])),
                Column::new("b", i32_chunks(&values, &[32])),
            ],
            &session,
        )
        .await
    })?;

    let gate = Arc::new(Mutex::new(PairedGate::default()));
    let watchdog_gate = Arc::clone(&gate);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(1));
        let mut gate = watchdog_gate.lock();
        if gate.polled.iter().all(|polled| *polled) {
            return;
        }
        gate.watchdog_fired = true;
        gate.polled = [true; 2];
        for waker in gate.wakers.iter_mut().filter_map(Option::take) {
            waker.wake();
        }
    });

    let source: Arc<dyn SegmentSource> = Arc::new(PairedPendingSource {
        buffers: Arc::from(fixture.segment_buffers.clone()),
        gate: Arc::clone(&gate),
    });
    let query = Query {
        name: "paired-pending",
        projection: select(vec!["b"], root()),
        filter: Some(gt(get_item("a", root()), lit(-1i32))),
    };
    let v1 = run_v1(&session, &fixture.layout, &fixture.segments, &query)?;
    let morsel = run_morsel(
        &session,
        &fixture.layout,
        &source,
        &query,
        MorselConfig {
            threads: 1,
            execution_mode,
            ..Default::default()
        },
    )?;

    assert_same_rows(&session, &v1_dtype(&fixture.layout, &query)?, &v1, &morsel)?;
    let gate = gate.lock();
    assert_eq!(gate.polled, [true; 2]);
    assert!(!gate.watchdog_fired, "the CPU worker parked on one read");
    if execution_mode == ExecutionMode::Push {
        let stats = morsel.stats.as_ref().expect("morsel runs report stats");
        assert!(stats.execute_io_blocks > 0);
        assert_eq!(stats.push_stale_wakes, 0);
        assert!(stats.push_inline_transfers > 0);
        assert!(stats.push_pipeline_runs > 0);
        assert!(stats.push_pipeline_stage_calls > 0);
        assert!(stats.push_pipeline_boundary_resumes >= 2);
    }
    Ok(())
}

#[derive(Default)]
struct BurstGate {
    requests: [usize; 3],
    polls: [usize; 3],
    wakers: [Option<Waker>; 3],
    released: bool,
    watchdog_fired: bool,
}

struct BurstPendingSource {
    buffers: Arc<[ByteBuffer]>,
    gate: Arc<Mutex<BurstGate>>,
}

impl SegmentSource for BurstPendingSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let index = *id as usize;
        let buffer = self.buffers.get(index).cloned();
        if index < 3 {
            self.gate.lock().requests[index] += 1;
        }
        let gate = Arc::clone(&self.gate);
        poll_fn(move |cx| {
            let Some(buffer) = buffer.as_ref() else {
                return Poll::Ready(Err(vortex_error::vortex_err!(
                    "missing burst segment {index}"
                )));
            };
            if index >= 3 {
                return Poll::Ready(Ok(BufferHandle::new_host(buffer.clone())));
            }

            let wakes = {
                let mut gate = gate.lock();
                gate.polls[index] += 1;
                if gate.released {
                    return Poll::Ready(Ok(BufferHandle::new_host(buffer.clone())));
                }
                gate.wakers[index] = Some(cx.waker().clone());
                if gate.polls.iter().all(|polls| *polls > 0) {
                    gate.released = true;
                    gate.wakers.iter_mut().filter_map(Option::take).collect()
                } else {
                    Vec::new()
                }
            };
            for waker in wakes {
                waker.wake_by_ref();
                waker.wake_by_ref();
            }
            Poll::Pending
        })
        .boxed()
    }
}

/// Burst wakeups for several exact cells neither lose a wake nor poll a ready cell again from
/// execution.
#[rstest]
fn burst_wakes_are_coalesced_without_duplicate_polls(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
) -> VortexResult<()> {
    let session = session();
    let values: Vec<i32> = (0..32).collect();
    let fixture = block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&values, &[32])),
                Column::new("b", i32_chunks(&values, &[32])),
                Column::new("c", i32_chunks(&values, &[32])),
            ],
            &session,
        )
        .await
    })?;

    let gate = Arc::new(Mutex::new(BurstGate::default()));
    let watchdog_gate = Arc::clone(&gate);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(1));
        let wakes = {
            let mut gate = watchdog_gate.lock();
            if gate.released {
                return;
            }
            gate.watchdog_fired = true;
            gate.released = true;
            gate.wakers
                .iter_mut()
                .filter_map(Option::take)
                .collect::<Vec<_>>()
        };
        for waker in wakes {
            waker.wake();
        }
    });

    let source: Arc<dyn SegmentSource> = Arc::new(BurstPendingSource {
        buffers: Arc::from(fixture.segment_buffers.clone()),
        gate: Arc::clone(&gate),
    });
    let query = Query {
        name: "burst-pending",
        projection: select(vec!["a", "b", "c"], root()),
        filter: None,
    };
    let v1 = run_v1(&session, &fixture.layout, &fixture.segments, &query)?;
    let morsel = run_morsel(
        &session,
        &fixture.layout,
        &source,
        &query,
        MorselConfig {
            threads: 1,
            execution_mode,
            ..Default::default()
        },
    )?;

    assert_same_rows(&session, &v1_dtype(&fixture.layout, &query)?, &v1, &morsel)?;
    let gate = gate.lock();
    assert_eq!(gate.requests, [1, 1, 1]);
    assert!(gate.polls.iter().all(|polls| (1..=2).contains(polls)));
    assert!(gate.polls.contains(&2));
    if execution_mode == ExecutionMode::Push {
        assert_eq!(gate.polls, [2, 2, 2]);
    }
    assert!(!gate.watchdog_fired);
    let stats = morsel.stats.as_ref().expect("morsel runs report stats");
    assert_eq!(stats.io_requests, 3);
    assert_eq!(stats.io_batches, 3);
    assert_eq!(stats.io_waits, 3);
    assert_eq!(stats.morsels_blocked_for_io, 1);
    assert!(stats.execute_io_blocks > 0);
    assert!(stats.io_blocks_per_morsel_max <= 3);
    if execution_mode == ExecutionMode::Push {
        assert_eq!(stats.push_stale_wakes, 0);
        assert!(stats.push_pipeline_runs > 0);
        assert!(stats.push_pipeline_stage_calls > 0);
        assert!(stats.push_pipeline_boundary_resumes >= 3);
    }
    Ok(())
}

#[derive(Default)]
struct SpeculativeGate {
    polls: [usize; 2],
    projection_waker: Option<Waker>,
    released: bool,
    watchdog_fired: bool,
}

struct SlowSpeculativeSource {
    buffers: Arc<[ByteBuffer]>,
    gate: Arc<Mutex<SpeculativeGate>>,
}

struct FailingProjectionSource {
    buffers: Arc<[ByteBuffer]>,
    projection_polls: Arc<AtomicUsize>,
}

impl SegmentSource for FailingProjectionSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let index = *id as usize;
        if index == 1 {
            self.projection_polls.fetch_add(1, Ordering::Relaxed);
            return async move { Err(vortex_err!("injected projection read failure")) }.boxed();
        }
        let buffer = self.buffers.get(index).cloned();
        async move {
            buffer
                .map(BufferHandle::new_host)
                .ok_or_else(|| vortex_err!("missing segment {index}"))
        }
        .boxed()
    }

    fn prefers_background_reads(&self) -> bool {
        true
    }
}

impl SegmentSource for SlowSpeculativeSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let index = *id as usize;
        let buffer = self.buffers.get(index).cloned();
        let gate = Arc::clone(&self.gate);
        poll_fn(move |cx| {
            let Some(buffer) = buffer.as_ref() else {
                return Poll::Ready(Err(vortex_error::vortex_err!(
                    "missing speculative segment {index}"
                )));
            };
            if index >= 2 {
                return Poll::Ready(Ok(BufferHandle::new_host(buffer.clone())));
            }
            let mut gate = gate.lock();
            gate.polls[index] += 1;
            if index == 0 || gate.released {
                Poll::Ready(Ok(BufferHandle::new_host(buffer.clone())))
            } else {
                gate.projection_waker = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .boxed()
    }
}

/// Required predicate IO resumes execution while speculative projection IO remains pending. An
/// empty predicate result retires the morsel without waiting for or consuming that projection.
#[rstest]
fn empty_filter_cancels_pending_speculative_io(
    #[values(ExecutionMode::Pull, ExecutionMode::Push)] execution_mode: ExecutionMode,
) -> VortexResult<()> {
    let session = session();
    let values: Vec<i32> = (0..32).collect();
    let fixture = block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&values, &[32])),
                Column::new("b", i32_chunks(&values, &[32])),
            ],
            &session,
        )
        .await
    })?;

    let gate = Arc::new(Mutex::new(SpeculativeGate::default()));
    let watchdog_gate = Arc::clone(&gate);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(1));
        let wake = {
            let mut gate = watchdog_gate.lock();
            if gate.released {
                return;
            }
            gate.watchdog_fired = true;
            gate.released = true;
            gate.projection_waker.take()
        };
        if let Some(waker) = wake {
            waker.wake();
        }
    });

    let source: Arc<dyn SegmentSource> = Arc::new(SlowSpeculativeSource {
        buffers: Arc::from(fixture.segment_buffers.clone()),
        gate: Arc::clone(&gate),
    });
    let query = Query {
        name: "cancel-speculative",
        projection: select(vec!["b"], root()),
        filter: Some(gt(get_item("a", root()), lit(i32::MAX - 1))),
    };
    let v1 = run_v1(&session, &fixture.layout, &fixture.segments, &query)?;
    let morsel = run_morsel(
        &session,
        &fixture.layout,
        &source,
        &query,
        MorselConfig {
            threads: 1,
            execution_mode,
            ..Default::default()
        },
    )?;

    assert_same_rows(&session, &v1_dtype(&fixture.layout, &query)?, &v1, &morsel)?;
    let gate = gate.lock();
    assert_eq!(gate.polls, [1, 1]);
    assert!(!gate.watchdog_fired, "execution waited for speculative IO");
    let stats = morsel.stats.as_ref().expect("morsel runs report stats");
    assert!(stats.io_blocks_per_morsel_max <= 1);
    Ok(())
}

#[rstest]
fn speculative_projection_errors_are_authoritative_only(
    #[values(
        DemandHintDelivery::Immediate,
        DemandHintDelivery::Disabled,
        DemandHintDelivery::Delayed(usize::MAX)
    )]
    demand_hints: DemandHintDelivery,
) -> VortexResult<()> {
    let session = session();
    let values: Vec<i32> = (0..32).collect();
    let fixture = block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&values, &[32])),
                Column::new("b", i32_chunks(&values, &[32])),
            ],
            &session,
        )
        .await
    })?;
    let projection_polls = Arc::new(AtomicUsize::new(0));
    let source: Arc<dyn SegmentSource> = Arc::new(FailingProjectionSource {
        buffers: Arc::from(fixture.segment_buffers.clone()),
        projection_polls: Arc::clone(&projection_polls),
    });
    let empty = Query {
        name: "unused-failing-projection",
        projection: select(vec!["b"], root()),
        filter: Some(gt(get_item("a", root()), lit(i32::MAX - 1))),
    };
    let v1 = run_v1(&session, &fixture.layout, &fixture.segments, &empty)?;
    let morsel = run_morsel(
        &session,
        &fixture.layout,
        &source,
        &empty,
        MorselConfig {
            execution_mode: ExecutionMode::Push,
            demand_hints,
            ..Default::default()
        },
    )?;
    assert_same_rows(&session, &v1_dtype(&fixture.layout, &empty)?, &v1, &morsel)?;
    let stats = morsel.stats.as_ref().expect("morsel runs report stats");
    if demand_hints == DemandHintDelivery::Immediate {
        assert!(stats.demand_io_suppressed > 0);
        assert!(stats.demand_io_candidates > 0);
        assert!(stats.demand_io_candidates <= stats.demand_hints_observed);
    }
    assert_eq!(stats.io_requests, 1);
    assert_eq!(projection_polls.load(Ordering::Relaxed), 0);

    let selected = Query {
        name: "used-failing-projection",
        projection: select(vec!["b"], root()),
        filter: Some(gt(get_item("a", root()), lit(-1_i32))),
    };
    let error = run_morsel(
        &session,
        &fixture.layout,
        &source,
        &selected,
        MorselConfig {
            execution_mode: ExecutionMode::Push,
            demand_hints,
            ..Default::default()
        },
    )
    .err()
    .ok_or_else(|| vortex_err!("an authoritative projection read must surface its error"))?;
    assert!(format!("{error}").contains("injected projection read failure"));
    assert!(projection_polls.load(Ordering::Relaxed) > 0);
    Ok(())
}

/// Unsupported shapes are build errors, never silent fallbacks.
#[rstest]
fn rejects_unsupported_layouts() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, 32)?;
    // A non-struct root: take a column's chunked layout directly.
    let column = fixture
        .layout
        .slot(1)?
        .expect("the fixture root has a first field");
    let err = crate::build_plan(
        &column,
        &select(vec!["a"], root()),
        None,
        ConjunctMode::Cascade,
    )
    .err()
    .expect("a chunked root must be rejected");
    assert!(
        format!("{err}").contains("struct"),
        "unexpected error: {err}"
    );
    Ok(())
}

fn v1_dtype(layout: &LayoutRef, query: &Query) -> VortexResult<DType> {
    Ok(query.projection.bind(layout.dtype())?.dtype().clone())
}

/// A guard against the fixtures silently degenerating into a single chunk per column.
#[rstest]
fn fixture_is_actually_misaligned() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let plan = crate::build_plan(
        &fixture.layout,
        &select(vec!["a", "b", "c"], root()),
        None,
        ConjunctMode::Cascade,
    )?;
    // Three columns cut into 3, 5 and 7 chunks share only the final boundary.
    assert!(
        plan.natural_splits().len() > 7,
        "expected the union of three chunkings, got {:?}",
        plan.natural_splits()
    );
    Ok(())
}
