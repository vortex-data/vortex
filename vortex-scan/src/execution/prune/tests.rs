// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_expr::Expression;
use vortex_expr::and;
use vortex_expr::get_item;
use vortex_expr::gt;
use vortex_expr::lit;
use vortex_expr::lt;
use vortex_expr::or;
use vortex_expr::root;
use vortex_io::request::IoIntent;
use vortex_mask::Mask;

use super::*;
use crate::execution::ChunkPlanner;
use crate::plan::ConcatPlan;
use crate::plan::ZonedPlan;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::prepare::ScanBuilder;
use crate::test::SESSION;
use crate::test::Store;
use crate::test::struct_plan;

const ZONE_ROWS: i32 = 4;
const ZONES: i32 = 4;
const TOTAL_ROWS: u64 = 16;

/// Builds a 16-row table `{a: i32, b: i32}` with 4 zones of 4 rows each:
/// - `a`: zones `[0..3], [4..7], [8..11], [12..15]` (optionally with `null` stats for `null_zone`)
/// - `b`: zones `[12..15], [8..11], [4..7], [0..3]` (reverse order)
fn zoned_two_column_scan(
    filter: Option<Expression>,
    with_zones: bool,
    a_null_zone: Option<i32>,
) -> VortexResult<(Arc<PreparedScan>, Arc<Store>)> {
    let store = Arc::new(Store::default());
    let i32_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let min_fn = Min.bind(NumericalAggregateOpts::skip_nans());
    let max_fn = Max.bind(NumericalAggregateOpts::skip_nans());

    let make_col = |zone_order: &[i32],
                    bounds: &[(Option<i32>, Option<i32>)]|
     -> VortexResult<PlanRef> {
        let chunks = zone_order
            .iter()
            .map(|&z| {
                let start = z * ZONE_ROWS;
                store.flat(&PrimitiveArray::from_iter(start..start + ZONE_ROWS).into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let data = ConcatPlan::try_new(i32_dtype.clone(), chunks)?.into_plan();
        if !with_zones {
            return Ok(data);
        }
        let mins = PrimitiveArray::from_option_iter(bounds.iter().map(|(lo, _)| *lo)).into_array();
        let maxs = PrimitiveArray::from_option_iter(bounds.iter().map(|(_, hi)| *hi)).into_array();
        let table = StructArray::from_fields(&[
            (min_fn.to_string().as_str(), mins),
            (max_fn.to_string().as_str(), maxs),
        ])?
        .into_array();
        let zones = store.flat(&table)?;
        Ok(ZonedPlan::new(
            data,
            zones,
            u64::try_from(ZONE_ROWS)?,
            vec![min_fn.clone(), max_fn.clone()].into(),
        )
        .into_plan())
    };

    let a_zones: Vec<i32> = (0..ZONES).collect();
    let b_zones: Vec<i32> = (0..ZONES).rev().collect();
    let a_bounds: Vec<_> = a_zones
        .iter()
        .map(|&z| {
            if a_null_zone == Some(z) {
                (None, None)
            } else {
                let lo = z * ZONE_ROWS;
                (Some(lo), Some(lo + ZONE_ROWS - 1))
            }
        })
        .collect();
    let b_bounds: Vec<_> = b_zones
        .iter()
        .map(|&z| {
            let lo = z * ZONE_ROWS;
            (Some(lo), Some(lo + ZONE_ROWS - 1))
        })
        .collect();

    let dtype = DType::Struct(
        StructFields::from_iter([("a", i32_dtype.clone()), ("b", i32_dtype.clone())]),
        Nullability::NonNullable,
    );
    let plan = struct_plan(
        TOTAL_ROWS,
        dtype.clone(),
        vec![
            make_col(&a_zones, &a_bounds)?,
            make_col(&b_zones, &b_bounds)?,
        ],
    );
    let mut builder = ScanBuilder::new(SESSION.clone(), store.input(plan));
    if let Some(expr) = filter {
        builder = builder.with_some_filter(Some(expr.bind(&dtype)?));
    }
    Ok((Arc::new(builder.prepare()?), store))
}

struct ObservePrune {
    inner: PrunePlanner,
    remaining_plans: Arc<AtomicUsize>,
}

impl Drop for ObservePrune {
    fn drop(&mut self) {
        self.remaining_plans
            .store(self.inner.pruning.len(), Ordering::SeqCst);
    }
}

impl IoConsumer for ObservePrune {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        self.inner.set_io_result(request, result);
    }
}

impl Planner for ObservePrune {
    fn state(&self) -> State {
        self.inner.state()
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        self.inner.compute()
    }
}

async fn run_prune(
    scan: &Arc<PreparedScan>,
    rows: Range<u64>,
) -> VortexResult<(bool, Option<Mask>, usize)> {
    let called = Arc::new(AtomicBool::new(false));
    let kept = Arc::new(Mutex::new(None));
    let remaining_plans = Arc::new(AtomicUsize::new(usize::MAX));
    let next: PruneNext = {
        let scan = Arc::clone(scan);
        let called = Arc::clone(&called);
        let kept = Arc::clone(&kept);
        Box::new(move |mask| {
            called.store(true, Ordering::SeqCst);
            *kept.lock() = mask;
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, 0..0, selection, None))
        })
    };
    let io = scan.io();
    let mut run = Run::new();
    run.admit(
        Box::new(ObservePrune {
            inner: PrunePlanner::new(Arc::clone(scan), rows.clone(), next),
            remaining_plans: Arc::clone(&remaining_plans),
        }),
        WorkScope {
            file_ordinal: 0,
            rows,
        },
        io.session(),
    );
    loop {
        match run.advance()? {
            Progress::Batch(_) => vortex_bail!("pruning alone produced a batch"),
            Progress::Spawn(scope, planner) => {
                run.admit(planner, scope, io.session());
            }
            Progress::RootDone(_) if run.live_roots() > 0 => {}
            Progress::RootDone(_) | Progress::Idle => {
                drop(run);
                return Ok((
                    called.load(Ordering::SeqCst),
                    kept.lock().take(),
                    remaining_plans.load(Ordering::SeqCst),
                ));
            }
            Progress::Waiting => std::future::poll_fn(|cx| run.poll_completion(cx)).await?,
        }
    }
}

#[test]
fn unprunable_scan_emits_next_immediately_with_none_and_no_io() -> VortexResult<()> {
    let (scan, store) =
        zoned_two_column_scan(Some(gt(get_item("a", root()), lit(5i32))), false, None)?;
    let kept = Arc::new(Mutex::new(Some(Mask::new_true(0))));
    let next: PruneNext = {
        let scan = Arc::clone(&scan);
        let kept = Arc::clone(&kept);
        Box::new(move |mask| {
            *kept.lock() = mask;
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, 0..0, selection, None))
        })
    };
    let mut planner = PrunePlanner::new(Arc::clone(&scan), 0..TOTAL_ROWS, next);
    assert_eq!(planner.state(), State::NeedsCompute);

    let PlannerOutput::Planner(scope, _) = planner.compute()? else {
        vortex_bail!("unprunable filter must continue immediately to next planner");
    };
    assert_eq!(scope.rows, 0..TOTAL_ROWS);
    assert!(kept.lock().is_none());
    assert_eq!(planner.state(), State::Done);
    assert!(matches!(planner.compute()?, PlannerOutput::Done));
    assert_eq!(store.requests(), 0);
    Ok(())
}

#[tokio::test]
async fn all_pruned_first_conjunct_short_circuits_without_calling_next() -> VortexResult<()> {
    // First conjunct `a > 100` excludes every zone; second conjunct `b > 0` must never run.
    let filter = and(
        gt(get_item("a", root()), lit(100i32)),
        gt(get_item("b", root()), lit(0i32)),
    );
    let (scan, store) = zoned_two_column_scan(Some(filter), true, None)?;

    let (called, kept, remaining) = run_prune(&scan, 0..TOTAL_ROWS).await?;
    assert!(
        !called,
        "next planner must not run when all rows are pruned"
    );
    assert!(kept.is_none());
    // The second conjunct (`b > 0`) was never popped once the first pruned every row.
    assert_eq!(remaining, 1);
    // Prefetch hints start both zone maps (2 segments), and no data segments are ever read.
    assert_eq!(store.requests(), 2);
    Ok(())
}

#[test]
fn first_compute_publishes_prefetch_hints_for_all_zone_maps() -> VortexResult<()> {
    let filter = and(
        gt(get_item("a", root()), lit(3i32)),
        gt(get_item("b", root()), lit(3i32)),
    );
    let (scan, _) = zoned_two_column_scan(Some(filter), true, None)?;
    let next: PruneNext = Box::new(|_| unreachable!("not reached before IO"));
    let mut planner = PrunePlanner::new(Arc::clone(&scan), 0..TOTAL_ROWS, next);

    let PlannerOutput::NeedsIO(batch) = planner.compute()? else {
        vortex_bail!("expected prefetch hints for both columns' zone maps");
    };
    assert_eq!(batch.len(), 2);
    assert!(batch.iter().all(|req| req.intent == IoIntent::Prefetch));
    assert_ne!(batch[0].request, batch[1].request);
    assert_ne!(batch[0].target, batch[1].target);
    Ok(())
}

#[tokio::test]
async fn conjunction_intersects_pruning_masks_across_columns() -> VortexResult<()> {
    // `a > 3` keeps zones 1,2,3 (rows 4..16); `b > 3` keeps zones 0,1,2 (rows 0..12).
    // Together only zones 1 and 2 (rows 4..12) survive.
    let filter = and(
        gt(get_item("a", root()), lit(3i32)),
        gt(get_item("b", root()), lit(3i32)),
    );
    let (scan, _) = zoned_two_column_scan(Some(filter), true, None)?;

    let (called, kept, remaining) = run_prune(&scan, 0..TOTAL_ROWS).await?;
    assert!(called);
    assert_eq!(remaining, 0);
    let expected = Mask::from_iter((0..16).map(|row| (4..12).contains(&row)));
    assert_eq!(kept, Some(expected));
    Ok(())
}

#[tokio::test]
async fn disjunction_prunes_only_zones_falsified_by_both_branches() -> VortexResult<()> {
    // `a < 4 OR a > 11`: zones 1 (4..7) and 2 (8..11) are excluded; zones 0 and 3 survive.
    let filter = or(
        lt(get_item("a", root()), lit(4i32)),
        gt(get_item("a", root()), lit(11i32)),
    );
    let (scan, _) = zoned_two_column_scan(Some(filter), true, None)?;

    let (called, kept, remaining) = run_prune(&scan, 0..TOTAL_ROWS).await?;
    assert!(called);
    assert_eq!(remaining, 0);
    let expected = Mask::from_iter((0..16).map(|row| !(4..12).contains(&row)));
    assert_eq!(kept, Some(expected));
    Ok(())
}

#[tokio::test]
async fn null_zone_stats_conservatively_keep_zone_and_subrange_clips_mask() -> VortexResult<()> {
    // Zone 0 has unknown (null) min/max stats so `a > 11` must keep zone 0 and zone 3,
    // while excluding zones 1 (4..7) and 2 (8..11).
    let filter = gt(get_item("a", root()), lit(11i32));
    let (scan, _) = zoned_two_column_scan(Some(filter), true, Some(0))?;

    // Run over a subrange 2..14 spanning half of zone 0, zones 1..2, and half of zone 3.
    let (called, kept, remaining) = run_prune(&scan, 2..14).await?;
    assert!(called);
    assert_eq!(remaining, 0);
    // Rows 2..4 (zone 0, null stats -> kept), 4..12 (zones 1..2 -> pruned), 12..14 (zone 3 -> kept).
    let expected = Mask::from_iter((2..14).map(|row| !(4..12).contains(&row)));
    assert_eq!(kept, Some(expected));
    Ok(())
}
