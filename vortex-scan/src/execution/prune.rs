// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Evaluates a filter's pruning plans once over a scan's rows, before any chunk is planned.

use std::ops::Range;
use std::sync::Arc;
use std::vec::IntoIter;

use rustc_hash::FxHashSet;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoIntent;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;

use super::PreparedScan;
use super::adaptor::AdaptorStep;
use super::adaptor::ScanAdaptor;
use super::filter::hint;
use super::filter::piece_mask;
use super::filter::segments_read;
use crate::execution::pipeline::Scan;
use crate::execution::pipeline::Split;
use crate::plan::PlanRef;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;

/// The planner a prune planner continues into, given the rows pruning kept, or `None` when
/// no pruning plan ran.
pub type PruneNext = Box<dyn FnOnce(Option<Mask>) -> Box<dyn Planner> + Send>;

/// Evaluates the filter's pruning plans over the scan's rows and emits the planner it was
/// built to continue into, with the rows no plan pruned. The zone maps are the only reads,
/// started together. Rows every plan prunes end the scan here: nothing is emitted. A filter
/// without pruning plans emits the next planner at once.
pub struct PrunePlanner {
    scan: Arc<PreparedScan>,
    rows: Range<u64>,
    /// What to emit once every pruning plan has run, taken when it is.
    next: Option<PruneNext>,
    /// Whether any pruning plan ran, so the kept rows mean something.
    pruned: bool,
    /// The rows no pruning plan has excluded yet, over `rows`.
    mask: Mask,
    pruning: IntoIter<PlanRef>,
    active: Option<ScanAdaptor>,
    pieces: Vec<Mask>,
    announced: bool,
    /// Ids for hints, counting down from `u32::MAX` so they never meet an adaptor's fetch ids.
    next_hint: u32,
    done: bool,
}

impl PrunePlanner {
    /// The planner of `scan`'s filter's pruning plans over `rows`, continuing into `next`.
    pub fn new(scan: Arc<PreparedScan>, rows: Range<u64>, next: PruneNext) -> Self {
        let pruning: Vec<_> = scan
            .filter
            .iter()
            .flat_map(|filter| filter.pruning_over(rows.end - rows.start).cloned())
            .collect();
        let len = usize::try_from(rows.end - rows.start).unwrap_or(usize::MAX);
        Self {
            scan,
            rows,
            next: Some(next),
            pruned: false,
            mask: Mask::new_true(len),
            pruning: pruning.into_iter(),
            active: None,
            pieces: Vec::new(),
            announced: false,
            next_hint: u32::MAX,
            done: false,
        }
    }
}

impl IoConsumer for PrunePlanner {
    fn set_io_result(&mut self, request: IoRequestId, result: IoResult) {
        if let Some(adaptor) = &mut self.active {
            adaptor.set_io_result(request, result);
        }
    }
}

impl Planner for PrunePlanner {
    fn state(&self) -> State {
        if self.done {
            return State::Done;
        }
        match &self.active {
            Some(adaptor) if adaptor.state() != State::Done => adaptor.state(),
            _ => State::NeedsCompute,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if self.done {
            return Ok(PlannerOutput::Done);
        }
        if !self.announced {
            self.announced = true;
            // A pruning plan whose zones an earlier scan of the file pruned compiles to its kept
            // result and lists no segment, so only unpruned plans prefetch their zone maps.
            let segments = segments_read(
                &self.scan.session,
                self.pruning.as_slice().iter(),
                self.rows.clone(),
                &self.mask,
            )?;
            let mut batch = Vec::new();
            let mut seen = FxHashSet::default();
            hint(
                &self.scan.locations,
                &mut self.next_hint,
                IoIntent::Prefetch,
                &segments,
                &mut seen,
                &mut batch,
            );
            if !batch.is_empty() {
                return Ok(PlannerOutput::NeedsIO(batch));
            }
        }
        if let Some(adaptor) = &mut self.active {
            if adaptor.state() != State::Done {
                return Ok(match adaptor.compute()? {
                    AdaptorStep::Yield => PlannerOutput::Continue,
                    AdaptorStep::NeedsIO(batch) => PlannerOutput::NeedsIO(batch),
                    AdaptorStep::Piece(piece) => {
                        self.pieces.push(piece_mask(&self.scan.session, piece)?);
                        PlannerOutput::Continue
                    }
                });
            }
            let input = self.mask.true_count();
            let excluded = Mask::concat(self.pieces.iter())?;
            vortex_ensure_eq!(excluded.len(), input);
            self.mask = self.mask.intersect_by_rank(&!excluded);
            tracing::debug!(
                row_start = self.rows.start,
                row_end = self.rows.end,
                input_rows = input,
                output_rows = self.mask.true_count(),
                "v2 pruning plan evaluated"
            );
            self.active = None;
            self.pieces.clear();
            self.pruned = true;
        }
        if self.mask.all_false() {
            self.done = true;
            return Ok(PlannerOutput::Done);
        }
        if let Some(plan) = self.pruning.next() {
            let scan = Scan::try_new(
                self.scan.session.clone(),
                plan,
                vec![Split {
                    rows: self.rows.clone(),
                    mask: self.mask.clone(),
                }],
            )?
            .with_row_offset(self.scan.row_offset)
            .with_value_cache(self.scan.values.clone());
            self.active = Some(ScanAdaptor::new(scan, Arc::clone(&self.scan.locations)));
            return Ok(PlannerOutput::Continue);
        }
        // Every pruning plan has run: the planner this one continues into gets the kept rows.
        self.done = true;
        tracing::debug!(
            row_start = self.rows.start,
            row_end = self.rows.end,
            kept_rows = self.mask.true_count(),
            "v2 pruning finished"
        );
        let next = self
            .next
            .take()
            .ok_or_else(|| vortex_err!("the prune planner has nothing to continue into"))?;
        let kept = self
            .pruned
            .then(|| std::mem::replace(&mut self.mask, Mask::new_true(0)));
        Ok(PlannerOutput::Planner(
            WorkScope {
                file_ordinal: 0,
                rows: self.rows.clone(),
            },
            next(kept),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
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

        let make_col = |bounds: &[(Option<i32>, Option<i32>)]| -> VortexResult<PlanRef> {
            let chunks = (0..ZONES)
                .map(|z| {
                    let start = z * ZONE_ROWS;
                    store.flat(&PrimitiveArray::from_iter(start..start + ZONE_ROWS).into_array())
                })
                .collect::<VortexResult<Vec<_>>>()?;
            let data = ConcatPlan::try_new(i32_dtype.clone(), chunks)?.into_plan();
            if !with_zones {
                return Ok(data);
            }
            let mins =
                PrimitiveArray::from_option_iter(bounds.iter().map(|(lo, _)| *lo)).into_array();
            let maxs =
                PrimitiveArray::from_option_iter(bounds.iter().map(|(_, hi)| *hi)).into_array();
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

        let a_bounds: Vec<_> = (0..ZONES)
            .map(|z| {
                if a_null_zone == Some(z) {
                    (None, None)
                } else {
                    let lo = z * ZONE_ROWS;
                    (Some(lo), Some(lo + ZONE_ROWS - 1))
                }
            })
            .collect();
        let b_bounds: Vec<_> = (0..ZONES)
            .rev()
            .map(|z| {
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
            vec![make_col(&a_bounds)?, make_col(&b_bounds)?],
        );
        let mut builder = ScanBuilder::new(SESSION.clone(), store.input(plan));
        if let Some(expr) = filter {
            builder = builder.with_some_filter(Some(expr.bind(&dtype)?));
        }
        Ok((Arc::new(builder.prepare()?), store))
    }

    async fn run_prune(
        scan: &Arc<PreparedScan>,
        rows: Range<u64>,
    ) -> VortexResult<(bool, Option<Mask>)> {
        let called = Arc::new(AtomicBool::new(false));
        let kept = Arc::new(Mutex::new(None));
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
            Box::new(PrunePlanner::new(Arc::clone(scan), rows.clone(), next)),
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
                    return Ok((called.load(Ordering::SeqCst), kept.lock().take()));
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

        let (called, kept) = run_prune(&scan, 0..TOTAL_ROWS).await?;
        assert!(
            !called,
            "next planner must not run when all rows are pruned"
        );
        assert!(kept.is_none());
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
        assert_eq!(batch[0].request, IoRequestId(u32::MAX));
        assert_eq!(batch[1].request, IoRequestId(u32::MAX - 1));
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

        let (called, kept) = run_prune(&scan, 0..TOTAL_ROWS).await?;
        assert!(called);
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

        let (called, kept) = run_prune(&scan, 0..TOTAL_ROWS).await?;
        assert!(called);
        let expected = Mask::from_iter((0..16).map(|row| !(4..12).contains(&row)));
        assert_eq!(kept, Some(expected));
        Ok(())
    }

    #[tokio::test]
    async fn null_zone_stats_conservatively_keep_zone_and_subrange_clips_mask() -> VortexResult<()>
    {
        // Zone 0 has unknown (null) min/max stats so `a > 11` must keep zone 0 and zone 3,
        // while excluding zones 1 (4..7) and 2 (8..11).
        let filter = gt(get_item("a", root()), lit(11i32));
        let (scan, _) = zoned_two_column_scan(Some(filter), true, Some(0))?;

        // Run over a subrange 2..14 spanning half of zone 0, zones 1..2, and half of zone 3.
        let (called, kept) = run_prune(&scan, 2..14).await?;
        assert!(called);
        // Rows 2..4 (zone 0, null stats -> kept), 4..12 (zones 1..2 -> pruned), 12..14 (zone 3 -> kept).
        let expected = Mask::from_iter((2..14).map(|row| !(4..12).contains(&row)));
        assert_eq!(kept, Some(expected));
        Ok(())
    }
}
