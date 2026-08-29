// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ScanBuilder`](vortex_layout::scan::scan_builder::ScanBuilder) integration.

use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::future::join_all;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutRef;
use vortex_layout::scan::scan_builder::ScanExecutor;
use vortex_layout::scan::scan_builder::ScanRequest;
use vortex_layout::segments::SegmentSource;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ExecPlan;
use crate::MorselExecutor;
use crate::MorselScan;
use crate::build_plan;
use crate::morsels;
use crate::nodes::ConjunctMode;

type PlanCacheKey = (String, Option<String>, ConjunctMode);

/// Morsel-driven execution backend for a layout scan builder.
pub struct MorselScanExecutor {
    layout: LayoutRef,
    segments: Arc<dyn SegmentSource>,
    target_rows: u64,
    conjunct_mode: ConjunctMode,
    threads: usize,
    plan_cache: Mutex<HashMap<PlanCacheKey, Arc<ExecPlan>>>,
}

impl MorselScanExecutor {
    /// Create an executor over a raw layout and its segment source.
    pub fn new(layout: LayoutRef, segments: Arc<dyn SegmentSource>) -> Self {
        Self {
            layout,
            segments,
            target_rows: 128 * 1024,
            conjunct_mode: ConjunctMode::Cascade,
            threads: 4,
            plan_cache: Mutex::default(),
        }
    }

    /// Set the target number of rows per morsel.
    pub fn with_target_rows(mut self, target_rows: u64) -> Self {
        self.target_rows = target_rows;
        self
    }

    /// Set the conjunct evaluation policy.
    pub fn with_conjunct_mode(mut self, conjunct_mode: ConjunctMode) -> Self {
        self.conjunct_mode = conjunct_mode;
        self
    }

    /// Set the number of affinity workers used by one scan.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }
}

impl ScanExecutor for MorselScanExecutor {
    fn build(
        &self,
        request: ScanRequest,
    ) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>> {
        if request.limit.is_some() {
            vortex_bail!("the morsel scan executor does not support limits");
        }
        if request.row_offset != 0 {
            vortex_bail!("the morsel scan executor does not support row offsets");
        }

        let projection = unbind(&request.projection)?;
        let filter = request.filter.as_ref().map(unbind).transpose()?;
        let key = (
            projection.to_string(),
            filter.as_ref().map(ToString::to_string),
            self.conjunct_mode,
        );
        let plan = {
            let mut cache = self.plan_cache.lock();
            match cache.get(&key) {
                Some(plan) => Arc::clone(plan),
                None => {
                    let plan = Arc::new(build_plan(
                        &self.layout,
                        &projection,
                        filter.as_ref(),
                        self.conjunct_mode,
                    )?);
                    cache.insert(key, Arc::clone(&plan));
                    plan
                }
            }
        };

        let full_range = request
            .row_range
            .clone()
            .unwrap_or_else(|| 0..plan.row_count());
        let selected = selected_morsels(
            morsels(&plan, self.target_rows),
            &full_range,
            &request.selection,
        );
        let pruning = selected
            .into_iter()
            .map(|(range, selection)| {
                let evaluation = request.filter.as_ref().map(|filter| {
                    request
                        .layout_reader
                        .pruning_evaluation(&range, filter, selection.clone())
                });
                (range, selection, evaluation.transpose())
            })
            .collect::<Vec<_>>();
        let segments = Arc::clone(&self.segments);
        let session = request.session.clone();
        let handle = request.session.handle();
        let threads = self.threads;

        Ok(vec![Box::pin(async move {
            let prepared = join_all(pruning.into_iter().map(
                |(range, selection, evaluation)| async move {
                    let pruning = match evaluation? {
                        Some(evaluation) => evaluation.await?,
                        None => Mask::new_true(selection.len()),
                    };
                    Ok::<_, vortex_error::VortexError>((range, pruning & &selection))
                },
            ))
            .await
            .into_iter()
            .collect::<VortexResult<Vec<_>>>()?;
            let demands = prepared
                .into_iter()
                .filter(|(_, demand)| !demand.all_false())
                .collect::<Vec<_>>();
            if demands.is_empty() {
                return Ok(None);
            }

            let executor = MorselExecutor::shared(Arc::clone(&plan), threads)?;
            let scan = MorselScan::new(Arc::clone(&plan), segments, session)
                .with_threads(threads)
                .with_morsel_demands(demands)?;
            let batches = handle
                .spawn_blocking(move || executor.run(&scan).map(|(batches, _)| batches))
                .await?;
            combine_batches(batches)
        })])
    }
}

fn combine_batches(mut batches: Vec<ArrayRef>) -> VortexResult<Option<ArrayRef>> {
    match batches.len() {
        0 => Ok(None),
        1 => Ok(batches.pop()),
        _ => {
            let dtype = batches[0].dtype().clone();
            ChunkedArray::try_new(batches, dtype).map(|array| Some(array.into_array()))
        }
    }
}

fn selected_morsels(
    morsels: Vec<Range<u64>>,
    row_range: &Range<u64>,
    selection: &vortex_scan::selection::Selection,
) -> Vec<(Range<u64>, Mask)> {
    morsels
        .into_iter()
        .filter_map(|range| {
            let range = range.start.max(row_range.start)..range.end.min(row_range.end);
            (range.start < range.end).then_some(range)
        })
        .filter_map(|range| {
            let mask = selection.row_mask(&range).mask().clone();
            (!matches!(mask.slices(), AllOr::None)).then_some((range, mask))
        })
        .collect()
}

fn unbind(expr: &BoundExpression) -> VortexResult<Expression> {
    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(Expression::Root);
    };
    Expression::try_new(
        scalar_fn.clone(),
        expr.children()
            .iter()
            .map(unbind)
            .collect::<VortexResult<Vec<_>>>()?,
    )
}
