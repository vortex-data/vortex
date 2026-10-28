// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One file planner walks the selection and hands children to independently scheduled tasks.

use std::ops::Range;

use futures::SinkExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::stream::BoxStream;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_scan::selection::Selection;

use super::ScanPlans;
use super::SelectedRows;
use super::projection::ProjectionPlanner;
use super::filter::FilterPlanner;

/// Advances the file's selection one work chunk at a time, emitting only selected projections.
pub(crate) struct FilePlanner {
    plans: ScanPlans,
    chunks: SelectionChunks,
}

impl FilePlanner {
    pub(crate) fn new(
        plans: ScanPlans,
        rows: Range<u64>,
        chunk_rows: usize,
        selection: Selection,
    ) -> Self {
        Self {
            plans,
            chunks: SelectionChunks {
                rows,
                chunk_rows: chunk_rows as u64,
                selection,
            },
        }
    }

    /// Runs the file planner on its own task. The receiver schedules each emitted child, while
    /// bounded handoff prevents the producer from allocating the rest of the scan in advance.
    pub(crate) fn into_stream(
        mut self,
        lookahead: usize,
    ) -> BoxStream<'static, VortexResult<(WorkScope, Box<dyn Planner>)>> {
        let (mut send, mut recv) = mpsc::channel(lookahead);
        let planning = self.plans.session.handle().spawn(async move {
            while self.state() == State::NeedsCompute {
                let output = match self.compute() {
                    Ok(PlannerOutput::Planner(scope, child)) => Ok((scope, child)),
                    Ok(PlannerOutput::Done) => break,
                    Ok(PlannerOutput::Continue) => continue,
                    Ok(output) => Err(vortex_err!("Unexpected file planner output: {output:?}")),
                    Err(error) => Err(error),
                };
                let failed = output.is_err();
                if send.send(output).await.is_err() || failed {
                    break;
                }
            }
        });
        async_stream::stream! {
            while let Some(child) = recv.next().await {
                yield child;
            }
            // Keep the task owned by the stream, propagate panics, and cancel it on early drop.
            planning.await;
        }
        .boxed()
    }
}

impl IoConsumer for FilePlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for FilePlanner {
    fn state(&self) -> State {
        if self.chunks.rows.is_empty() {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        loop {
            let Some(rows) = self.chunks.next() else {
                return Ok(PlannerOutput::Done);
            };
            if let Some((scope, child)) = plan_selection(
                &self.plans,
                WorkScope {
                    file_ordinal: 0,
                    rows,
                },
                &self.chunks.selection,
            ) {
                return Ok(PlannerOutput::Planner(scope, child));
            }
        }
    }
}

/// A lazy cursor over fixed row boundaries. Indexed selections seek directly to occupied chunks.
struct SelectionChunks {
    rows: Range<u64>,
    chunk_rows: u64,
    selection: Selection,
}

impl Iterator for SelectionChunks {
    type Item = Range<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rows.is_empty() {
            return None;
        }
        let first = match &self.selection {
            Selection::IncludeByIndex(indices) => indices
                .get(indices.partition_point(|&row| row < self.rows.start))
                .copied(),
            Selection::IncludeRoaring(indices) => {
                let mut iter = indices.iter();
                iter.advance_to(self.rows.start);
                iter.next()
            }
            _ => Some(self.rows.start),
        };
        let Some(first) = first.filter(|&row| row < self.rows.end) else {
            self.rows.start = self.rows.end;
            return None;
        };
        let aligned_start = first / self.chunk_rows * self.chunk_rows;
        let start = aligned_start.max(self.rows.start);
        let end = aligned_start
            .saturating_add(self.chunk_rows)
            .min(self.rows.end);
        self.rows.start = end;
        Some(start..end)
    }
}

/// Plans only the selected span, retaining one projection planner and IO session per work chunk.
fn plan_selection(
    plans: &ScanPlans,
    scope: WorkScope,
    selection: &Selection,
) -> Option<(WorkScope, Box<dyn Planner>)> {
    let selected = select_rows(scope, selection)?;
    if plans.filter.is_some() {
        return Some((
            selected.scope.clone(),
            Box::new(FilterPlanner::new(plans.clone(), selected)),
        ));
    }
    Some((
        selected.scope.clone(),
        Box::new(ProjectionPlanner::new(plans.clone(), selected)),
    ))
}

fn select_rows(mut scope: WorkScope, selection: &Selection) -> Option<SelectedRows> {
    let count = match selection {
        Selection::IncludeByIndex(indices) => {
            let first = indices.partition_point(|&row| row < scope.rows.start);
            let end = indices.partition_point(|&row| row < scope.rows.end);
            if first == end {
                return None;
            }
            scope.rows = indices[first]..indices[end - 1] + 1;
            (end - first) as u64
        }
        _ => selection.row_count_in_range(&scope.rows),
    };
    if count == 0 {
        return None;
    }
    // Contiguous selections need neither a bitmap nor a filtering operator.
    let mask = if count == scope.rows.end - scope.rows.start {
        Mask::new_true(usize::try_from(count).vortex_expect("a work chunk fits usize"))
    } else {
        selection.row_mask(&scope.rows).mask().clone()
    };
    let first = mask.first()?;
    let end = mask.last()? + 1;
    scope.rows = scope.rows.start + first as u64..scope.rows.start + end as u64;
    Some(SelectedRows {
        scope,
        mask: mask.slice(first..end),
    })
}

#[cfg(test)]
mod tests;
