// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A selection stage that hands its surviving rows to another planner.

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_mask::Mask;

use crate::planning::next::Next;
use crate::planning::planner::Planner;
use crate::planning::planner::PlannerOutput;
use crate::planning::planner::State;
use crate::planning::planner::WorkScope;
use crate::selection::Selection;

/// Selected rows passed between planning stages.
///
/// The mask is relative to `scope.rows.start`, and its length equals the scope's row count.
/// A stage can narrow the scope while rebasing the mask to preserve absolute row coordinates.
pub struct SelectedRows {
    /// The input row domain covering the selected rows, including any unselected gaps.
    pub scope: WorkScope,
    /// The rows within that domain that the successor should process.
    pub mask: Mask,
}

/// Applies a scan selection to a scope, then constructs the next planner with the surviving rows.
///
/// The successor may project, filter, or further partition the rows. This stage neither reads
/// data nor decides how those rows are executed. An empty selection creates no successor.
/// Compact selections exclude unselected leading and trailing rows from the successor's scope.
pub struct SelectionPlanner {
    scope: Option<WorkScope>,
    selection: Arc<Selection>,
    next: Option<Next<SelectedRows>>,
}

impl SelectionPlanner {
    /// Creates an unevaluated selection stage. The mask and successor are built in `compute`.
    pub fn new(scope: WorkScope, selection: Arc<Selection>, next: Next<SelectedRows>) -> Self {
        Self {
            scope: Some(scope),
            selection,
            next: Some(next),
        }
    }
}

impl IoConsumer for SelectionPlanner {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for SelectionPlanner {
    fn state(&self) -> State {
        if self.scope.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let mut scope = self
            .scope
            .take()
            .ok_or_else(|| vortex_err!("SelectionPlanner computed after completion"))?;
        let selected = self.selection.row_mask(&scope.rows);
        let Some(start) = selected.mask().first() else {
            return Ok(PlannerOutput::Done);
        };
        let end = selected.mask().last().vortex_expect("nonempty selection") + 1;
        let mut mask = selected.mask().clone();
        // Narrow clusters to avoid planning unrelated chunks. Preserve alignment for sparse
        // gathers: slicing their encoded inputs at arbitrary boundaries can cost more than it saves.
        if end - start <= mask.true_count().saturating_mul(16) {
            mask = mask.slice(start..end);
            scope.rows = scope.rows.start + start as u64..scope.rows.start + end as u64;
        }
        let next = self
            .next
            .take()
            .ok_or_else(|| vortex_err!("SelectionPlanner has no successor"))?;
        let child = next(SelectedRows {
            scope: scope.clone(),
            mask,
        })?;
        Ok(PlannerOutput::Planner(scope, child))
    }
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::vortex_bail;

    use super::*;
    use crate::planning::next::next_fn;
    use crate::strict_sorted_buffer::StrictSortedBuffer;

    struct Finished;

    impl IoConsumer for Finished {
        fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
    }

    impl Planner for Finished {
        fn state(&self) -> State {
            State::Done
        }

        fn compute(&mut self) -> VortexResult<PlannerOutput> {
            Ok(PlannerOutput::Done)
        }
    }

    #[rstest]
    #[case::all(Selection::All, vec![10, 11, 12, 13, 14])]
    #[case::include(Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from(vec![1, 10, 13, 20]))?), vec![10, 13])]
    #[case::exclude(Selection::ExcludeByIndex(StrictSortedBuffer::try_new(Buffer::from(vec![1, 10, 13, 20]))?), vec![11, 12, 14])]
    #[case::roaring(Selection::IncludeRoaring([1, 10, 13, 20].into_iter().collect()), vec![10, 13])]
    #[case::exclude_roaring(Selection::ExcludeRoaring([1, 10, 13, 20].into_iter().collect()), vec![11, 12, 14])]
    #[case::interior(Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from(vec![11, 12, 13]))?), vec![11, 12, 13])]
    #[case::singleton(Selection::IncludeRoaring([12].into_iter().collect()), vec![12])]
    #[case::excluded_edges(Selection::ExcludeRoaring([10, 14].into_iter().collect()), vec![11, 12, 13])]
    #[case::empty(Selection::IncludeRoaring(Default::default()), vec![])]
    fn hands_selected_rows_to_successor(
        #[case] selection: Selection,
        #[case] expected: Vec<u64>,
    ) -> VortexResult<()> {
        let scope = WorkScope {
            file_ordinal: 7,
            rows: 10..15,
        };
        let received = Arc::new(Mutex::new(None));
        let output = Arc::clone(&received);
        let next = next_fn(move |selected: SelectedRows| {
            *output.lock() = Some(selected);
            Ok(Finished)
        });
        let mut planner = SelectionPlanner::new(scope.clone(), Arc::new(selection), next);
        assert_eq!(planner.state(), State::NeedsCompute);
        assert!(received.lock().is_none());
        let result = planner.compute()?;
        assert_eq!(planner.state(), State::Done);
        let selected = received.lock().take();
        if expected.is_empty() {
            assert!(matches!(result, PlannerOutput::Done));
            assert!(selected.is_none());
        } else {
            let PlannerOutput::Planner(child_scope, _) = result else {
                vortex_bail!("selection should emit its successor");
            };
            assert_eq!(child_scope.file_ordinal, scope.file_ordinal);
            assert_eq!(
                child_scope.rows,
                expected[0]..expected[expected.len() - 1] + 1
            );
            let selected = selected.ok_or_else(|| vortex_err!("missing successor input"))?;
            assert_eq!(selected.scope, child_scope);
            assert_eq!(
                selected.mask.len() as u64,
                child_scope.rows.end - child_scope.rows.start
            );
            let rows = selected
                .mask
                .iter()
                .enumerate()
                .filter(|(_, selected)| *selected)
                .map(|(index, _)| selected.scope.rows.start + index as u64)
                .collect::<Vec<_>>();
            assert_eq!(rows, expected);
        }
        assert!(planner.compute().is_err());
        Ok(())
    }

    #[test]
    fn scattered_selection_keeps_its_aligned_scope() -> VortexResult<()> {
        let scope = WorkScope {
            file_ordinal: 0,
            rows: 0..65_536,
        };
        let expected = scope.clone();
        let next = next_fn(move |selected: SelectedRows| {
            assert_eq!(selected.scope, expected);
            assert_eq!(selected.mask.len(), 65_536);
            assert_eq!(
                selected.mask.indices(),
                vortex_mask::AllOr::Some(&[100, 50_000][..])
            );
            Ok(Finished)
        });
        let selection =
            Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from(vec![
                100, 50_000,
            ]))?);
        let mut planner = SelectionPlanner::new(scope, Arc::new(selection), next);
        assert!(matches!(planner.compute()?, PlannerOutput::Planner(..)));
        Ok(())
    }
}
