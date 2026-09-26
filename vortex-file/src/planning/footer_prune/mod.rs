// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File-statistics pruning: finish without output when the footer proves no row can match.

use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use crate::pruning::can_prune_file_stats;
use vortex_session::VortexSession;

use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_scan::planning::next::Next;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use crate::planning::OpenedFile;

/// Decides from the footer alone whether the file can be skipped.
///
/// Missing or inconclusive statistics retain the file; only a proof of no match rejects it. An
/// empty file is rejected outright. The stage never requests IO.
pub struct FooterPrune {
    opened: Option<OpenedFile>,
    filter: Option<Expression>,
    session: VortexSession,
    next: Next<OpenedFile>,
}

impl FooterPrune {
    /// Creates the stage for `opened`, handing survivors to `next`.
    pub fn new(
        opened: OpenedFile,
        filter: Option<Expression>,
        session: VortexSession,
        next: Next<OpenedFile>,
    ) -> Self {
        Self {
            opened: Some(opened),
            filter,
            session,
            next,
        }
    }

    /// Whether the file statistics prove the filter false for every row.
    fn can_prune(&self, opened: &OpenedFile) -> VortexResult<bool> {
        let footer = &opened.footer;
        let Some(filter) = &self.filter else {
            return Ok(false);
        };
        let Some((stats, fields)) = footer
            .statistics()
            .zip(footer.dtype().as_struct_fields_opt())
        else {
            return Ok(false);
        };
        can_prune_file_stats(
            &filter.bind(footer.dtype())?,
            footer.row_count(),
            stats,
            fields,
            &self.session,
        )
    }
}

impl IoConsumer for FooterPrune {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for FooterPrune {
    fn state(&self) -> State {
        if self.opened.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(opened) = self.opened.take() else {
            vortex_bail!("FooterPrune: compute called after Done");
        };
        let row_count = opened.footer.row_count();
        if row_count == 0 || self.can_prune(&opened)? {
            return Ok(PlannerOutput::Done);
        }
        let scope = WorkScope {
            file_ordinal: 0,
            rows: 0..row_count,
        };
        Ok(PlannerOutput::Planner(scope, (self.next)(opened)?))
    }
}

#[cfg(test)]
mod tests;
