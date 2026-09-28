// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter and project over natural splits, one morsel per split.
//!
//! Data reads go through the protocol. Each [`SplitMorsel`] holds the layout reader's future and
//! polls it once per compute against a [`PollingSegmentSource`]; the segments the reader could not
//! get become the morsel's `NeedsIO` batch, and delivered bytes let the next poll get further. The
//! reader itself is unchanged and no runtime is involved. This planner supplies the footer's
//! segment locations and one morsel per natural split.

use std::ops::Range;
use std::sync::Arc;

use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_layout::LayoutReader;
use vortex_layout::scan::planning::PollingSegmentSource;
use vortex_layout::scan::planning::SegmentLocation;
use vortex_layout::scan::planning::SplitMorsel;
use vortex_layout::segments::SegmentSource;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;

use crate::VortexFile;
use crate::planning::OpenedFile;

/// Opens the file's layout on the first compute, then emits one [`SplitMorsel`] per natural
/// split in file order and finishes.
pub struct FilterProject {
    opened: Option<OpenedFile>,
    filter: Option<Expression>,
    projection: Expression,
    session: VortexSession,
    prepared: Option<Prepared>,
    next_split: usize,
}

struct Prepared {
    /// Shared by every morsel of the file, so a segment two splits both need is read once.
    source: Arc<PollingSegmentSource>,
    reader: Arc<dyn LayoutReader>,
    filter: Option<BoundExpression>,
    projection: BoundExpression,
    splits: Vec<Range<u64>>,
}

impl FilterProject {
    /// Creates the stage; the layout is opened and expressions bound on the first compute.
    pub fn new(
        opened: OpenedFile,
        filter: Option<Expression>,
        projection: Expression,
        session: VortexSession,
    ) -> Self {
        Self {
            opened: Some(opened),
            filter,
            projection,
            session,
            prepared: None,
            next_split: 0,
        }
    }

    fn prepare(&mut self, opened: OpenedFile) -> VortexResult<()> {
        let dtype = opened.footer.dtype().clone();
        let locations = opened
            .footer
            .segment_specs_with_metadata()
            .iter()
            .map(|spec| SegmentLocation {
                offset: spec.offset,
                length: spec.length,
                alignment: spec.alignment,
            })
            .collect();
        let source = Arc::new(PollingSegmentSource::new(locations));
        let file = VortexFile::new(
            opened.footer,
            Arc::clone(&source) as Arc<dyn SegmentSource>,
            self.session.clone(),
        );
        self.prepared = Some(Prepared {
            source,
            reader: file.layout_reader()?,
            filter: self
                .filter
                .as_ref()
                .map(|filter| filter.bind(&dtype))
                .transpose()?,
            projection: self.projection.bind(&dtype)?,
            splits: file.splits()?,
        });
        Ok(())
    }
}

impl IoConsumer for FilterProject {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for FilterProject {
    fn state(&self) -> State {
        match &self.prepared {
            None if self.opened.is_some() => State::NeedsCompute,
            None => State::Done,
            Some(prepared) if self.next_split < prepared.splits.len() => State::NeedsCompute,
            Some(_) => State::Done,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if let Some(opened) = self.opened.take() {
            self.prepare(opened)?;
            return Ok(PlannerOutput::Continue);
        }
        let Some(prepared) = &self.prepared else {
            vortex_bail!("FilterProject: compute called after Done");
        };
        let Some(range) = prepared.splits.get(self.next_split).cloned() else {
            return Ok(PlannerOutput::Done);
        };
        self.next_split += 1;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: range.clone(),
        };
        Ok(PlannerOutput::Morsel(
            scope,
            Box::new(SplitMorsel::new(
                Arc::clone(&prepared.source),
                Arc::clone(&prepared.reader),
                range,
                prepared.filter.clone(),
                prepared.projection.clone(),
            )),
        ))
    }
}

#[cfg(test)]
mod tests;
