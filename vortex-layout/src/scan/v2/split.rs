// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::future::try_join_all;
use vortex_array::ArrayRef;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::request::IoConsumer;
use vortex_io::request::IoResult;
use vortex_mask::Mask;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::State;
use vortex_session::VortexSession;

use crate::layouts::row_idx::RowIdxLayoutReader;
use crate::scan::planning::PollingSegmentSource;
use crate::scan::planning::SplitMorsel;
use crate::scan::v2::ScanFile;
use crate::segments::SegmentId;

/// Everything one split needs, captured when the scan is executed.
pub(super) struct SplitTask<A> {
    pub(super) session: VortexSession,
    pub(super) file: ScanFile,
    pub(super) range: Range<u64>,
    pub(super) mask: Mask,
    pub(super) filter: Option<BoundExpression>,
    pub(super) projection: BoundExpression,
    /// The row offset of the file's first row, when the scan uses row-index expressions.
    pub(super) row_idx_offset: Option<u64>,
    pub(super) map_fn: Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>,
}

impl<A> SplitTask<A> {
    /// Drives a [`SplitMorsel`] to completion, fetching each batch of segments it asks for from
    /// the file's segment source.
    ///
    /// Every split gets its own polling source and reader. The polling source's miss list is not
    /// safe to share between morsels that compute concurrently, as split tasks do.
    pub(super) async fn run(self) -> VortexResult<Option<A>> {
        let Self {
            session,
            file,
            range,
            mask,
            filter,
            projection,
            row_idx_offset,
            map_fn,
        } = self;
        if mask.all_false() {
            return Ok(None);
        }

        let source = Arc::new(PollingSegmentSource::new(Arc::clone(&file.locations)));
        let mut reader = file.layout.new_reader(
            "".into(),
            Arc::clone(&source) as _,
            &session,
            &Default::default(),
        )?;
        if let Some(row_offset) = row_idx_offset {
            reader = Arc::new(RowIdxLayoutReader::new(row_offset, reader, session));
        }

        let mut morsel = SplitMorsel::new(source, reader, range, mask, filter, projection);
        let mut output = None;
        loop {
            match morsel.state() {
                State::Done => break,
                State::NeedsCompute => {
                    if let MorselOutput::Batch(array) = morsel.compute()?
                        && output.replace(array).is_some()
                    {
                        vortex_bail!("SplitMorsel produced more than one batch");
                    }
                }
                State::NeedsIO(batch) => {
                    let reads = batch
                        .iter()
                        .map(|request| file.segments.request(SegmentId::from(request.request.0)));
                    for (request, bytes) in batch.iter().zip(try_join_all(reads).await?) {
                        morsel.set_io_result(request.request, IoResult::Bytes(bytes));
                    }
                }
            }
        }
        output.map(|array| map_fn(array)).transpose()
    }
}
