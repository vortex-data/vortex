// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::scan::planner::ComputeFn;
use crate::v2::scan::split::SplitId;

/// Identifies a segment read dispatched to the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadId(pub(crate) u32);

/// Identifies a compute task dispatched to the driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeId(pub(crate) u32);

/// An action the driver must perform on behalf of the scan.
pub enum ScanAction {
    /// Read the segment identified by `segment_id` from `source`, then report back with `read_id`.
    ReadSegment {
        read_id: ReadId,
        source: Arc<dyn SegmentSource>,
        segment_id: SegmentId,
    },
    /// Execute the compute function with the given inputs, then report back with `compute_id`.
    Compute {
        compute_id: ComputeId,
        compute: ComputeFn,
        segments: Vec<ByteBuffer>,
        inputs: Vec<ArrayRef>,
    },
    /// A split result is ready for the consumer.
    Emit {
        split_id: SplitId,
        result: Option<ArrayRef>,
    },
    /// The scan is complete—all splits have been emitted.
    Done,
}

/// An event the driver reports back to the scan after completing an action.
pub enum ScanEvent {
    /// A segment read completed successfully.
    SegmentReady { read_id: ReadId, buffer: ByteBuffer },
    /// A compute task completed successfully.
    ComputeReady {
        compute_id: ComputeId,
        result: ArrayRef,
    },
    /// A segment read failed.
    SegmentFailed { read_id: ReadId, error: VortexError },
    /// A compute task failed.
    ComputeFailed {
        compute_id: ComputeId,
        error: VortexError,
    },
}
