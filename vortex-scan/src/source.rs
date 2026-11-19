// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Source inputs and byte access for scans.

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_io::request::SegmentReader;
pub use vortex_io::segments::SegmentFuture;
pub use vortex_io::segments::SegmentId;
pub use vortex_io::segments::SegmentSource;

use crate::execution::SegmentLocation;
use crate::execution::cache::ScanCache;
use crate::plan::LazyPlanRef;

/// The plan and byte source used to prepare a scan.
///
/// The lazy root defers lowering until preparation runs on a compute task. Segment identifiers
/// are local to this input. Reuse a cache only while the plan, dtype, locations, and source
/// describe the same data.
#[derive(Clone)]
pub struct ScanInput {
    /// The dtype of the root plan.
    pub dtype: DType,
    /// The root plan, initialized without reading payloads.
    pub root: LazyPlanRef,
    /// Byte locations indexed by source-local segment identifiers.
    pub locations: Arc<[SegmentLocation]>,
    /// How the session's I/O scheduler reads segments by source-local identifier.
    pub reader: SegmentReader,
    /// State shared by repeated scans of this input.
    pub cache: Option<ScanCache>,
}
