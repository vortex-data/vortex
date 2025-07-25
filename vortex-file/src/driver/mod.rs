// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod direct;

use crate::SegmentSpec;
use std::sync::Arc;
use vortex_error::VortexResult;
use vortex_io::ReadAt;
use vortex_layout::segments::SegmentSource;

/// A trait for providing an implementation of a [`SegmentSource`].
pub trait FileDriver {
    fn create_segment_source(
        &self,
        read: Arc<dyn ReadAt>,
        segments: Arc<[SegmentSpec]>,
    ) -> VortexResult<Arc<dyn SegmentSource>>;
}
