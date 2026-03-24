// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_flatbuffers::FlatBuffer;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::segments::SegmentSource;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;

impl LayoutRef {
    /// Load a lazy layout from a flatbuffer.
    pub fn from_flatbuffer(
        fb: &FlatBuffer,
        dtype: &DType,
        layout_ids: Arc<[LayoutId]>,
        array_ctx: ReadContext,
        source: &Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        LayoutChild::from_flatbuffer(fb, layout_ids, array_ctx, source, session)?.resolve(dtype)
    }
}
