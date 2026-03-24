// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::LayoutVTable;

pub type LayoutPluginRef = Arc<dyn LayoutPlugin>;

/// Arguments for deserializing a layout from a flatbuffer.
pub struct LayoutDeserialize<'a> {
    pub dtype: &'a DType,
    pub row_count: u64,
    pub metadata: &'a [u8],
    pub children: Vec<LayoutChild>,
    pub segments: Vec<SegmentId>,
    pub segment_source: &'a Arc<dyn SegmentSource>,
    pub array_ctx: &'a ReadContext,
    pub session: &'a VortexSession,
}

pub trait LayoutPlugin: 'static + Send + Sync {
    fn id(&self) -> LayoutId;

    fn deserialize(&self, args: LayoutDeserialize<'_>) -> VortexResult<LayoutRef>;
}

impl Debug for dyn LayoutPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("LayoutPlugin").field(&self.id()).finish()
    }
}

impl<V: LayoutVTable> LayoutPlugin for V {
    fn id(&self) -> LayoutId {
        V::id(self)
    }

    fn deserialize(&self, args: LayoutDeserialize<'_>) -> VortexResult<LayoutRef> {
        let metadata = V::deserialize_metadata(
            args.metadata,
            args.dtype,
            args.row_count,
            &args.children,
            &args.array_ctx,
        )?;
        Ok(LayoutRef(Arc::new(Layout {
            vtable: self.clone(),
            metadata,
            dtype: args.dtype.clone(),
            row_count: args.row_count,
            children: args.children,
            segments: args.segments,
            segment_source: args.segment_source.clone(),
            array_ctx: args.array_ctx.clone(),
            session: args.session.clone(),
        })))
    }
}
