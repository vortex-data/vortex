// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::segments::SegmentSource;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::LayoutVTable;

pub type LayoutPluginRef = Arc<dyn LayoutPlugin>;

pub trait LayoutPlugin: 'static + Send + Sync {
    fn id(&self) -> LayoutId;

    fn deserialize(
        &self,
        dtype: &DType,
        metadata: &[u8],
        children: Vec<LayoutChild>,
        segment_source: &Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef>;
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

    fn deserialize(
        &self,
        _dtype: &DType,
        _metadata: &[u8],
        _children: Vec<LayoutChild>,
        _segment_source: &Arc<dyn SegmentSource>,
        _session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        todo!()
    }
}
