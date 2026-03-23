// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::segments::SegmentSource;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;

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
