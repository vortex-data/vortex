// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use cudarc::driver::CudaContext;
use once_cell::sync::OnceCell;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::segments::SegmentSource;
use crate::{GpuLayoutReaderRef, LayoutChildren};

pub struct LazyGpuReaderChildren {
    children: Arc<dyn LayoutChildren>,
    segment_source: Arc<dyn SegmentSource>,
    cache: Vec<OnceCell<GpuLayoutReaderRef>>,
}

impl LazyGpuReaderChildren {
    pub fn new(children: Arc<dyn LayoutChildren>, segment_source: Arc<dyn SegmentSource>) -> Self {
        let nchildren = children.nchildren();
        let cache = (0..nchildren).map(|_| OnceCell::new()).collect();
        Self {
            children,
            segment_source,
            cache,
        }
    }

    pub fn get(
        &self,
        idx: usize,
        dtype: &DType,
        name: &Arc<str>,
        ctx: &Arc<CudaContext>,
    ) -> VortexResult<&GpuLayoutReaderRef> {
        if idx >= self.cache.len() {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.cache.len());
        }

        self.cache[idx].get_or_try_init(|| {
            let child = self.children.child(idx, dtype)?;
            child.new_gpu_reader(name.clone(), self.segment_source.clone(), ctx.clone())
        })
    }
}
