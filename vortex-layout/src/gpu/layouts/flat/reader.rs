// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use futures::FutureExt;
use vortex_array::serde::ArrayParts;
use vortex_array::stats::Precision;
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexResult, VortexUnwrap as _};
use vortex_expr::Expression;
use vortex_gpu::create_run_jit_kernel;

use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentSource;
use crate::{GpuArrayFuture, GpuLayoutReader, ShareGpuArrayFuture};

pub struct GpuFlatReader {
    layout: FlatLayout,
    name: Arc<str>,
    segment_source: Arc<dyn SegmentSource>,
    context: Arc<CudaContext>,
}

impl GpuFlatReader {
    pub(crate) fn new(
        layout: FlatLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        context: Arc<CudaContext>,
    ) -> Self {
        Self {
            layout,
            name,
            segment_source,
            context,
        }
    }

    /// Register the segment request and return a future that would resolve into the deserialised array.
    fn array_future(&self) -> ShareGpuArrayFuture {
        let row_count = usize::try_from(self.layout.row_count()).vortex_unwrap();

        // We create the segment_fut here to ensure we give the segment reader visibility into
        // how to prioritize this segment, even if the `array` future has already been initialized.
        // This is gross... see the function's TODO for a maybe better solution?
        let segment_fut = self.segment_source.request(self.layout.segment_id());

        let ctx = self.layout.array_ctx().clone();
        let dtype = self.layout.dtype().clone();
        async move {
            let segment = segment_fut.await?;
            ArrayParts::try_from(segment)?
                .decode(&ctx, &dtype, row_count)
                .map_err(Arc::new)
        }
        .boxed()
        .shared()
    }
}

impl GpuLayoutReader for GpuFlatReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> Precision<u64> {
        Precision::Exact(self.layout.row_count())
    }

    fn register_splits(
        &self,
        _field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        splits.insert(row_offset + self.layout.row_count());
        Ok(())
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
    ) -> VortexResult<GpuArrayFuture> {
        assert_eq!(
            row_range.clone(),
            0..self.layout.row_count(),
            "Row range {row_range:?} must cover whole layout"
        );
        let name = self.name.clone();
        let array = self.array_future();
        let expr = expr.clone();
        let ctx = self.context.clone();

        Ok(async move {
            log::debug!("Flat array evaluation {} - {}", name, expr);

            let array = array.clone().await?;
            let (gpu_result, _) = create_run_jit_kernel(&ctx, &array)?;
            Ok(vec![gpu_result])
        }
        .boxed())
    }
}
