use std::ops::Range;
use std::sync::Arc;

use async_once_cell::Lazy;
use futures::future::BoxFuture;
use futures::FutureExt;
use vortex_array::{Array, ContextRef};
use vortex_dtype::{FieldMask, FieldPath};
use vortex_error::{vortex_err, vortex_panic, VortexError, VortexExpect, VortexResult};

use crate::layouts::flat::range_reader::FlatRangeReader;
use crate::layouts::flat::FlatLayout;
use crate::reader::LayoutReader;
use crate::segments::AsyncSegmentReader;
use crate::{Layout, LayoutRangeReader, LayoutReaderExt, LayoutVTable};

pub(crate) type LazyArray = Arc<Lazy<VortexResult<Array>, BoxFuture<'static, VortexResult<Array>>>>;

pub struct FlatReader {
    layout: Layout,
    array: LazyArray,
}

impl FlatReader {
    pub(crate) fn try_new(
        layout: Layout,
        ctx: ContextRef,
        segments: Arc<dyn AsyncSegmentReader>,
    ) -> VortexResult<Self> {
        if layout.encoding().id() != FlatLayout.id() {
            vortex_panic!("Mismatched layout ID")
        }

        let segment_id = layout
            .segment_id(0)
            .ok_or_else(|| vortex_err!("FlatLayout missing segment"))?;
        let dtype = layout.dtype().clone();
        let row_count = usize::try_from(layout.row_count())
            .vortex_expect("FlatLayout row count does not fit within usize");

        let array = Arc::new(Lazy::new(
            async move {
                // Fetch all the array segment.
                let buffer = segments.get(segment_id).await?;
                Ok::<_, VortexError>(Array::deserialize(buffer, ctx, dtype, row_count)?)
            }
            .boxed(),
        ));

        Ok(Self { layout, array })
    }
}

impl LayoutReader for FlatReader {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn range_reader(
        &self,
        row_range: Range<u64>,
        field_mask: Arc<[FieldMask]>,
    ) -> Arc<dyn LayoutRangeReader> {
        if row_range.start >= self.row_count() || row_range.end > self.row_count() {
            vortex_panic!("Row range out of bounds")
        }
        let start =
            usize::try_from(row_range.start).vortex_expect("Flat row range must fit within usize");
        let end =
            usize::try_from(row_range.end).vortex_expect("Flat row range must fit within usize");

        Arc::new(FlatRangeReader {
            row_range: start..end,
            array: self.array.clone(),
        }) as _
    }
}
