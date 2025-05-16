mod eval_expr;
mod reader;
pub mod writer;

use std::sync::Arc;

use vortex_array::{ArrayContext, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, FieldMask};
use vortex_error::VortexResult;

use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildren, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutVisitor, VTable, vtable,
};

vtable!(Flat);

impl VTable for FlatVTable {
    type Layout = FlatLayout;
    type Encoding = FlatLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.flat")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(FlatLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        0
    }

    fn visit_children(
        _layout: &Self::Layout,
        _field_mask: Option<&[FieldMask]>,
        _visitor: &mut dyn LayoutVisitor,
    ) {
    }

    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId> {
        vec![layout.segment_id]
    }

    fn new_reader(
        layout: &Arc<Self::Layout>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        todo!()
    }

    fn build(
        encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
    ) -> VortexResult<Self::Layout> {
        todo!()
    }
}

#[derive(Debug)]
pub struct FlatLayoutEncoding;

#[derive(Clone)]
pub struct FlatLayout {
    row_count: u64,
    dtype: DType,
    segment_id: SegmentId,
}

impl FlatLayout {
    pub fn new(row_count: u64, dtype: DType, segment_id: SegmentId) -> Self {
        Self {
            row_count,
            dtype,
            segment_id,
        }
    }

    pub(super) fn segment_id(&self) -> SegmentId {
        self.segment_id
    }
}
