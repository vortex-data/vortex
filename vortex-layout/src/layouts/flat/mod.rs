mod reader;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use vortex_array::{ArrayContext, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, FieldMask};
use vortex_error::{VortexResult, vortex_bail, vortex_panic};

use crate::children::LayoutChildren;
use crate::layouts::flat::reader::FlatReader;
use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildType, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef, VTable, vtable,
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

    fn metadata(_layout: &Self::Layout) -> Self::Metadata {
        EmptyMetadata
    }

    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId> {
        vec![layout.segment_id]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        0
    }

    fn child(_layout: &Self::Layout, _idx: usize) -> VortexResult<LayoutRef> {
        vortex_bail!("Flat layout has no children");
    }

    fn child_type(_layout: &Self::Layout, _idx: usize) -> LayoutChildType {
        vortex_panic!("Flat layout has no children");
    }

    fn register_splits(
        layout: &Self::Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        for path in field_mask {
            if path.matches_root() {
                splits.insert(row_offset + layout.row_count());
                break;
            }
        }
        Ok(())
    }

    fn new_reader(
        layout: &Self::Layout,
        name: &Arc<str>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(FlatReader::new(
            layout.clone(),
            name.clone(),
            segment_source.clone(),
            ctx.clone(),
        )))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        _children: &dyn LayoutChildren,
    ) -> VortexResult<Self::Layout> {
        if segment_ids.len() != 1 {
            vortex_bail!("Flat layout must have exactly one segment ID");
        }
        Ok(FlatLayout {
            row_count,
            dtype: dtype.clone(),
            segment_id: segment_ids[0],
        })
    }
}

#[derive(Debug)]
pub struct FlatLayoutEncoding;

#[derive(Clone, Debug)]
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

    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }
}
