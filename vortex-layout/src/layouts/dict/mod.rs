mod reader;
pub mod writer;
use std::sync::Arc;

use arcref::ArcRef;
use reader::DictReader;
use vortex_array::{ArrayContext, DeserializeMetadata, ProstMetadata};
use vortex_dtype::{DType, FieldMask, FieldPath, PType};
use vortex_error::VortexResult;

use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildren, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef, LayoutVisitor, VTable,
    vtable,
};

vtable!(Dict);

impl VTable for DictVTable {
    type Layout = DictLayout;
    type Encoding = DictLayoutEncoding;
    type Metadata = ProstMetadata<DictLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        ArcRef::new_ref("vortex.dict")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        ArcRef::new_ref(DictLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.codes.row_count()
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        layout.values.dtype()
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        2
    }

    fn visit_children(
        layout: &Self::Layout,
        _field_mask: Option<&[FieldMask]>,
        visitor: &mut dyn LayoutVisitor,
    ) {
        visitor.visit_child("values", 0, Some(&FieldPath::root()), &layout.values);
        visitor.visit_child("codes", 0, None, &layout.codes);
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn new_reader(
        layout: &Arc<Self::Layout>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(DictReader::try_new(
            layout.clone(),
            segment_source,
            ctx,
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
    ) -> VortexResult<Self::Layout> {
        let values = children.child(0, dtype);
        let codes = children.child(
            1,
            &DType::Primitive(metadata.codes_ptype(), dtype.nullability()),
        );
        Ok(DictLayout { values, codes })
    }
}

#[derive(Debug)]
pub struct DictLayoutEncoding;

#[derive(Debug)]
pub struct DictLayout {
    values: LayoutRef,
    codes: LayoutRef,
}

impl DictLayout {
    pub(super) fn new(values: LayoutRef, codes: LayoutRef) -> Self {
        Self { values, codes }
    }
}

#[derive(prost::Message)]
pub struct DictLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    // i32 is required for proto, use the generated getter to read this field.
    codes_ptype: i32,
}

impl DictLayoutMetadata {
    pub fn new(codes_ptype: PType) -> Self {
        let mut metadata = Self::default();
        metadata.set_codes_ptype(codes_ptype);
        metadata
    }
}
