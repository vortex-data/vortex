mod reader;
pub mod writer;

use std::sync::Arc;

use reader::StructReader;
use vortex_array::{ArrayContext, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, Field, FieldMask, FieldPath, StructDType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic};

use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildren, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef, LayoutVisitor, VTable,
    vtable,
};

vtable!(Struct);

impl VTable for StructVTable {
    type Layout = StructLayout;
    type Encoding = StructLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.struct")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(StructLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        layout.struct_dtype().nfields()
    }

    fn visit_children(
        layout: &Self::Layout,
        field_mask: Option<&[FieldMask]>,
        visitor: &mut dyn LayoutVisitor,
    ) {
        layout
            .matching_fields(
                field_mask.unwrap_or_else(|| &[FieldMask::All]),
                |_mask, idx| {
                    let dtype = layout.struct_dtype().field_by_index(idx)?;
                    let child = layout.children.child(idx, &dtype);
                    let name = layout.struct_dtype().field_name(idx)?;
                    visitor.visit_child(
                        name.as_ref(),
                        0,
                        Some(&FieldPath::from_name(name)),
                        &child,
                    );
                    Ok(())
                },
            )
            .vortex_expect("unreachable");
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn new_reader(
        layout: &Arc<Self::Layout>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(StructReader::try_new(
            layout.clone(),
            segment_source.clone(),
            ctx.clone(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
    ) -> VortexResult<Self::Layout> {
        let struct_dt = dtype
            .as_struct()
            .ok_or_else(|| vortex_err!("Expected struct dtype"))?;
        if children.nchildren() != struct_dt.nfields() {
            vortex_bail!(
                "Struct layout has {} children, but dtype has {} fields",
                children.nchildren(),
                struct_dt.nfields()
            );
        }
        Ok(StructLayout {
            row_count,
            dtype: dtype.clone(),
            children: children.to_arc(),
        })
    }
}

#[derive(Debug)]
pub struct StructLayoutEncoding;

#[derive(Debug)]
pub struct StructLayout {
    row_count: u64,
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
}

impl StructLayout {
    pub fn new(row_count: u64, dtype: DType, children: Arc<[LayoutRef]>) -> Self {
        Self {
            row_count,
            dtype,
            children,
        }
    }

    pub fn struct_dtype(&self) -> &Arc<StructDType> {
        let DType::Struct(dtype, _) = self.dtype() else {
            vortex_panic!("Mismatched dtype {} for struct layout", self.dtype());
        };
        dtype
    }

    pub fn matching_fields<F>(&self, field_mask: &[FieldMask], mut per_child: F) -> VortexResult<()>
    where
        F: FnMut(FieldMask, usize) -> VortexResult<()>,
    {
        // If the field mask contains an `All` fields, then enumerate all fields.
        if field_mask.iter().any(|mask| mask.matches_all()) {
            for idx in 0..self.struct_dtype().nfields() {
                per_child(FieldMask::All, idx)?;
            }
            return Ok(());
        }

        // Enumerate each field in the mask
        for path in field_mask {
            let Some(field) = path.starting_field()? else {
                // skip fields not in mask
                continue;
            };
            let Field::Name(field_name) = field else {
                vortex_bail!("Expected field name, got {:?}", field);
            };
            let idx = self.struct_dtype().find(field_name)?;

            per_child(path.clone().step_into()?, idx)?;
        }

        Ok(())
    }
}
