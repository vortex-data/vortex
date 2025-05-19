mod reader;
pub mod writer;

use std::collections::BTreeSet;
use std::sync::Arc;

use reader::StructReader;
use vortex_array::{ArrayContext, DeserializeMetadata, EmptyMetadata};
use vortex_dtype::{DType, Field, FieldMask, StructDType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err, vortex_panic};

use crate::children::{LayoutChildren, OwnedLayoutChildren};
use crate::segments::{SegmentId, SegmentSource};
use crate::{
    LayoutChildType, LayoutEncodingRef, LayoutId, LayoutReaderRef, LayoutRef, VTable, vtable,
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

    fn metadata(_layout: &Self::Layout) -> Self::Metadata {
        EmptyMetadata
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        layout.struct_dtype().nfields()
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        layout
            .children
            .child(idx, &layout.struct_dtype().field_by_index(idx)?)
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        LayoutChildType::Field(
            layout
                .struct_dtype()
                .field_name(idx)
                .vortex_expect("Field index out of bounds")
                .clone(),
        )
    }

    fn register_splits(
        layout: &Self::Layout,
        field_mask: &[FieldMask],
        row_offset: u64,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        layout.matching_fields(field_mask, |mask, idx| {
            layout
                .child(idx)?
                .register_splits(&[mask], row_offset, splits)
        })
    }

    fn new_reader(
        layout: &Self::Layout,
        name: &Arc<str>,
        segment_source: &Arc<dyn SegmentSource>,
        ctx: &ArrayContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(StructReader::try_new(
            layout.clone(),
            name.clone(),
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

#[derive(Clone, Debug)]
pub struct StructLayout {
    row_count: u64,
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
}

impl StructLayout {
    pub fn new(row_count: u64, dtype: DType, children: Vec<LayoutRef>) -> Self {
        Self {
            row_count,
            dtype,
            children: OwnedLayoutChildren::layout_children(children),
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
