// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;

use std::sync::Arc;

use reader::StructReader;
use vortex_array::DeserializeMetadata;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::SessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::LayoutChildType;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::LayoutChildren;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

vtable!(Struct);

impl VTable for Struct {
    type Layout = StructLayout;
    type Encoding = StructLayoutEncoding;
    type Metadata = EmptyMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new("vortex.struct")
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
        let validity_children = if layout.dtype.is_nullable() { 1 } else { 0 };
        layout.struct_fields().nfields() + validity_children
    }

    fn child(layout: &Self::Layout, index: usize) -> VortexResult<LayoutRef> {
        let schema_index = if layout.dtype.is_nullable() {
            index.saturating_sub(1)
        } else {
            index
        };

        let child_dtype = if index == 0 && layout.dtype.is_nullable() {
            DType::Bool(Nullability::NonNullable)
        } else {
            layout
                .struct_fields()
                .field_by_index(schema_index)
                .ok_or_else(|| vortex_err!("Missing field {schema_index}"))?
        };

        layout.children.child(index, &child_dtype)
    }

    fn child_type(layout: &Self::Layout, idx: usize) -> LayoutChildType {
        let schema_index = if layout.dtype.is_nullable() {
            idx.saturating_sub(1)
        } else {
            idx
        };

        if idx == 0 && layout.dtype.is_nullable() {
            LayoutChildType::Auxiliary("validity".into())
        } else {
            LayoutChildType::Field(
                layout
                    .struct_fields()
                    .field_name(schema_index)
                    .vortex_expect("Field index out of bounds")
                    .clone(),
            )
        }
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(StructReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.session(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: &ReadContext,
    ) -> VortexResult<Self::Layout> {
        let struct_dt = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Expected struct dtype"))?;

        let expected_children = struct_dt.nfields() + (dtype.is_nullable() as usize);
        vortex_ensure!(
            children.nchildren() == expected_children,
            "Struct layout has {} children, but dtype has {} fields",
            children.nchildren(),
            struct_dt.nfields()
        );

        Ok(StructLayout {
            row_count,
            dtype: dtype.clone(),
            children: children.to_arc(),
        })
    }

    fn with_children(layout: &mut Self::Layout, children: Vec<LayoutRef>) -> VortexResult<()> {
        let struct_dt = layout
            .dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Expected struct dtype"))?;

        let expected_children = struct_dt.nfields() + (layout.dtype.is_nullable() as usize);
        vortex_ensure!(
            children.len() == expected_children,
            "StructLayout expects {} children, got {}",
            expected_children,
            children.len()
        );

        layout.children = OwnedLayoutChildren::layout_children(children);
        Ok(())
    }
}

#[derive(Debug)]
pub struct StructLayoutEncoding;

/// Decomposes a struct-typed column into one child per field, enabling columnar projection.
///
/// Queries that only need a subset of fields can skip reading the rest entirely.
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

    pub fn struct_fields(&self) -> &StructFields {
        self.dtype
            .as_struct_fields_opt()
            .vortex_expect("Struct layout dtype must be a struct")
    }

    #[inline]
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    #[inline]
    pub fn children(&self) -> &Arc<dyn LayoutChildren> {
        &self.children
    }

    pub fn matching_fields<F>(&self, field_mask: &[FieldMask], mut per_child: F) -> VortexResult<()>
    where
        F: FnMut(FieldMask, usize) -> VortexResult<()>,
    {
        // If the field mask contains an `All` fields, then enumerate all fields.
        if field_mask.iter().any(|mask| mask.matches_all()) {
            for idx in 0..self.struct_fields().nfields() {
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
                vortex_bail!("Expected field name, got {field:?}");
            };
            let idx = self
                .struct_fields()
                .find(field_name)
                .ok_or_else(|| vortex_err!("Field not found: {field_name}"))?;

            per_child(path.clone().step_into()?, idx)?;
        }

        Ok(())
    }
}
