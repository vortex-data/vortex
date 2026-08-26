// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::sync::Arc;

use reader::StructReader;
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
use vortex_session::registry::CachedId;
pub use writer::StructStrategy;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentSource;

/// Struct layout vtable.
#[derive(Clone, Debug)]
pub struct Struct;

/// Backwards-compatible name for the struct layout plugin.
pub use Struct as StructLayoutEncoding;

/// A layout decomposing a struct into one child per field and optional validity.
pub type StructLayout = Layout<Struct>;

impl VTable for Struct {
    type LayoutData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.struct");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        validate_children(args.dtype, args.children.nchildren())?;

        for idx in 0..args.children.nchildren() {
            let child_row_count = args.children.child_row_count(idx);
            vortex_ensure!(
                child_row_count == args.row_count,
                "Struct child {idx} row count does not match parent"
            );
        }
        Ok(())
    }

    fn nslots(layout: &Layout<Self>) -> usize {
        // Slot 0 is always reserved for validity (present only when nullable); the remaining
        // slots are the struct fields, so field `i` always occupies slot `i + 1`.
        layout.struct_fields().nfields() + 1
    }

    fn slot_to_child(layout: &Layout<Self>, slot: usize) -> Option<usize> {
        let nullable = layout.dtype().is_nullable();
        match slot {
            0 => nullable.then_some(0),
            _ => Some(slot - 1 + usize::from(nullable)),
        }
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        slot_dtype(layout.dtype(), slot)
    }

    fn child_type(layout: &Layout<Self>, slot: usize) -> LayoutChildType {
        if slot == 0 {
            LayoutChildType::Auxiliary("validity".into())
        } else {
            LayoutChildType::Field(
                layout
                    .struct_fields()
                    .field_name(slot - 1)
                    .vortex_expect("Field index out of bounds")
                    .clone(),
            )
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(StructReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.session(),
            ctx.clone(),
        )?))
    }
}

/// Convenience methods for [`StructLayout`].
pub trait StructLayoutExt: Sized {
    /// Construct a struct layout from owned children.
    fn new(row_count: u64, dtype: DType, children: Vec<LayoutRef>) -> Self;

    /// Returns the struct fields.
    fn struct_fields(&self) -> &StructFields;

    /// Invokes `per_child` for fields selected by `field_mask`.
    fn matching_fields<F>(&self, field_mask: &[FieldMask], per_child: F) -> VortexResult<()>
    where
        F: FnMut(FieldMask, usize) -> VortexResult<()>;
}

impl StructLayoutExt for StructLayout {
    fn new(row_count: u64, dtype: DType, children: Vec<LayoutRef>) -> Self {
        validate_children(&dtype, children.len()).vortex_expect("invalid struct children");
        LayoutParts::new(
            Struct,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(children),
            (),
        )
        .into_typed()
    }

    /// Returns the struct fields.
    fn struct_fields(&self) -> &StructFields {
        self.dtype()
            .as_struct_fields_opt()
            .vortex_expect("Struct layout dtype must be a struct")
    }

    /// Invokes `per_child` for fields selected by `field_mask`.
    fn matching_fields<F>(&self, field_mask: &[FieldMask], mut per_child: F) -> VortexResult<()>
    where
        F: FnMut(FieldMask, usize) -> VortexResult<()>,
    {
        if field_mask.iter().any(|mask| mask.matches_all()) {
            for idx in 0..self.struct_fields().nfields() {
                per_child(FieldMask::All, idx)?;
            }
            return Ok(());
        }

        for path in field_mask {
            let Some(field) = path.starting_field()? else {
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

fn validate_children(dtype: &DType, nchildren: usize) -> VortexResult<()> {
    let fields = dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("Expected struct dtype"))?;
    let expected = fields.nfields() + usize::from(dtype.is_nullable());
    vortex_ensure!(
        nchildren == expected,
        "Struct layout has {nchildren} children, expected {expected}"
    );
    Ok(())
}

fn slot_dtype(dtype: &DType, slot: usize) -> VortexResult<DType> {
    if slot == 0 {
        Ok(DType::Bool(Nullability::NonNullable))
    } else {
        dtype
            .as_struct_fields_opt()
            .and_then(|fields| fields.field_by_index(slot - 1))
            .ok_or_else(|| vortex_err!("Missing field {}", slot - 1))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::PType;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::layouts::flat::FlatLayout;
    use crate::layouts::flat::FlatLayoutExt;
    use crate::segments::SegmentId;

    fn flat_child(dtype: DType, segment: u32) -> LayoutRef {
        FlatLayout::new(3, dtype, SegmentId::from(segment), ReadContext::new([])).into_layout()
    }

    fn two_field_struct(nullability: Nullability) -> DType {
        let i32 = DType::Primitive(PType::I32, Nullability::NonNullable);
        DType::Struct(
            StructFields::from_iter([("a", i32.clone()), ("b", i32)]),
            nullability,
        )
    }

    /// Field `i` must occupy logical slot `i + 1` regardless of whether the struct is nullable,
    /// with slot 0 reserved for validity (absent when non-nullable).
    #[test]
    fn field_slots_are_stable_across_nullability() -> VortexResult<()> {
        let i32 = DType::Primitive(PType::I32, Nullability::NonNullable);
        let bool_ = DType::Bool(Nullability::NonNullable);

        let non_null = StructLayout::new(
            3,
            two_field_struct(Nullability::NonNullable),
            vec![flat_child(i32.clone(), 0), flat_child(i32.clone(), 1)],
        );
        // Validity + two fields, but validity is absent so only two serialized children.
        assert_eq!(non_null.nslots(), 3);
        assert_eq!(non_null.nchildren(), 2);
        assert_eq!(non_null.slot_to_child(0), None);
        assert_eq!(non_null.slot_to_child(1), Some(0));
        assert_eq!(non_null.slot_to_child(2), Some(1));
        // The validity slot is absent, so materializing it and its type both yield `None`.
        assert!(non_null.slot(0)?.is_none());
        assert_eq!(non_null.slot_type(0), None);

        let nullable = StructLayout::new(
            3,
            two_field_struct(Nullability::Nullable),
            vec![
                flat_child(bool_, 0),
                flat_child(i32.clone(), 1),
                flat_child(i32, 2),
            ],
        );
        assert_eq!(nullable.nslots(), 3);
        assert_eq!(nullable.nchildren(), 3);
        assert_eq!(nullable.slot_to_child(0), Some(0));
        assert_eq!(nullable.slot_to_child(1), Some(1));
        assert_eq!(nullable.slot_to_child(2), Some(2));
        // The validity slot is present only for the nullable struct.
        assert!(nullable.slot(0)?.is_some());
        assert_eq!(
            nullable.slot_type(0),
            Some(LayoutChildType::Auxiliary("validity".into()))
        );

        // Field "a" is slot 1 in both layouts, even though its serialized index differs.
        for layout in [&non_null, &nullable] {
            assert_eq!(
                layout.slot_type(1),
                Some(LayoutChildType::Field(FieldName::from("a")))
            );
        }

        Ok(())
    }
}
