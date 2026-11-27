// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
pub mod writer;

use std::any::Any;
use std::sync::Arc;

use reader::StructReader;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::SessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
pub use writer::StructStrategy;

use crate::DynLayout;
use crate::Layout;
use crate::LayoutBuildContext;
use crate::LayoutChildType;
use crate::LayoutDeserialization;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutPlugin;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::LayoutSerialization;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::layouts::variant;
use crate::layouts::variant::StructLayoutMetadata;
use crate::layouts::variant::VariantStorage;
use crate::lowering::slots_from_layout;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::PackPlan;
use crate::plan::physical::PlanRef;
use crate::segments::SegmentSource;

/// Struct layout vtable.
#[derive(Clone, Debug)]
pub struct Struct;

/// Backwards-compatible name for the struct layout plugin.
pub use Struct as StructLayoutEncoding;

/// A layout decomposing a struct into one child per field and optional validity.
///
/// A struct layout also stores a shredded Variant, as the struct of its core storage and its
/// shredded child; see [`variant`](crate::layouts::variant). The layout then reads the Variant
/// as the pack of its columns.
pub type StructLayout = Layout<Struct>;

impl VTable for Struct {
    /// How the layout stores a Variant, if its dtype is a Variant.
    type LayoutData = Option<VariantStorage>;
    type Metadata = ProstMetadata<StructLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.struct");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        ProstMetadata(
            layout
                .data()
                .as_ref()
                .map(VariantStorage::metadata)
                .unwrap_or_default(),
        )
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &StructLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        let variant = if args.dtype.is_variant() {
            Some(VariantStorage::deserialize(
                args.dtype,
                metadata,
                args.session,
            )?)
        } else {
            vortex_ensure!(
                metadata.is_empty(),
                "Struct layout of a struct must have no metadata"
            );
            None
        };
        let known_children = if variant.is_some() {
            // The Variant layout declares two required columns. Later optional columns
            // do not participate in validation, partitioning or planning by this reader.
            vortex_ensure!(
                args.children.nchildren() >= 2,
                "Variant layout needs core and shredded children"
            );
            2
        } else {
            Layout::<Struct>::validate_children(args.dtype, args.children.nchildren())?;
            args.children.nchildren()
        };

        for idx in 0..known_children {
            let child_row_count = args.children.child_row_count(idx);
            vortex_ensure_eq!(
                child_row_count,
                args.row_count,
                "Struct child {idx} row count does not match parent"
            );
        }
        Ok(variant)
    }

    fn nslots(layout: &Layout<Self>) -> usize {
        // Slot 0 is always reserved for validity (present only when nullable); the remaining
        // slots are the struct fields, so field `i` always occupies slot `i + 1`.
        layout.struct_fields().nfields() + 1
    }

    fn slot_to_child(layout: &Layout<Self>, slot: usize) -> Option<usize> {
        let nullable = layout.storage_dtype().is_nullable();
        match slot {
            0 => nullable.then_some(0),
            _ => Some(slot - 1 + usize::from(nullable)),
        }
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        StructLayout::slot_dtype(layout.storage_dtype(), slot)
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

    /// A Variant is the stored columns packed back together. An extraction over this plan folds
    /// into the pack, and from there into a read of the shredded column it names.
    fn new_plan(layout: &Layout<Self>) -> VortexResult<PlanRef> {
        // SAFETY: Struct layout construction validates child count and row counts. Its slots are
        // validity then one per field, which are the Pack slots, and its logical slot schema
        // defines every field dtype plus the optional non-nullable boolean validity dtype.
        let plan = unsafe {
            PackPlan::from_slots_unchecked(
                layout.struct_fields().clone(),
                layout.storage_dtype().nullability(),
                layout.row_count(),
                slots_from_layout(&layout.to_layout()),
            )
        }
        .into_plan();

        match layout.data() {
            Some(variant) => Ok(EvalPlan::try_new(variant.variant()?, plan)?.into_plan()),
            None => Ok(plan),
        }
    }
}

/// The layout plugin of the struct layout.
///
/// A struct layout of a struct is written as `vortex.struct`. A struct layout of a Variant is
/// written as [`variant::layout_id`], so the writer can gate that format on the selected editions.
#[derive(Clone, Copy, Debug, Default)]
pub struct StructPlugin;

impl LayoutPlugin for StructPlugin {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> LayoutId {
        VTable::id(&Struct)
    }

    fn serialized_ids(&self) -> Vec<LayoutId> {
        vec![VTable::id(&Struct), variant::layout_id()]
    }

    fn serialize(
        &self,
        layout: &dyn DynLayout,
        _session: &VortexSession,
    ) -> VortexResult<LayoutSerialization> {
        let serialized_id = if layout.dtype().is_variant() {
            variant::layout_id()
        } else {
            VTable::id(&Struct)
        };
        Ok(LayoutSerialization {
            serialized_id,
            metadata: layout.metadata(),
        })
    }

    fn deserialize(
        &self,
        parts: LayoutDeserialization<'_>,
        ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<LayoutRef> {
        let wire_id = parts.serialized_id;
        let is_variant = wire_id == variant::layout_id();
        vortex_ensure!(
            is_variant || wire_id == VTable::id(&Struct),
            "Unknown struct layout ID {wire_id}"
        );
        vortex_ensure!(
            parts.dtype.is_variant() == is_variant,
            "Layout {} cannot store {}",
            parts.serialized_id,
            parts.dtype
        );
        // The contract of `vortex.struct` is frozen with no metadata at all.
        vortex_ensure!(
            is_variant || parts.metadata.is_empty(),
            "Struct layout of a struct must have no metadata"
        );
        LayoutPlugin::deserialize(
            &Struct,
            LayoutDeserialization {
                serialized_id: VTable::id(&Struct),
                ..parts
            },
            ctx,
        )
    }
}

impl Layout<Struct> {
    /// Construct a struct layout from owned children.
    pub fn new(row_count: u64, dtype: DType, children: Vec<LayoutRef>) -> Self {
        Self::validate_children(&dtype, children.len()).vortex_expect("invalid struct children");
        LayoutParts::new(
            Struct,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(children),
            None,
        )
        .into_typed()
    }

    /// Creates a struct layout of a Variant of `dtype`, shredded as `shredded`, over the struct
    /// layout `storage` of its stored struct.
    ///
    /// # Errors
    ///
    /// Returns an error if `dtype` is not a Variant, or `storage` is not a struct layout of the
    /// struct that stores a Variant of `dtype` shredded as `shredded`.
    pub fn try_new_variant(
        dtype: DType,
        shredded: DType,
        storage: &LayoutRef,
        complete: bool,
    ) -> VortexResult<Self> {
        let variant = VariantStorage::try_new(&dtype, shredded, complete)?;
        vortex_ensure_eq!(
            storage.dtype(),
            variant.storage_dtype(),
            "Struct layout of a Variant does not store its dtype and shredded dtype"
        );
        let Some(storage) = storage.as_opt::<Struct>() else {
            vortex_bail!(
                "Struct layout of a Variant must store a struct layout, found {}",
                storage.encoding_id()
            );
        };
        Ok(LayoutParts::new(
            Struct,
            dtype,
            storage.row_count(),
            Vec::new(),
            Arc::clone(storage.children()),
            Some(variant),
        )
        .into_typed())
    }

    /// Returns the fields of the struct that the layout stores.
    ///
    /// For a struct, these are its own fields. For a Variant, these are the fields of the
    /// struct of its core storage and its shredded child.
    pub fn struct_fields(&self) -> &StructFields {
        self.storage_dtype()
            .as_struct_fields_opt()
            .vortex_expect("Struct layout must store a struct")
    }

    /// Returns the dtype of the struct that the layout stores.
    fn storage_dtype(&self) -> &DType {
        self.data()
            .as_ref()
            .map_or_else(|| self.dtype(), VariantStorage::storage_dtype)
    }

    /// Invokes `per_child` for fields selected by `field_mask`.
    pub fn matching_fields<F>(&self, field_mask: &[FieldMask], mut per_child: F) -> VortexResult<()>
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

    fn validate_children(dtype: &DType, nchildren: usize) -> VortexResult<()> {
        let fields = dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Expected struct dtype"))?;
        let expected = fields.nfields() + usize::from(dtype.is_nullable());
        vortex_ensure_eq!(nchildren, expected);
        Ok(())
    }

    /// Returns the dtype of the child in logical `slot`: slot 0 is the non-nullable validity
    /// bitmap, and slot `i + 1` is struct field `i`.
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
}

#[cfg(test)]
mod tests {
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;
    use vortex_expr::lit;
    use vortex_expr::mask;
    use vortex_io::runtime::single::block_on;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::layout_children;
    use crate::layouts::flat::FlatLayout;
    use crate::segments::SegmentId;
    use crate::segments::TestSegments;
    use crate::test::SESSION;

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

    /// Reads `layout` back from the wire representation `id` and `metadata`.
    fn read(layout: &LayoutRef, id: LayoutId, metadata: &[u8]) -> VortexResult<LayoutRef> {
        let read_ctx = ReadContext::new([]);
        let children = layout_children(layout.children()?);
        StructPlugin.deserialize(
            LayoutDeserialization {
                serialized_id: id,
                dtype: layout.dtype(),
                row_count: layout.row_count(),
                metadata,
                segment_ids: vec![],
                children: children.as_ref(),
            },
            &LayoutBuildContext {
                session: &SESSION,
                array_read_ctx: &read_ctx,
            },
        )
    }

    fn nullable_variant_layout() -> VortexResult<StructLayout> {
        let dtype = DType::Variant(Nullability::Nullable);
        let shredded = DType::Primitive(PType::I64, Nullability::Nullable);
        let storage = StructLayout::new(
            3,
            VariantStorage::try_new(&dtype, shredded.clone(), true)?
                .storage_dtype()
                .clone(),
            vec![
                flat_child(dtype.clone(), 0),
                flat_child(shredded.clone(), 1),
            ],
        )
        .into_layout();
        StructLayout::try_new_variant(dtype, shredded, &storage, true)
    }

    #[test]
    fn constant_projection_keeps_its_declared_dtype_over_variant_storage() -> VortexResult<()> {
        let layout = nullable_variant_layout()?;
        let reader = layout.new_reader(
            "".into(),
            Arc::new(TestSegments::default()),
            &SESSION,
            &Default::default(),
        )?;
        let expr = mask(lit(1i32), lit(true)).bind(reader.dtype())?;
        let projection = reader.projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?;
        let result = block_on(|_| projection)?;
        assert_eq!(result.dtype(), expr.dtype());
        let expected = PrimitiveArray::from_option_iter([Some(1i32); 3]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    /// A struct layout of a struct keeps the contract of `vortex.struct`: no metadata. A struct
    /// layout of a Variant is written as `vortex.variant`, and each wire ID reads only its own
    /// dtype.
    #[test]
    fn variant_is_written_under_its_own_id() -> VortexResult<()> {
        let i32 = DType::Primitive(PType::I32, Nullability::NonNullable);
        let plain = StructLayout::new(
            3,
            two_field_struct(Nullability::NonNullable),
            vec![flat_child(i32.clone(), 0), flat_child(i32, 1)],
        )
        .into_layout();
        let written = StructPlugin.serialize(plain.as_ref(), &SESSION)?;
        assert_eq!(written.serialized_id, VTable::id(&Struct));
        assert!(written.metadata.is_empty());

        let stored = nullable_variant_layout()?;
        assert_eq!(stored.nchildren(), 2);
        assert_eq!(stored.slot_to_child(0), None);
        let stored = stored.into_layout();

        let written = StructPlugin.serialize(stored.as_ref(), &SESSION)?;
        assert_eq!(written.serialized_id, variant::layout_id());
        check_metadata("variant.metadata", &written.metadata);
        // The extra child has a different row count. Reading known columns must ignore it.
        let mut children = stored.children()?;
        children.push(
            StructLayout::new(
                0,
                DType::Struct(StructFields::empty(), Nullability::NonNullable),
                vec![],
            )
            .into_layout(),
        );
        let children = layout_children(children);
        let parts = LayoutDeserialization {
            serialized_id: written.serialized_id,
            dtype: stored.dtype(),
            row_count: 3,
            metadata: &written.metadata,
            segment_ids: vec![],
            children: children.as_ref(),
        };
        let ctx = LayoutBuildContext {
            session: &SESSION,
            array_read_ctx: &ReadContext::new([]),
        };
        let extended = StructPlugin.deserialize(parts, &ctx)?;
        assert_eq!(extended.children()?.len(), 2);
        let read_back = read(&stored, written.serialized_id, &written.metadata)?;
        assert_eq!(
            read_back.as_::<Struct>().struct_fields(),
            stored.as_::<Struct>().struct_fields()
        );

        assert!(read(&stored, VTable::id(&Struct), &written.metadata).is_err());
        assert!(read(&plain, variant::layout_id(), &[]).is_err());
        // Bytes that decode to no known field are still metadata.
        assert!(read(&plain, VTable::id(&Struct), &[0x18, 0x01]).is_err());
        Ok(())
    }
}
