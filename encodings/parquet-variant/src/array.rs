// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::Field;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::EmptyArrayData;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::Variant;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::list::ListArrayExt;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowSession;
use vortex_arrow::ExecuteArrowExt;
use vortex_arrow::level_validity;
use vortex_arrow::to_arrow_null_buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ParquetVariant;
use crate::ParquetVariantArray;
use crate::arrow::prepare_variant_for_export;

#[array_slots(ParquetVariant)]
pub struct ParquetVariantSlots {
    /// The validity bitmap indicating which elements are non-null.
    #[slot(0)]
    pub validity_child: Option<ArrayRef>,
    /// The metadata array for the Parquet variant values.
    #[slot(1)]
    pub metadata: ArrayRef,
    /// The value array containing the Parquet variant data.
    #[slot(2)]
    pub value: Option<ArrayRef>,
    /// The typed value array for strongly-typed Parquet variant data.
    #[slot(3)]
    pub typed_value: Option<ArrayRef>,
}

impl ParquetVariant {
    /// Creates a Parquet Variant array from canonical extension storage slots.
    ///
    /// The `metadata` slot must be non-nullable binary, `value` must be binary when present, and
    /// at least one of `value` or `typed_value` must be present.
    pub fn try_new(
        validity: Validity,
        metadata: ArrayRef,
        value: Option<ArrayRef>,
        typed_value: Option<ArrayRef>,
    ) -> VortexResult<Array<Self>> {
        let len = metadata.len();
        let dtype = DType::Variant(validity.nullability());
        let slots = ParquetVariantSlots {
            validity_child: validity_to_child(&validity, len),
            metadata,
            value,
            typed_value,
        }
        .into_slots();
        Array::try_from_parts(ArrayParts::new(
            ParquetVariant,
            dtype,
            len,
            EmptyArrayData,
            slots,
        ))
    }

    /// Import Arrow Parquet Variant storage through the session.
    ///
    /// The result is nullable when the Arrow variant has a null buffer. See the
    /// [import nullability rules](ArrowSession::from_arrow_array) for how the children import.
    pub fn from_arrow_variant(
        arrow_variant: &ArrowVariantArray,
        session: &ArrowSession,
    ) -> VortexResult<ArrayRef> {
        Self::from_arrow_variant_with_nullability(
            arrow_variant,
            arrow_variant.nulls().is_some(),
            session,
        )
    }

    pub(crate) fn from_arrow_variant_nullable(
        arrow_variant: &ArrowVariantArray,
        session: &ArrowSession,
    ) -> VortexResult<ArrayRef> {
        Self::from_arrow_variant_with_nullability(arrow_variant, true, session)
    }

    pub(crate) fn from_arrow_variant_with_nullability(
        arrow_variant: &ArrowVariantArray,
        nullable: bool,
        session: &ArrowSession,
    ) -> VortexResult<ArrayRef> {
        let storage = arrow_variant.inner();
        let mut value_nullable = false;
        let mut typed_value_nullable = false;
        for field in storage.fields() {
            match field.name().as_str() {
                "value" => value_nullable = field.is_nullable(),
                "typed_value" => typed_value_nullable = field.is_nullable(),
                _ => {}
            }
        }
        let validity = level_validity(arrow_variant.nulls(), nullable)?;

        let metadata = session
            .from_arrow_array(ArrowArrayRef::clone(arrow_variant.metadata_column()), false)?;

        let value = Some(session.from_arrow_array(
            ArrowArrayRef::clone(arrow_variant.value_column()),
            value_nullable,
        )?);

        let typed_value = arrow_variant
            .typed_value_column()
            .map(|tv| session.from_arrow_array(ArrowArrayRef::clone(tv), typed_value_nullable))
            .transpose()?;

        ParquetVariant::try_new(validity, metadata, value, typed_value).map(IntoArray::into_array)
    }
}

pub(crate) fn core_storage_without_typed_value(
    array: &ParquetVariantArray,
) -> VortexResult<ArrayRef> {
    // The spec requires at least one of `value`/`typed_value` to be present
    // (matching the Arrow canonical extension contract). When we lift `typed_value` out into
    // the outer `VariantArray::shredded` slot and the original had no `value`, synthesize an
    // all-null `value` so the remaining `ParquetVariant` still satisfies that invariant and
    // can round-trip back through `to_arrow`.
    let value = array.value().cloned().or_else(|| {
        array.typed_value().map(|_| {
            ConstantArray::new(
                Scalar::null(DType::Binary(Nullability::Nullable)),
                array.len(),
            )
            .into_array()
        })
    });

    ParquetVariant::try_new(array.validity()?, array.metadata().clone(), value, None)
        .map(IntoArray::into_array)
}

/// Converts a Parquet `typed_value` tree into the storage-agnostic canonical shredded tree.
///
/// Parquet shredding represents nested fields with wrapper structs containing `value` and/or
/// `typed_value`. This strips those wrappers, preserves list/struct shape, and leaves primitive
/// typed values unchanged.
pub(crate) fn logical_shredded_from_parquet_typed_value(
    metadata: &ArrayRef,
    typed_value: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if typed_value.dtype().is_list() {
        // Lists keep their offsets and validity. Each element reads its row's metadata, so the
        // elements are taken from a `ListArray` whose offsets map elements to rows.
        let list_array = match typed_value.as_opt::<List>() {
            Some(list_array) => list_array.into_owned(),
            None => list_from_list_view(typed_value.execute::<ListViewArray>(ctx)?, ctx)?,
        };
        let elements = list_array.elements().clone();
        let element_metadata =
            element_metadata(metadata, list_array.offsets(), elements.len(), ctx)?;
        let elements =
            logical_shredded_from_parquet_field(&element_metadata, elements.clone(), ctx)?
                .unwrap_or(elements);
        return Ok(ListArray::try_new(
            elements,
            list_array.offsets().clone(),
            list_array.list_validity(),
            ctx,
        )?
        .into_array());
    }

    if !typed_value.dtype().is_struct() {
        return Ok(typed_value);
    }
    let struct_array = typed_value.execute::<StructArray>(ctx)?;

    // For object shredding, each struct field is a logical object field.
    let mut names = Vec::new();
    let mut fields = Vec::new();
    for (name, field) in struct_array
        .names()
        .iter()
        .zip(struct_array.iter_unmasked_fields())
    {
        if let Some(logical_field) =
            logical_shredded_from_parquet_field(metadata, field.clone(), ctx)?
        {
            names.push(FieldName::from(name.as_ref()));
            fields.push(logical_field);
        }
    }

    Ok(StructArray::try_new(
        FieldNames::from_iter(names),
        fields,
        struct_array.len(),
        struct_array.validity()?,
    )?
    .into_array())
}

/// Converts one Parquet shredded field to the corresponding canonical shredded child.
///
/// A wrapper with only a `value` column becomes an unshredded Variant child, because the Parquet
/// shredding spec keeps a shredded field's value in its own wrapper, not in the parent residual.
/// Returns `None` only for a wrapper with neither column.
fn logical_shredded_from_parquet_field(
    metadata: &ArrayRef,
    field: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !field.dtype().is_struct() {
        return Ok(Some(field));
    }
    let field_struct = field.execute::<StructArray>(ctx)?;

    let only_parquet_fields = field_struct
        .names()
        .iter()
        .all(|name| matches!(name.as_ref(), "value" | "typed_value"));
    if only_parquet_fields {
        let validity = field_struct.validity()?;
        let value = field_struct
            .unmasked_field_by_name_opt("value")
            .map(|value| {
                if validity.definitely_no_nulls() {
                    Ok(value.clone())
                } else {
                    value
                        .clone()
                        .mask(validity.to_array(value.len()))?
                        .optimize_ctx(ctx)
                }
            })
            .transpose()?;
        let Some(typed_value) = field_struct.unmasked_field_by_name_opt("typed_value") else {
            let Some(value) = value else {
                return Ok(None);
            };
            let field_validity = inferred_shredded_field_validity(Some(&value), None, ctx)?;
            let parquet_field =
                ParquetVariant::try_new(field_validity, metadata.clone(), Some(value), None)?;
            return VariantArray::try_new(parquet_field.into_array(), None)
                .map(|array| Some(array.into_array()));
        };
        // `unmasked_field_by_name_opt` intentionally ignores the parent struct validity.
        // Reapply it here so null wrapper rows become null typed/raw rows downstream.
        let typed_value = if validity.definitely_no_nulls() {
            typed_value.clone()
        } else {
            typed_value
                .clone()
                .mask(validity.to_array(typed_value.len()))?
                .optimize_ctx(ctx)?
        };

        let Some(value) = value else {
            // Fully shredded field: recurse through the typed subtree and expose its
            // logical shape directly.
            return logical_shredded_from_parquet_typed_value(metadata, typed_value, ctx).map(Some);
        };

        // Partially shredded terminal object: keep raw `value` available as the nested
        // Variant core storage while exposing any typed children as nested `shredded`.
        let validity = inferred_shredded_field_validity(Some(&value), Some(&typed_value), ctx)?;
        let parquet_field =
            ParquetVariant::try_new(validity, metadata.clone(), Some(value), Some(typed_value))?;
        let shredded = parquet_field
            .typed_value()
            .cloned()
            .map(|typed_value| {
                logical_shredded_from_parquet_typed_value(metadata, typed_value, ctx)
            })
            .transpose()?;
        return VariantArray::try_new(core_storage_without_typed_value(&parquet_field)?, shredded)
            .map(|array| Some(array.into_array()));
    }

    logical_shredded_from_parquet_typed_value(metadata, field_struct.into_array(), ctx).map(Some)
}

/// Returns the metadata of each list element: the metadata of the row whose list holds it.
fn element_metadata(
    metadata: &ArrayRef,
    offsets: &ArrayRef,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let offsets = offsets
        .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(ctx)?;

    let mut rows = BufferMut::<u64>::zeroed(len);
    for (row, bounds) in offsets.as_slice::<u64>().windows(2).enumerate() {
        let start = usize::try_from(bounds[0])?;
        let end = usize::try_from(bounds[1])?;
        rows.get_mut(start..end)
            .ok_or_else(|| vortex_err!("list offsets {start}..{end} exceed {len} elements"))?
            .fill(row as u64);
    }
    metadata.take(rows.freeze().into_array())
}

fn inferred_shredded_field_validity(
    value: Option<&ArrayRef>,
    typed_value: Option<&ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Validity> {
    let len = value
        .or(typed_value)
        .map(ArrayRef::len)
        .vortex_expect("at least one shredded field child");
    let value_mask = value
        .map(|value| value.validity()?.execute_mask(len, ctx))
        .transpose()?;
    let typed_mask = typed_value
        .map(|typed_value| typed_value.validity()?.execute_mask(len, ctx))
        .transpose()?;
    let validity = match (value_mask, typed_mask) {
        (Some(value_mask), Some(typed_mask)) => &value_mask | &typed_mask,
        (Some(mask), None) | (None, Some(mask)) => mask,
        (None, None) => unreachable!("at least one shredded field child"),
    };
    Ok(Validity::from_mask(validity, Nullability::Nullable))
}

/// Reconstructs a Parquet `typed_value` tree from the storage-agnostic canonical shredded tree.
///
/// This is the inverse of [`logical_shredded_from_parquet_typed_value`]: canonicalization strips
/// the Parquet `value`/`typed_value` wrapper shells out of the shredded tree (representing
/// partially shredded fields as nested [`VariantArray`]s), and this re-adds them. The result is a
/// valid Parquet `typed_value` that can be reattached to a [`ParquetVariant`] and serialized to
/// Arrow, where [`parquet_variant_compute::unshred_variant`] merges it back with the residual
/// `value`.
///
/// `forward` then `inverse` is structure-preserving for the shapes canonicalization produces;
/// fields the forward transform omitted (Parquet wrapper shells with no `typed_value`) are served
/// from the residual `value` and are intentionally not reconstructed here.
pub(crate) fn parquet_typed_value_from_logical_shredded(
    shredded: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(list_array) = shredded.as_opt::<List>() {
        let elements = parquet_shredded_field_from_logical(list_array.elements().clone(), ctx)?;
        return Ok(ListArray::try_new(
            elements,
            list_array.offsets().clone(),
            list_array.list_validity(),
            ctx,
        )?
        .into_array());
    }

    // A shredded object read from a file can be chunked, which hides its fields.
    let shredded = if shredded.dtype().is_struct() && !shredded.is::<Struct>() {
        shredded.execute::<StructArray>(ctx)?.into_array()
    } else {
        shredded
    };
    let Some(struct_array) = shredded.as_opt::<Struct>() else {
        // A bare typed leaf (a fully shredded scalar) is already a valid Parquet `typed_value`.
        return Ok(shredded);
    };

    let mut names = Vec::with_capacity(struct_array.names().len());
    let mut fields = Vec::with_capacity(struct_array.names().len());
    for (name, field) in struct_array
        .names()
        .iter()
        .zip(struct_array.iter_unmasked_fields())
    {
        names.push(FieldName::from(name.as_ref()));
        fields.push(parquet_shredded_field_from_logical(field.clone(), ctx)?);
    }

    Ok(StructArray::try_new(
        FieldNames::from_iter(names),
        fields,
        struct_array.len(),
        struct_array.validity()?,
    )?
    .into_array())
}

/// Reconstructs one Parquet shredded field shell (`{value?, typed_value}`) from its canonical
/// representation, the inverse of [`logical_shredded_from_parquet_field`].
fn parquet_shredded_field_from_logical(
    logical_field: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = logical_field.len();

    // Partially shredded fields canonicalize to a nested Variant whose core storage holds the
    // residual `value` and whose own shredded tree holds the typed children.
    if let Some(variant) = logical_field.as_opt::<Variant>() {
        // The core storage of a Variant read from a file can be chunked or masked.
        let core = prepare_variant_for_export(variant.core_storage().clone(), None, ctx)?;
        let core = core.as_::<ParquetVariant>();
        let value = core.value().cloned().ok_or_else(|| {
            vortex_err!("cannot rebuild Parquet shredded field: partially shredded Variant has no residual value")
        })?;
        let typed_value = variant
            .shredded()
            .cloned()
            .map(|shredded| parquet_typed_value_from_logical_shredded(shredded, ctx))
            .transpose()?;

        let mut names = vec![FieldName::from("value")];
        let mut fields = vec![value];
        if let Some(typed_value) = typed_value {
            names.push(FieldName::from("typed_value"));
            fields.push(typed_value);
        }
        return Ok(StructArray::try_new(
            FieldNames::from_iter(names),
            fields,
            len,
            Validity::NonNullable,
        )?
        .into_array());
    }

    // Fully shredded field: rebuild its typed subtree and wrap it in a `typed_value`-only shell.
    let typed_value = parquet_typed_value_from_logical_shredded(logical_field, ctx)?;
    Ok(StructArray::try_new(
        FieldNames::from_iter([FieldName::from("typed_value")]),
        vec![typed_value],
        len,
        Validity::NonNullable,
    )?
    .into_array())
}

/// Non-slot accessors and Arrow conversion for Parquet Variant storage arrays.
pub trait ParquetVariantArrayExt: TypedArrayRef<ParquetVariant> {
    /// Returns the outer row validity for the Variant values.
    fn parquet_variant_validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.dtype().nullability())
    }

    /// Converts this storage array to Arrow's canonical Parquet Variant extension storage.
    fn to_arrow(
        &self,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowVariantArray> {
        let metadata = self.metadata();
        let len = metadata.len();
        let nulls = to_arrow_null_buffer(self.parquet_variant_validity(), len, ctx)?;

        let mut fields = Vec::with_capacity(3);
        let mut arrays: Vec<ArrowArrayRef> = Vec::with_capacity(3);

        let metadata_arrow = metadata
            .clone()
            .execute_arrow_with_options(None, options, ctx)?;
        fields.push(Arc::new(Field::new(
            "metadata",
            metadata_arrow.data_type().clone(),
            false,
        )));
        arrays.push(metadata_arrow);

        if let Some(value) = self.value() {
            let value_arrow = value
                .clone()
                .execute_arrow_with_options(None, options, ctx)?;
            fields.push(Arc::new(Field::new(
                "value",
                value_arrow.data_type().clone(),
                value.dtype().is_nullable(),
            )));
            arrays.push(value_arrow);
        }

        if let Some(typed_value) = self.typed_value() {
            let tv_arrow = typed_value
                .clone()
                .execute_arrow_with_options(None, options, ctx)?;
            fields.push(Arc::new(Field::new(
                "typed_value",
                tv_arrow.data_type().clone(),
                typed_value.dtype().is_nullable(),
            )));
            arrays.push(tv_arrow);
        }

        let struct_array = arrow_array::StructArray::try_new(fields.into(), arrays, nulls)?;
        Ok(ArrowVariantArray::try_new(&struct_array)?)
    }
}

impl<T: TypedArrayRef<ParquetVariant>> ParquetVariantArrayExt for T {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_array::Array as _;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::StringArray;
    use arrow_array::StructArray;
    use arrow_array::builder::BinaryViewBuilder;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Fields;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant_compute::ShreddedSchemaBuilder;
    use parquet_variant_compute::VariantArray as ArrowVariantArray;
    use parquet_variant_compute::VariantArrayBuilder;
    use parquet_variant_compute::json_to_variant;
    use parquet_variant_compute::shred_variant;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::variant::VariantArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::validity::Validity;
    use vortex_arrow::ArrowExportOptions;
    use vortex_arrow::ArrowSessionExt;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::ParquetVariant;
    use crate::array::ParquetVariantArrayExt;
    use crate::array::ParquetVariantArraySlotsExt;
    use crate::array::parquet_typed_value_from_logical_shredded;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn assert_arrow_variant_storage_roundtrip(struct_array: StructArray) -> VortexResult<()> {
        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        let vortex_arr = ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())?;
        assert_eq!(vortex_arr.len(), struct_array.len());
        assert_eq!(
            vortex_arr.dtype(),
            &DType::Variant(if struct_array.nulls().is_some() {
                Nullability::Nullable
            } else {
                Nullability::NonNullable
            })
        );
        let inner = vortex_arr
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant child"))?;

        let mut ctx = SESSION.create_execution_ctx();
        let roundtripped = inner.to_arrow(&ArrowExportOptions::default(), &mut ctx)?;
        let roundtripped = roundtripped.inner();

        assert_eq!(struct_array.len(), roundtripped.len());
        assert_eq!(struct_array.column_names(), roundtripped.column_names());
        assert_eq!(struct_array.nulls(), roundtripped.nulls());
        assert_eq!(struct_array.fields().len(), roundtripped.fields().len());

        for (expected, actual) in struct_array
            .fields()
            .iter()
            .zip(roundtripped.fields().iter())
        {
            assert_eq!(expected.name(), actual.name());
            assert_eq!(expected.data_type(), actual.data_type());
            assert_eq!(expected.is_nullable(), actual.is_nullable());
        }

        for (expected, actual) in struct_array
            .columns()
            .iter()
            .zip(roundtripped.columns().iter())
        {
            assert_eq!(expected.to_data(), actual.to_data());
        }

        Ok(())
    }

    fn binary_view_array<const N: usize>(values: [&[u8]; N]) -> ArrowArrayRef {
        let mut builder = BinaryViewBuilder::new();
        for value in values {
            builder.append_value(value);
        }
        Arc::new(builder.finish())
    }

    #[test]
    fn test_from_arrow_variant_with_shredded_typed_value() -> VortexResult<()> {
        let mut metadata_builder = BinaryViewBuilder::new();
        let min_metadata = [1u8, 0];
        for _ in 0..3 {
            metadata_builder.append_value(min_metadata);
        }
        let metadata = metadata_builder.finish();

        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![Some(10), None, Some(30)]));

        let struct_fields: Fields = vec![
            Arc::new(Field::new("metadata", DataType::BinaryView, false)),
            Arc::new(Field::new("typed_value", DataType::Int32, true)),
        ]
        .into();
        let struct_array =
            StructArray::try_new(struct_fields, vec![Arc::new(metadata), typed_value], None)?;

        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;

        let vortex_arr = ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())?;
        assert_eq!(vortex_arr.len(), 3);
        assert_eq!(
            vortex_arr.dtype(),
            &DType::Variant(Nullability::NonNullable)
        );

        let parquet_array = vortex_arr
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant array"))?;
        let typed_value = parquet_array
            .typed_value()
            .ok_or_else(|| vortex_err!("expected typed_value child"))?;
        assert_arrays_eq!(
            typed_value,
            PrimitiveArray::from_option_iter([Some(10), None, Some(30)]),
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }

    #[test]
    fn test_to_arrow_basic() -> VortexResult<()> {
        let metadata = VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00"]).into_array();
        let value = VarBinViewArray::from_iter_bin([b"\x10", b"\x11"]).into_array();
        let pv_array = ParquetVariant::try_new(Validity::NonNullable, metadata, Some(value), None)?;

        let mut ctx = SESSION.create_execution_ctx();
        let variant_arr = pv_array.to_arrow(&ArrowExportOptions::default(), &mut ctx)?;
        let struct_arr = variant_arr.inner();

        assert_eq!(struct_arr.num_columns(), 2);
        assert_eq!(struct_arr.column_names(), &["metadata", "value"]);

        assert_eq!(struct_arr.len(), 2);
        assert_eq!(struct_arr.null_count(), 0);
        assert_eq!(
            struct_arr.column(0).to_data(),
            binary_view_array([b"\x01\x00", b"\x01\x00"]).to_data()
        );
        assert_eq!(
            struct_arr.column(1).to_data(),
            binary_view_array([b"\x10", b"\x11"]).to_data()
        );
        Ok(())
    }

    #[test]
    fn test_to_arrow_with_typed_value() -> VortexResult<()> {
        let metadata = VarBinViewArray::from_iter_bin([b"\x01\x00", b"\x01\x00"]).into_array();
        let value = VarBinViewArray::from_iter_bin([b"\x10", b"\x11"]).into_array();
        let typed_value = buffer![1i32, 2].into_array();
        let pv_array = ParquetVariant::try_new(
            Validity::NonNullable,
            metadata,
            Some(value),
            Some(typed_value),
        )?;

        let mut ctx = SESSION.create_execution_ctx();
        let variant_arr = pv_array.to_arrow(&ArrowExportOptions::default(), &mut ctx)?;
        let struct_arr = variant_arr.inner();

        assert_eq!(struct_arr.num_columns(), 3);
        assert_eq!(
            struct_arr.column_names(),
            &["metadata", "value", "typed_value"]
        );

        assert_eq!(struct_arr.len(), 2);
        assert_eq!(struct_arr.null_count(), 0);
        assert_eq!(
            struct_arr.column(0).to_data(),
            binary_view_array([b"\x01\x00", b"\x01\x00"]).to_data()
        );
        assert_eq!(
            struct_arr.column(1).to_data(),
            binary_view_array([b"\x10", b"\x11"]).to_data()
        );
        assert_eq!(
            struct_arr.column(2).to_data(),
            Int32Array::from(vec![1, 2]).to_data()
        );
        Ok(())
    }

    #[test]
    fn test_arrow_variant_roundtrip_unshredded_storage() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        builder.append_variant(PqVariant::from(true));

        assert_arrow_variant_storage_roundtrip(builder.build().into_inner())
    }

    #[test]
    fn test_arrow_variant_import_typed_value_only_normalizes_storage() -> VortexResult<()> {
        let metadata = binary_view_array([b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![metadata, typed_value],
            None,
        )?;

        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        let vortex_arr = ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())?;
        let parquet_array = vortex_arr
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant array"))?;
        let mut ctx = SESSION.create_execution_ctx();
        let value = parquet_array
            .value()
            .ok_or_else(|| vortex_err!("expected synthesized value child"))?;
        assert!(value.all_invalid(&mut ctx)?);

        let typed_value = parquet_array
            .typed_value()
            .ok_or_else(|| vortex_err!("expected typed_value child"))?;
        assert_arrays_eq!(
            typed_value,
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), Some(30)]),
            &mut ctx
        );
        Ok(())
    }

    /// A shredded object field whose shell has only a `value` column holds its value there, not
    /// in the residual object, so canonicalizing must keep the field.
    #[test]
    fn canonical_keeps_shredded_field_with_only_value() -> VortexResult<()> {
        // Metadata with the dictionary `["a"]`, and the Variant int8 7.
        let metadata = binary_view_array([b"\x01\x01\x00\x01a"]);
        let a_value = binary_view_array([b"\x0c\x07"]);
        let a_shell: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![Arc::new(Field::new("value", DataType::BinaryView, true))].into(),
            vec![a_value],
            None,
        )?);
        let typed_value: ArrowArrayRef = Arc::new(StructArray::try_new(
            vec![Arc::new(Field::new(
                "a",
                a_shell.data_type().clone(),
                false,
            ))]
            .into(),
            vec![a_shell],
            None,
        )?);
        let residual: ArrowArrayRef = Arc::new({
            let mut builder = BinaryViewBuilder::new();
            builder.append_null();
            builder.finish()
        });
        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new(
                    "typed_value",
                    typed_value.data_type().clone(),
                    true,
                )),
            ]
            .into(),
            vec![metadata, residual, typed_value],
            None,
        )?;

        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        let parquet_array = ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())?;
        let mut ctx = SESSION.create_execution_ctx();
        let expected = parquet_array.execute_scalar(0, &mut ctx)?;
        let canonical = parquet_array.execute::<Canonical>(&mut ctx)?.into_array();
        assert_eq!(canonical.execute_scalar(0, &mut ctx)?, expected);
        Ok(())
    }

    #[test]
    fn test_arrow_variant_import_value_and_typed_value_preserves_storage() -> VortexResult<()> {
        let metadata = binary_view_array([b"\x01\x00", b"\x01\x00"]);
        let value = binary_view_array([b"\x10", b"\x11"]);
        let typed_value: ArrowArrayRef = Arc::new(Int32Array::from(vec![1, 2]));

        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
                Arc::new(Field::new("typed_value", DataType::Int32, false)),
            ]
            .into(),
            vec![metadata, value, typed_value],
            None,
        )?;

        let arrow_variant = ArrowVariantArray::try_new(&struct_array)?;
        let vortex_arr = ParquetVariant::from_arrow_variant(&arrow_variant, &SESSION.arrow())?;
        let parquet_array = vortex_arr
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant array"))?;
        assert!(parquet_array.value().is_some());
        assert!(parquet_array.typed_value().is_some());

        let mut ctx = SESSION.create_execution_ctx();
        let roundtripped = parquet_array.to_arrow(&ArrowExportOptions::default(), &mut ctx)?;
        let roundtripped = roundtripped.inner();
        assert_eq!(
            roundtripped.column_names(),
            &["metadata", "value", "typed_value"]
        );
        for idx in 0..3 {
            assert_eq!(
                struct_array.column(idx).to_data(),
                roundtripped.column(idx).to_data()
            );
        }
        Ok(())
    }

    #[test]
    fn test_arrow_variant_roundtrip_with_outer_nulls() -> VortexResult<()> {
        let metadata = binary_view_array([b"\x01\x00", b"\x01\x00", b"\x01\x00"]);
        let value = binary_view_array([b"\x10", b"\x00", b"\x11"]);
        let struct_array = StructArray::try_new(
            vec![
                Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                Arc::new(Field::new("value", DataType::BinaryView, true)),
            ]
            .into(),
            vec![metadata, value],
            Some(NullBuffer::from(vec![true, false, true])),
        )?;

        assert_arrow_variant_storage_roundtrip(struct_array)
    }

    #[test]
    fn test_arrow_variant_roundtrip_with_variant_null_and_outer_null() -> VortexResult<()> {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(PqVariant::Null);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("present"));
        let inner = builder.build().into_inner();

        let struct_array = StructArray::try_new(
            inner.fields().clone(),
            inner.columns().to_vec(),
            Some(NullBuffer::from(vec![true, false, true])),
        )?;

        assert_arrow_variant_storage_roundtrip(struct_array)
    }

    /// `parquet_typed_value_from_logical_shredded` must invert the wrapper-stripping that
    /// canonicalization performs: an object-shredded Parquet variant, once canonicalized and then
    /// rebuilt, must produce the same per-row values as the original.
    #[test]
    fn parquet_typed_value_inverse_roundtrips_object_shredding() -> VortexResult<()> {
        // Shred `$.a` as Int32 over conforming, non-conforming, and missing-field rows.
        let json: ArrowArrayRef = Arc::new(StringArray::from(vec![
            Some(r#"{"a":1,"b":"x"}"#),
            Some(r#"{"a":"not-a-number","b":"y"}"#),
            Some(r#"{"b":"z"}"#),
        ]));
        let shredding = ShreddedSchemaBuilder::new()
            .with_path("a", &DataType::Int32)?
            .build();
        let shredded = shred_variant(&json_to_variant(&json)?, &shredding)?;
        let original = ParquetVariant::from_arrow_variant(&shredded, &SESSION.arrow())?;
        assert!(
            original
                .as_opt::<ParquetVariant>()
                .ok_or_else(|| vortex_err!("expected parquet variant"))?
                .typed_value()
                .is_some(),
            "fixture must be shredded"
        );

        // Canonicalize: the forward transform lifts `typed_value` into a logical shredded child.
        let mut ctx = SESSION.create_execution_ctx();
        let Canonical::Variant(canonical) = original.clone().execute::<Canonical>(&mut ctx)? else {
            return Err(vortex_err!("expected canonical variant"));
        };
        let core = canonical
            .core_storage()
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("expected parquet variant core storage"))?;
        let logical = canonical
            .shredded()
            .ok_or_else(|| vortex_err!("expected canonical shredded child"))?
            .clone();

        // Inverse transform: rebuild a Parquet `typed_value` and reattach it.
        let rebuilt_typed_value = parquet_typed_value_from_logical_shredded(logical, &mut ctx)?;
        let rebuilt = ParquetVariant::try_new(
            ParquetVariantArrayExt::parquet_variant_validity(&core),
            core.metadata().clone(),
            core.value().cloned(),
            Some(rebuilt_typed_value),
        )?
        .into_array();

        assert_eq!(rebuilt.len(), original.len());
        for idx in 0..original.len() {
            assert_eq!(
                rebuilt.execute_scalar(idx, &mut ctx)?,
                original.execute_scalar(idx, &mut ctx)?,
                "row {idx}"
            );
        }
        Ok(())
    }
}
