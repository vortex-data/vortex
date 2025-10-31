// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::{Hash, Hasher};

use vortex_compute::mask::MaskValidity;
use vortex_dtype::{DType, FieldName};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_vector::VectorOps;

use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::{ArrayVTable, NotSupported, OperatorVTable, VTable, VisitorVTable};
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayEq, ArrayHash, ArrayRef, EncodingId,
    EncodingRef, Precision, vtable,
};

vtable!(GetItem);

/// An array that extracts the given field from a Struct array.
///
/// The validity of the field is intersected with the validity of the parent Struct array.
#[derive(Debug, Clone)]
pub struct GetItemArray {
    child: ArrayRef,
    field: FieldName,
    dtype: DType,
    stats: ArrayStats,
}

impl GetItemArray {
    /// Create a new get_item array.
    pub fn try_new(child: ArrayRef, field: FieldName) -> VortexResult<Self> {
        let DType::Struct(fields, _) = child.dtype() else {
            vortex_bail!(
                "GetItem can only be applied to Struct arrays, got {}",
                child.dtype()
            );
        };

        let Some(dtype) = fields.field(&field) else {
            vortex_bail!("Field '{}' does not exist in Struct array", field);
        };

        // Make the field nullable if the parent struct is nullable
        let dtype = dtype.with_nullability(dtype.nullability() | child.dtype().nullability());

        Ok(Self {
            child,
            field,
            dtype,
            stats: ArrayStats::default(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct GetItemEncoding;

impl VTable for GetItemVTable {
    type Array = GetItemArray;
    type Encoding = GetItemEncoding;
    type ArrayVTable = Self;
    type CanonicalVTable = NotSupported;
    type OperationsVTable = NotSupported;
    type ValidityVTable = NotSupported;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = NotSupported;
    type OperatorVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::from("vortex.get_item")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::from(GetItemEncoding.as_ref())
    }
}

impl ArrayVTable<GetItemVTable> for GetItemVTable {
    fn len(array: &GetItemArray) -> usize {
        array.child.len()
    }

    fn dtype(array: &GetItemArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &GetItemArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &GetItemArray, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
        array.field.hash(state);
    }

    fn array_eq(array: &GetItemArray, other: &GetItemArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision) && array.field == other.field
    }
}

impl VisitorVTable<GetItemVTable> for GetItemVTable {
    fn visit_buffers(_array: &GetItemArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // No buffers
    }

    fn visit_children(array: &GetItemArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("struct", array.child.as_ref());
    }
}

impl OperatorVTable<GetItemVTable> for GetItemVTable {
    fn bind(
        array: &GetItemArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let child = ctx.bind(&array.child, selection)?;

        // Find the index of the field in the struct
        let idx = array
            .child
            .dtype()
            .as_struct_fields()
            .find(&array.field)
            .ok_or_else(|| vortex_err!("Field '{}' does not exist in Struct array", array.field))?;

        Ok(kernel(move || {
            let struct_ = child.execute()?.into_struct();

            // We must intersect the validity with that of the parent struct
            let field = struct_.fields()[idx].clone();
            let field = MaskValidity::mask_validity(field, struct_.validity());

            Ok(field)
        }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{bitbuffer, buffer};
    use vortex_dtype::{FieldNames, Nullability, PTypeDowncast};
    use vortex_vector::VectorOps;

    use crate::arrays::{BoolArray, PrimitiveArray, StructArray};
    use crate::compute::arrays::get_item::GetItemArray;
    use crate::validity::Validity;
    use crate::{ArrayOperator, IntoArray};

    #[test]
    fn test_get_item_basic() {
        // Create a non-nullable struct with non-nullable fields
        let int_field = PrimitiveArray::from_iter([10i32, 20, 30, 40]);
        let bool_field = BoolArray::from_iter([true, false, true, false]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["numbers", "flags"]),
            vec![int_field.into_array(), bool_field.into_array()],
            4,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();

        // Extract the "numbers" field
        let get_item = GetItemArray::try_new(struct_array, "numbers".into())
            .unwrap()
            .into_array();

        // Verify the dtype is non-nullable
        assert_eq!(get_item.dtype().nullability(), Nullability::NonNullable);

        // Execute and verify the values
        let result = get_item.execute().unwrap().into_primitive().into_i32();
        assert_eq!(result.elements(), &buffer![10i32, 20, 30, 40]);
    }

    #[test]
    fn test_get_item_nullable_struct_nonnullable_field() {
        // Create a nullable struct with non-nullable field
        // The result should be nullable because the struct is nullable
        let int_field = PrimitiveArray::from_iter([10i32, 20, 30, 40]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["numbers"]),
            vec![int_field.into_array()],
            4,
            Validity::from_iter([true, false, true, false]),
        )
        .unwrap()
        .into_array();

        // Extract the "numbers" field
        let get_item = GetItemArray::try_new(struct_array, "numbers".into())
            .unwrap()
            .into_array();

        // The dtype should be nullable even though the field itself is non-nullable
        assert_eq!(get_item.dtype().nullability(), Nullability::Nullable);

        // Execute and verify values and validity
        let result = get_item.execute().unwrap().into_primitive().into_i32();
        assert_eq!(result.elements(), &buffer![10i32, 20, 30, 40]);

        // Check that validity was properly intersected
        // Elements at indices 1 and 3 should be null due to struct validity
        assert_eq!(result.validity().to_bit_buffer(), bitbuffer![1 0 1 0]);
    }

    #[test]
    fn test_get_item_with_selection() {
        // Create a struct with multiple fields
        let int_field = PrimitiveArray::from_iter([10i32, 20, 30, 40, 50, 60]);
        let bool_field = BoolArray::from_iter([true, false, true, false, true, false]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["numbers", "flags"]),
            vec![int_field.into_array(), bool_field.into_array()],
            6,
            Validity::from_iter([true, true, false, true, true, false]),
        )
        .unwrap()
        .into_array();

        // Extract the "numbers" field
        let get_item = GetItemArray::try_new(struct_array, "numbers".into())
            .unwrap()
            .into_array();

        // Apply selection mask [1 0 1 0 1 0] => select indices 0, 2, 4
        let selection = bitbuffer![1 0 1 0 1 0].into_array();
        let result = get_item
            .execute_with_selection(Some(&selection))
            .unwrap()
            .into_primitive()
            .into_i32();

        // Should have 3 elements: indices 0, 2, 4
        assert_eq!(result.len(), 3);
        assert_eq!(result.elements(), &buffer![10i32, 30, 50]);

        // Check validity: index 0 is valid, index 2 is null (struct), index 4 is valid
        assert_eq!(result.validity().to_bit_buffer(), bitbuffer![1 0 1]);
    }

    #[test]
    fn test_get_item_intersects_validity() {
        // Test that field validity is intersected with struct validity
        // Field has nulls at indices 1, 3
        let int_field =
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(30), None, Some(50)]);

        // Struct has nulls at indices 2, 4
        let struct_array = StructArray::try_new(
            FieldNames::from(["values"]),
            vec![int_field.into_array()],
            5,
            Validity::from_iter([true, true, false, true, false]),
        )
        .unwrap()
        .into_array();

        let get_item = GetItemArray::try_new(struct_array, "values".into())
            .unwrap()
            .into_array();

        let result = get_item.execute().unwrap().into_primitive().into_i32();

        // Verify that nulls are correctly combined:
        // Index 0: valid (both valid)
        // Index 1: null (field null)
        // Index 2: null (struct null)
        // Index 3: null (field null)
        // Index 4: null (struct null)
        assert_eq!(result.validity().to_bit_buffer(), bitbuffer![1 0 0 0 0]);
    }

    #[test]
    fn test_get_item_bool_field() {
        // Test extracting a boolean field
        let bool_field = BoolArray::from_iter([true, false, true, false]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["flags"]),
            vec![bool_field.into_array()],
            4,
            Validity::from_iter([true, false, true, true]),
        )
        .unwrap()
        .into_array();

        let get_item = GetItemArray::try_new(struct_array, "flags".into())
            .unwrap()
            .into_array();

        let result = get_item.execute().unwrap().into_bool();

        // Verify values
        assert_eq!(result.bits(), &bitbuffer![1 0 1 0]);

        // Verify validity (index 1 should be null from struct)
        assert_eq!(result.validity().to_bit_buffer(), bitbuffer![1 0 1 1]);
    }
}
