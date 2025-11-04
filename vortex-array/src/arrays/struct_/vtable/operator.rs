// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_vector::Vector;
use vortex_vector::struct_::StructVector;

use crate::ArrayRef;
use crate::arrays::{StructArray, StructVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::vtable::{OperatorVTable, ValidityHelper};

impl OperatorVTable<StructVTable> for StructVTable {
    fn bind(
        array: &StructArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        // Bind all child field arrays with the selection.
        let field_kernels: Vec<_> = array
            .fields()
            .iter()
            .map(|field| ctx.bind(field, selection))
            .collect::<VortexResult<_>>()?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        Ok(kernel(move || {
            // Execute all child field kernels.
            let fields: Vec<Vector> = field_kernels
                .into_iter()
                .map(|k| k.execute())
                .collect::<VortexResult<_>>()?;
            let validity_mask = validity.execute()?;

            Ok(StructVector::try_new(Arc::new(fields.into_boxed_slice()), validity_mask)?.into())
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::{FieldNames, PTypeDowncast};
    use vortex_vector::VectorOps;

    use crate::IntoArray;
    use crate::arrays::{BoolArray, PrimitiveArray, StructArray};
    use crate::validity::Validity;

    #[test]
    fn test_struct_operator_basic() {
        // Create a struct array with two fields: integers and booleans.
        let int_field = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let bool_field = BoolArray::from_iter([true, false, true, false, true]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["ints", "bools"]),
            vec![int_field.into_array(), bool_field.into_array()],
            5,
            Validity::AllValid,
        )
        .unwrap();

        // Execute without selection.
        let result = struct_array.execute().unwrap();
        assert_eq!(result.len(), 5);

        // Verify the struct vector fields.
        let struct_vector = result.as_struct();
        let fields = struct_vector.fields();
        assert_eq!(fields.len(), 2);

        // Verify the integer field values match the original.
        let int_vector = fields[0].as_primitive().clone().into_i32();
        assert_eq!(int_vector.elements().as_slice(), &[1, 2, 3, 4, 5]);

        // Verify the boolean field values match the original.
        let bool_vector = fields[1].as_bool();
        let bool_values: Vec<bool> = (0..5).map(|i| bool_vector.bits().value(i)).collect();
        assert_eq!(bool_values, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_struct_operator_with_mask() {
        // Create a struct array with two fields.
        let int_field = PrimitiveArray::from_iter([10i32, 20, 30, 40, 50, 60]);
        let bool_field = BoolArray::from_iter([true, false, true, false, true, false]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["numbers", "flags"]),
            vec![int_field.into_array(), bool_field.into_array()],
            6,
            Validity::AllValid,
        )
        .unwrap();

        // Create a selection mask that selects indices 0, 2, 4 (alternating pattern).
        let selection = BoolArray::from_iter([true, false, true, false, true, false]).into_array();

        // Execute with selection mask.
        let result = struct_array
            .execute_with_selection(Some(&Arc::new(selection)))
            .unwrap();

        // Verify the result has the filtered length.
        assert_eq!(result.len(), 3);

        // Verify the struct vector fields.
        let struct_vector = result.as_struct();
        let fields = struct_vector.fields();
        assert_eq!(fields.len(), 2);

        // Verify the integer field has the correct filtered values (indices 0, 2, 4).
        let int_vector = fields[0].as_primitive().clone().into_i32();
        assert_eq!(int_vector.elements().as_slice(), &[10, 30, 50]);

        // Verify the boolean field has the correct filtered values (indices 0, 2, 4).
        let bool_vector = fields[1].as_bool();
        let bool_values: Vec<bool> = (0..3).map(|i| bool_vector.bits().value(i)).collect();
        assert_eq!(bool_values, vec![true, true, true]);
    }

    #[test]
    fn test_struct_operator_null_handling() {
        // Create fields with nulls.
        let int_field = PrimitiveArray::from_option_iter([
            Some(100i32),
            None,
            Some(200),
            Some(300),
            None,
            Some(400),
        ]);

        // Create bool field with its own validity.
        let bool_array = BoolArray::from_iter([true, false, true, false, true, false]);
        let bool_validity = Validity::from_iter([true, true, false, true, true, false]);
        let bool_field = BoolArray::from_bit_buffer(bool_array.bit_buffer().clone(), bool_validity);

        // Create struct with its own validity mask (rows 1 and 4 are null).
        let struct_validity = Validity::from_iter([true, false, true, true, false, true]);

        let struct_array = StructArray::try_new(
            FieldNames::from(["values", "flags"]),
            vec![int_field.into_array(), bool_field.into_array()],
            6,
            struct_validity,
        )
        .unwrap();

        // Create a selection mask that selects indices 0, 1, 2, 4, 5.
        let selection = BoolArray::from_iter([true, true, true, false, true, true]).into_array();

        // Execute with selection mask.
        let result = struct_array
            .execute_with_selection(Some(&Arc::new(selection)))
            .unwrap();

        assert_eq!(result.len(), 5);

        // Verify the struct vector fields.
        let struct_vector = result.as_struct();
        let fields = struct_vector.fields();
        assert_eq!(fields.len(), 2);

        // Verify integer field has the correct filtered values with nulls.
        // Selected indices: 0, 1, 2, 4, 5 from [Some(100), None, Some(200), Some(300), None, Some(400)].
        let int_vector = fields[0].as_primitive().clone().into_i32();
        let int_values: Vec<Option<i32>> = (0..5).map(|i| int_vector.get(i).copied()).collect();
        assert_eq!(
            int_values,
            vec![Some(100), None, Some(200), None, Some(400)]
        );

        // Verify boolean field values.
        // Selected indices: 0, 1, 2, 4, 5 from [T, F, T, F, T, F].
        let bool_vector = fields[1].as_bool();
        let bool_values: Vec<bool> = (0..5).map(|i| bool_vector.bits().value(i)).collect();
        assert_eq!(bool_values, vec![true, false, true, true, false]);

        // Verify the struct-level validity is correctly propagated.
        // Original struct validity: [T, F, T, T, F, T]
        // Selected indices: 0, 1, 2, 4, 5 -> validity: [T, F, T, F, T].
        let validity_mask = struct_vector.validity();
        let struct_validity_values: Vec<bool> = (0..5).map(|i| validity_mask.value(i)).collect();
        assert_eq!(struct_validity_values, vec![true, false, true, false, true]);
    }
}
