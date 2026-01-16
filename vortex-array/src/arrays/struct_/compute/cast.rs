// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::compute::CastKernel;
use crate::compute::CastKernelAdapter;
use crate::compute::cast;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

impl CastKernel for StructVTable {
    fn cast(&self, array: &StructArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let Some(target_sdtype) = dtype.as_struct_fields_opt() else {
            return Ok(None);
        };

        let source_sdtype = array.struct_fields();

        let fields_match_order = target_sdtype.nfields() == source_sdtype.nfields()
            && target_sdtype
                .names()
                .iter()
                .zip(source_sdtype.names().iter())
                .all(|(f1, f2)| f1 == f2);

        let mut cast_fields = Vec::with_capacity(target_sdtype.nfields());
        if fields_match_order {
            for (field, target_type) in array.fields().iter().zip_eq(target_sdtype.fields()) {
                let cast_field = cast(field, &target_type)?;
                cast_fields.push(cast_field);
            }
        } else {
            // Re-order, handle fields by value instead.
            // Track which source field indices have been used to handle duplicate field names.
            let mut used_source_indices = vec![false; source_sdtype.nfields()];

            for (target_name, target_type) in
                target_sdtype.names().iter().zip_eq(target_sdtype.fields())
            {
                // Find the first unused source field with this name.
                let src_field_idx = source_sdtype
                    .names()
                    .iter()
                    .enumerate()
                    .find(|(idx, name)| !used_source_indices[*idx] && *name == target_name)
                    .map(|(idx, _)| idx);

                match src_field_idx {
                    None => {
                        // No source field with this name => evolve the schema compatibly.
                        // If the field is nullable, we add a new ConstantArray field with the type.
                        vortex_ensure!(
                            target_type.is_nullable(),
                            "CAST for struct only supports added nullable fields"
                        );

                        cast_fields.push(
                            ConstantArray::new(Scalar::null(target_type), array.len()).into_array(),
                        );
                    }
                    Some(src_idx) => {
                        // Mark this source field as used.
                        used_source_indices[src_idx] = true;
                        // Field exists in source field. Cast it to the target type.
                        let cast_field = cast(&array.fields()[src_idx], &target_type)?;
                        cast_fields.push(cast_field);
                    }
                }
            }
        }

        let validity = array
            .validity()
            .clone()
            .cast_nullability(dtype.nullability(), array.len())?;

        StructArray::try_new(
            target_sdtype.names().clone(),
            cast_fields,
            array.len(),
            validity,
        )
        .map(|a| Some(a.into_array()))
    }
}

register_kernel!(CastKernelAdapter(StructVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::DecimalDType;
    use vortex_dtype::FieldNames;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;

    use crate::Array;
    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::BoolArray;
    use crate::arrays::ListArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::validity::Validity;

    #[rstest]
    #[case(create_test_struct(false))]
    #[case(create_test_struct(true))]
    #[case(create_nested_struct())]
    #[case(create_simple_struct())]
    fn test_cast_struct_conformance(#[case] array: StructArray) {
        test_cast_conformance(array.as_ref());
    }

    fn create_test_struct(nullable: bool) -> StructArray {
        let names = FieldNames::from(["a", "b"]);

        let a = buffer![1i32, 2, 3].into_array();
        let b = VarBinArray::from_iter(
            vec![Some("x"), None, Some("z")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();

        StructArray::try_new(
            names,
            vec![a, b],
            3,
            if nullable {
                Validity::AllValid
            } else {
                Validity::NonNullable
            },
        )
        .unwrap()
    }

    fn create_nested_struct() -> StructArray {
        // Create inner struct
        let inner_names = FieldNames::from(["x", "y"]);

        let x = buffer![1.0f32, 2.0, 3.0].into_array();
        let y = buffer![4.0f32, 5.0, 6.0].into_array();
        let inner_struct = StructArray::try_new(inner_names, vec![x, y], 3, Validity::NonNullable)
            .unwrap()
            .into_array();

        // Create outer struct with inner struct as a field
        let outer_names: FieldNames = ["id", "point"].into();
        // Outer struct would have fields: id (I64) and point (inner struct)

        let ids = buffer![100i64, 200, 300].into_array();

        StructArray::try_new(
            outer_names,
            vec![ids, inner_struct],
            3,
            Validity::NonNullable,
        )
        .unwrap()
    }

    fn create_simple_struct() -> StructArray {
        let names = FieldNames::from(["value"]);
        // Simple struct with a single U8 field

        let values = buffer![42u8].into_array();

        StructArray::try_new(names, vec![values], 1, Validity::NonNullable).unwrap()
    }

    #[test]
    fn cast_nullable_all_invalid() {
        let empty_struct = StructArray::try_new(
            FieldNames::from(["a"]),
            vec![PrimitiveArray::new::<i32>(buffer![], Validity::AllInvalid).to_array()],
            0,
            Validity::AllInvalid,
        )
        .unwrap()
        .to_array();

        let target_dtype = DType::struct_(
            [("a", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::NonNullable,
        );

        let result = crate::compute::cast(&empty_struct, &target_dtype).unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn cast_duplicate_field_names_to_nullable() {
        let names = FieldNames::from(["a", "a"]);
        let field1 = buffer![1i32, 2, 3].into_array();
        let field2 = buffer![10i64, 20, 30].into_array();

        let struct_array =
            StructArray::try_new(names, vec![field1, field2], 3, Validity::NonNullable).unwrap();

        let target_dtype = struct_array.dtype().as_nullable();

        let result = crate::compute::cast(struct_array.as_ref(), &target_dtype).unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_struct().fields().len(), 2);
    }

    #[test]
    fn cast_add_fields() {
        let names = FieldNames::from(["a", "b"]);
        let field1 = buffer![1i32, 2, 3].into_array();
        let field2 = buffer![10i64, 20, 30].into_array();
        let target_dtype = DType::struct_(
            [
                ("a", field1.dtype().clone()),
                ("b", field2.dtype().clone()),
                (
                    "c",
                    DType::Decimal(DecimalDType::new(38, 10), Nullability::Nullable),
                ),
            ],
            Nullability::NonNullable,
        );

        let struct_array =
            StructArray::try_new(names, vec![field1, field2], 3, Validity::NonNullable).unwrap();

        let result = crate::compute::cast(struct_array.as_ref(), &target_dtype).unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), 3);
        assert_eq!(result.to_struct().fields().len(), 3);
    }

    /// Regression test for <https://github.com/vortex-data/vortex/issues/5865>.
    ///
    /// When casting a struct with duplicate field names using the name-based lookup path
    /// (triggered by schema evolution), [`StructFields::find()`] returns the same index for
    /// all fields with the same name, causing incorrect field casting.
    #[test]
    fn cast_struct_with_duplicate_names_schema_evolution() {
        // Source struct: Two fields both named "a" with different types.
        // Field 0: bool?
        // Field 1: list(bool?)
        let names = FieldNames::from(["a", "a"]);

        let bool_field = BoolArray::from_iter([Some(true), None, Some(false)]).into_array();

        let list_elements = BoolArray::from_iter([Some(true), Some(false)]).into_array();
        let list_offsets = buffer![0i32, 1, 1, 2].into_array(); // 3 lists: [true], [], [false]
        let list_field = ListArray::try_new(list_elements, list_offsets, Validity::NonNullable)
            .unwrap()
            .into_array();

        let struct_array = StructArray::try_new(
            names,
            vec![bool_field.clone(), list_field.clone()],
            3,
            Validity::NonNullable,
        )
        .unwrap();

        // Target dtype: Same fields PLUS a new nullable field "c".
        // This triggers name-based lookup because field count differs (2 vs 3).
        let target_dtype = DType::struct_(
            [
                ("a", bool_field.dtype().clone()),
                ("a", list_field.dtype().clone()),
                ("c", DType::Primitive(PType::I32, Nullability::Nullable)),
            ],
            Nullability::NonNullable,
        );

        // BUG: The name-based lookup path calls find("a") twice, getting index 0 both times.
        // This tries to cast bool_field to list(bool?) which fails.
        let result = crate::compute::cast(struct_array.as_ref(), &target_dtype);
        assert!(
            result.is_ok(),
            "cast with duplicate field names failed: {:?}",
            result.err()
        );
    }
}
