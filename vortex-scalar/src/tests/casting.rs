// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for type casting and coercion between different scalar types.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_dtype::half::f16;
    use vortex_dtype::{DType, ExtDType, ExtID, FieldDType, Nullability, PType, StructFields};
    use vortex_error::VortexExpect;

    use crate::{InnerScalarValue, PValue, Scalar, ScalarValue};

    #[test]
    fn cast_to_from_extension_types() {
        let apples = ExtDType::new(
            ExtID::new(Arc::from("apples")),
            Arc::from(DType::Primitive(PType::U16, Nullability::NonNullable)),
            None,
        );
        let ext_dtype = DType::Extension(Arc::from(apples.clone()));
        let ext_scalar = Scalar::new(ext_dtype.clone(), ScalarValue(InnerScalarValue::Bool(true)));
        let storage_scalar = Scalar::new(
            DType::clone(apples.storage_dtype()),
            ScalarValue(InnerScalarValue::Primitive(PValue::U16(1000))),
        );

        // to self
        let expected_dtype = &ext_dtype;
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // to nullable self
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type
        let expected_dtype = apples.storage_dtype();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type, nullable
        let expected_dtype = &apples.storage_dtype().as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from storage type to extension
        let expected_dtype = &ext_dtype;
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from storage type to extension, nullable
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from *compatible* storage type to extension
        let storage_scalar_u64 = Scalar::new(
            DType::clone(apples.storage_dtype()),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(1000))),
        );
        let expected_dtype = &ext_dtype;
        let actual = storage_scalar_u64.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from *incompatible* storage type to extension
        let apples_u8 = ExtDType::new(
            ExtID::new(Arc::from("apples")),
            Arc::from(DType::Primitive(PType::U8, Nullability::NonNullable)),
            None,
        );
        let expected_dtype = &DType::Extension(Arc::from(apples_u8));
        let result = storage_scalar.cast(expected_dtype);
        assert!(
            result
                .as_ref()
                .is_err_and(|err| { err.to_string().contains("Cannot cast u16 to u8") }),
            "{result:?}"
        );
    }

    #[test]
    fn test_f16_coercion_from_u64() {
        let f16_value = f16::from_f32(5.722046e-6);
        let u64_bits = f16_value.to_bits() as u64;

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        assert_eq!(
            scalar.as_primitive().pvalue().unwrap(),
            PValue::F16(f16_value)
        );
    }

    #[test]
    fn test_f16_coercion_from_u32() {
        let f16_value = f16::from_f32(0.42);
        let u32_bits = f16_value.to_bits() as u32;

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(u32_bits))),
        );

        assert_eq!(
            scalar.as_primitive().pvalue().unwrap(),
            PValue::F16(f16_value)
        );
    }

    #[test]
    fn test_f16_coercion_from_u16() {
        let f16_value = f16::from_f32(1.5);
        let u16_bits = f16_value.to_bits();

        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U16(u16_bits))),
        );

        assert_eq!(
            scalar.as_primitive().pvalue().unwrap(),
            PValue::F16(f16_value)
        );
    }

    #[test]
    fn test_f32_coercion_from_u32() {
        let f32_value = std::f32::consts::PI;
        let u32_bits = f32_value.to_bits();

        let scalar = Scalar::new(
            DType::Primitive(PType::F32, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(u32_bits))),
        );

        assert_eq!(
            scalar.as_primitive().pvalue().unwrap(),
            PValue::F32(f32_value)
        );
    }

    #[test]
    fn test_f64_coercion_from_u64() {
        let f64_value = std::f64::consts::E;
        let u64_bits = f64_value.to_bits();

        let scalar = Scalar::new(
            DType::Primitive(PType::F64, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        assert_eq!(
            scalar.as_primitive().pvalue().unwrap(),
            PValue::F64(f64_value)
        );
    }

    #[test]
    fn test_struct_field_coercion() {
        let f16_value = f16::from_f32(0.42);
        let f32_value = std::f32::consts::PI;

        let struct_dtype = DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    FieldDType::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
                ),
                (
                    "b",
                    FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable)),
                ),
                (
                    "c",
                    FieldDType::from(DType::Primitive(PType::F32, Nullability::NonNullable)),
                ),
            ]),
            Nullability::NonNullable,
        );

        let field_values = vec![
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(42))),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value.to_bits() as u64,
            ))),
            ScalarValue(InnerScalarValue::Primitive(PValue::F32(f32_value))),
        ];

        let scalar = Scalar::new(
            struct_dtype,
            ScalarValue(InnerScalarValue::List(field_values.into())),
        );

        let struct_scalar = scalar.as_struct();
        let fields = struct_scalar.fields().unwrap().collect::<Vec<_>>();

        // Check first field (no coercion needed)
        assert_eq!(fields[0].as_primitive().pvalue().unwrap(), PValue::U32(42));

        // Check second field (f16 coerced from u64)
        assert_eq!(
            fields[1].as_primitive().pvalue().unwrap(),
            PValue::F16(f16_value)
        );

        // Check third field (no coercion needed)
        assert_eq!(
            fields[2].as_primitive().pvalue().unwrap(),
            PValue::F32(f32_value)
        );
    }

    #[test]
    fn test_fake_coercion_for_matching_type() {
        // Test that when types already match, no coercion happens
        let i32_value = 42i32;
        let scalar = Scalar::new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::I32(i32_value))),
        );

        assert_eq!(
            scalar.as_primitive().pvalue().unwrap(),
            PValue::I32(i32_value)
        );
    }

    #[test]
    fn test_list_element_coercion() {
        let f16_value1 = f16::from_f32(1.0);
        let f16_value2 = f16::from_f32(2.0);

        let list_dtype = DType::List(
            Arc::new(DType::Primitive(PType::F16, Nullability::NonNullable)),
            Nullability::NonNullable,
        );

        let elements = vec![
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value1.to_bits() as u64,
            ))),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value2.to_bits() as u64,
            ))),
        ];

        let scalar = Scalar::new(
            list_dtype,
            ScalarValue(InnerScalarValue::List(elements.into())),
        );

        let list_scalar = scalar.as_list();
        let elements = list_scalar.elements().unwrap();

        for (i, expected) in [f16_value1, f16_value2].iter().enumerate() {
            assert_eq!(
                elements[i].as_primitive().pvalue().unwrap(),
                PValue::F16(*expected)
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_coercion_with_overflow_protection() {
        // Test that values too large for target type are not coerced
        let large_u64 = u64::MAX;

        // This should NOT be coerced to F16 because it's too large
        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(large_u64))),
        );

        let _ = scalar.as_primitive(); // Should panic
    }

    #[test]
    fn test_extension_dtype_coercion() {
        // Create an extension type with f16 storage
        let ext_id = ExtID::new("test_f16_ext".into());
        let storage_dtype = Arc::new(DType::Primitive(PType::F16, Nullability::NonNullable));
        let ext_dtype = Arc::new(ExtDType::new(ext_id, storage_dtype, None));

        // Test f16 value stored as u64 gets coerced through extension type
        let f16_value = f16::from_f32(0.42);
        let u64_bits = f16_value.to_bits() as u64;

        let scalar = Scalar::new(
            DType::Extension(ext_dtype),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        // Verify the value was coerced to f16
        assert_eq!(
            scalar
                .as_extension()
                .storage()
                .as_primitive()
                .pvalue()
                .unwrap(),
            PValue::F16(f16_value)
        );
    }

    #[test]
    fn test_extension_dtype_nested_struct_coercion() {
        // Create an extension type with struct storage that contains f16 field
        let ext_id = ExtID::new("test_struct_ext".into());
        let struct_dtype = Arc::new(DType::Struct(
            StructFields::from_iter([
                (
                    "id",
                    FieldDType::from(DType::Primitive(PType::U32, Nullability::NonNullable)),
                ),
                (
                    "value",
                    FieldDType::from(DType::Primitive(PType::F16, Nullability::NonNullable)),
                ),
            ]),
            Nullability::NonNullable,
        ));
        let ext_dtype = Arc::new(ExtDType::new(ext_id, struct_dtype, None));

        // Create struct value with f16 stored as u64
        let f16_value = f16::from_f32(1.5);
        let field_values = vec![
            ScalarValue(InnerScalarValue::Primitive(PValue::U32(123))),
            ScalarValue(InnerScalarValue::Primitive(PValue::U64(
                f16_value.to_bits() as u64,
            ))),
        ];

        let scalar = Scalar::new(
            DType::Extension(ext_dtype),
            ScalarValue(InnerScalarValue::List(field_values.into())),
        );

        // Verify the struct field was coerced
        let list_elems = scalar
            .as_extension()
            .storage()
            .as_struct()
            .fields()
            .vortex_expect("non null")
            .collect::<Vec<_>>();
        assert_eq!(
            list_elems[0].as_primitive().pvalue().unwrap(),
            PValue::U32(123)
        );
        assert_eq!(
            list_elems[1].as_primitive().pvalue().unwrap(),
            PValue::F16(f16_value)
        );
    }
}
