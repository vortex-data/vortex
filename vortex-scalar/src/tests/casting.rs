// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for type casting and coercion between different scalar types.

#[cfg(test)]
mod tests {
    use std::fmt::Display;
    use std::sync::Arc;

    use vortex_dtype::DType;
    use vortex_dtype::ExtDType;
    use vortex_dtype::ExtID;
    use vortex_dtype::FieldDType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_dtype::StructFields;
    use vortex_dtype::extension::ExtDTypeVTable;
    use vortex_dtype::half::f16;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::PValue;
    use crate::Scalar;
    use crate::ScalarValue;
    use crate::extension::ExtScalarVTable;

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct MultiplyVTable;

    impl ExtDTypeVTable for MultiplyVTable {
        type Metadata = usize;

        fn id(&self) -> ExtID {
            ExtID::new_ref("simple_ext")
        }

        fn validate_dtype(
            &self,
            _options: &Self::Metadata,
            storage_dtype: &DType,
        ) -> VortexResult<()> {
            match storage_dtype {
                DType::Primitive(..) => Ok(()),
                _ => Err(vortex_err!("Expected primitive dtype for simple extension")),
            }
        }
    }

    impl ExtScalarVTable for MultiplyVTable {
        type Value<'a> = usize;

        fn unpack<'a>(
            &self,
            metadata: &'a <Self as ExtDTypeVTable>::Metadata,
            _storage_dtype: &'a DType,
            storage_value: &'a ScalarValue,
        ) -> Self::Value<'a> {
            let pvalue = storage_value
                .as_primitive_opt()
                .vortex_expect("storage value was not a primitive");

            let raw_value = match *pvalue {
                PValue::U8(v) => v as usize,
                PValue::U16(v) => v as usize,
                PValue::U32(v) => v as usize,
                PValue::U64(v) => usize::try_from(v).vortex_expect("unable to convert to usize"),
                PValue::I8(v) => v as usize,
                PValue::I16(v) => v as usize,
                PValue::I32(v) => v as usize,
                PValue::I64(v) => usize::try_from(v).vortex_expect("unable to convert to usize"),
                _ => panic!("Expected an integer PValue"),
            };

            raw_value * metadata
        }

        fn validate_scalar_value(
            &self,
            _metadata: &<Self as ExtDTypeVTable>::Metadata,
            _storage_dtype: &DType,
            _storage_value: &ScalarValue,
        ) -> VortexResult<()> {
            // Any primitive type is fine so we don't need to verify this.
            Ok(())
        }
    }

    #[test]
    fn cast_to_from_extension_types() {
        // Multiply all values by 42.
        let simple_ext = ExtDType::<MultiplyVTable>::try_new(
            42,
            DType::Primitive(PType::U16, Nullability::NonNullable),
        )
        .unwrap();

        let ext_dtype = DType::Extension(simple_ext.clone().erased());
        let storage_scalar = Scalar::new(
            DType::clone(simple_ext.storage_dtype()),
            Some(ScalarValue::Primitive(PValue::U16(1000))),
        );

        // Multiply all values by 42.
        let ext_scalar = Scalar::extension::<MultiplyVTable>(42, storage_scalar.clone());
        assert_eq!(ext_scalar.dtype(), &ext_dtype);

        // to self
        let expected_dtype = &ext_dtype;
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // to nullable self
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type
        let expected_dtype = simple_ext.storage_dtype();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast to the storage type, nullable
        let expected_dtype = &simple_ext.storage_dtype().as_nullable();
        let actual = ext_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // TODO(connor): This is not possible without giving the cast method a session so it can
        // look up the vtable of the extension type it wants to cast to.
        /*
        // cast from storage type to extension
        let expected_dtype = &ext_dtype;
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);

        // cast from storage type to extension, nullable
        let expected_dtype = &ext_dtype.as_nullable();
        let actual = storage_scalar.cast(expected_dtype).unwrap();
        assert_eq!(actual.dtype(), expected_dtype);
        */

        // cast from *incompatible* storage type to extension
        let simple_ext_u8 = ExtDType::<MultiplyVTable>::try_new(
            0,
            DType::Primitive(PType::U8, Nullability::NonNullable),
        )
        .unwrap();
        let expected_dtype = &DType::Extension(simple_ext_u8.erased());
        let result = storage_scalar.cast(expected_dtype);
        assert!(
            result
                .as_ref()
                .is_err_and(|err| { err.to_string().contains("Cannot cast u16 to u8") }),
            "{result:?}"
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
            Some(ScalarValue::Primitive(PValue::U32(42))),
            Some(ScalarValue::Primitive(PValue::U64(
                f16_value.to_bits() as u64
            ))),
            Some(ScalarValue::Primitive(PValue::F32(f32_value))),
        ];

        let scalar = Scalar::new(struct_dtype, Some(ScalarValue::List(field_values)));

        let struct_scalar = scalar.as_struct();
        let fields: Vec<_> = (0..3)
            .map(|i| struct_scalar.field_by_idx(i).unwrap())
            .collect();

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
        // Test that when types already match, no coercion happens.
        let i32_value = 42i32;
        let scalar = Scalar::new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            Some(ScalarValue::Primitive(PValue::I32(i32_value))),
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
            Some(ScalarValue::Primitive(PValue::U64(
                f16_value1.to_bits() as u64
            ))),
            Some(ScalarValue::Primitive(PValue::U64(
                f16_value2.to_bits() as u64
            ))),
        ];

        let scalar = Scalar::new(list_dtype, Some(ScalarValue::List(elements)));

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
        // Test that values too large for target type are not coerced.
        let large_u64 = u64::MAX;

        // This should NOT be coerced to F16 because it's too large.
        let scalar = Scalar::new(
            DType::Primitive(PType::F16, Nullability::NonNullable),
            Some(ScalarValue::Primitive(PValue::U64(large_u64))),
        );

        let _ = scalar.as_primitive(); // Should panic
    }

    #[test]
    fn test_extension_dtype_coercion() {
        // Create an extension type with f16 storage
        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
        struct F16Ext;
        impl ExtDTypeVTable for F16Ext {
            type Metadata = usize;

            fn id(&self) -> ExtID {
                ExtID::new_ref("f16_ext")
            }

            fn validate_dtype(
                &self,
                _options: &Self::Metadata,
                _storage_dtype: &DType,
            ) -> VortexResult<()> {
                Ok(())
            }
        }

        struct Nothing;

        impl Display for Nothing {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Nothing")
            }
        }

        impl ExtScalarVTable for F16Ext {
            type Value<'a> = Nothing;

            fn unpack<'a>(
                &self,
                _metadata: &'a <Self as ExtDTypeVTable>::Metadata,
                _storage_dtype: &'a DType,
                _storage_value: &'a ScalarValue,
            ) -> Self::Value<'a> {
                Nothing
            }

            fn validate_scalar_value(
                &self,
                _metadata: &<Self as ExtDTypeVTable>::Metadata,
                _storage_dtype: &DType,
                _storage_value: &ScalarValue,
            ) -> VortexResult<()> {
                Ok(())
            }
        }

        let storage_dtype = DType::Primitive(PType::F16, Nullability::NonNullable);

        // Test f16 value stored as u64 gets coerced through extension type
        let f16_value = f16::from_f32(0.42);
        let u64_bits = f16_value.to_bits() as u64;

        let storage_scalar = Scalar::new(
            storage_dtype,
            Some(ScalarValue::Primitive(PValue::U64(u64_bits))),
        );

        let scalar = Scalar::extension::<F16Ext>(0, storage_scalar);

        // Verify the value was coerced to f16
        assert_eq!(
            scalar
                .as_extension()
                .to_storage_scalar()
                .as_primitive()
                .pvalue()
                .unwrap(),
            PValue::F16(f16_value)
        );
    }

    #[test]
    fn test_extension_dtype_nested_struct_coercion() {
        // Create an extension type with struct storage that contains f16 field
        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
        struct StructExt;
        impl ExtDTypeVTable for StructExt {
            type Metadata = usize;

            fn id(&self) -> ExtID {
                ExtID::new_ref("struct_ext")
            }

            fn validate_dtype(
                &self,
                _options: &Self::Metadata,
                _storage_dtype: &DType,
            ) -> VortexResult<()> {
                Ok(())
            }
        }

        struct Nothing;

        impl Display for Nothing {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Nothing")
            }
        }

        impl ExtScalarVTable for StructExt {
            type Value<'a> = Nothing;

            fn unpack<'a>(
                &self,
                _metadata: &'a <Self as ExtDTypeVTable>::Metadata,
                _storage_dtype: &'a DType,
                _storage_value: &'a ScalarValue,
            ) -> Self::Value<'a> {
                Nothing
            }

            fn validate_scalar_value(
                &self,
                _metadata: &<Self as ExtDTypeVTable>::Metadata,
                _storage_dtype: &DType,
                _storage_value: &ScalarValue,
            ) -> VortexResult<()> {
                Ok(())
            }
        }

        let struct_dtype = DType::Struct(
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
        );

        // Create struct value with f16 stored as u64
        let f16_value = f16::from_f32(1.5);
        let field_values = vec![
            Some(ScalarValue::Primitive(PValue::U32(123))),
            Some(ScalarValue::Primitive(PValue::U64(
                f16_value.to_bits() as u64
            ))),
        ];

        let storage_scalar = Scalar::new(struct_dtype, Some(ScalarValue::List(field_values)));

        let scalar = Scalar::extension::<StructExt>(0, storage_scalar);

        // Verify the struct field was coerced
        let list_elems = scalar
            .as_extension()
            .to_storage_scalar()
            .as_struct()
            .fields_iter()
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
