// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for type casting and coercion between different scalar types.

#[cfg(test)]
mod tests {
    use std::f32;
    use std::sync::Arc;

    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use crate::dtype::DType;
    use crate::dtype::FieldDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::dtype::UnionVariants;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtId;
    use crate::dtype::extension::ExtVTable;
    use crate::dtype::half::f16;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct Apples;

    impl ExtVTable for Apples {
        type Metadata = usize;
        type NativeValue<'a> = &'a str;

        #[expect(clippy::disallowed_methods, reason = "test-only id")]
        fn id(&self) -> ExtId {
            ExtId::new("apples")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(vec![])
        }

        fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
            Ok(0)
        }

        fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
            Ok(())
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            _storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok("")
        }
    }

    impl Apples {
        fn new() -> ExtDType<Apples> {
            ExtDType::try_new(0, DType::Primitive(PType::U16, Nullability::NonNullable))
                .vortex_expect("valid apples dtype")
        }
    }

    #[test]
    fn cast_to_from_extension_types() {
        let apples = Apples::new();

        let ext_dtype = DType::Extension(apples.clone().erased());
        let ext_scalar = Scalar::new(
            ext_dtype.clone(),
            Some(ScalarValue::Primitive(PValue::U16(1000))),
        );

        let storage_scalar = Scalar::new(
            DType::clone(apples.storage_dtype()),
            Some(ScalarValue::Primitive(PValue::U16(1000))),
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

        // cast from *incompatible* storage type to extension
        let apples_u8 =
            ExtDType::<Apples>::try_new(0, DType::Primitive(PType::U8, Nullability::NonNullable))
                .unwrap();
        let expected_dtype = &DType::Extension(apples_u8.erased());
        let result = storage_scalar.cast(expected_dtype);
        assert!(
            result
                .as_ref()
                .is_err_and(|err| { err.to_string().contains("Cannot cast 1000u16 to u8") }),
            "{result:?}"
        );
    }

    #[test]
    fn test_struct_field_coercion() {
        let f16_value = f16::from_f32(0.42);
        let f32_value = f32::consts::PI;

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

        let scalar = Scalar::new(struct_dtype, Some(ScalarValue::Tuple(field_values)));

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

        let scalar = Scalar::new(list_dtype, Some(ScalarValue::Tuple(elements)));

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
        impl ExtVTable for F16Ext {
            type Metadata = usize;
            type NativeValue<'a> = &'a str;

            #[expect(clippy::disallowed_methods, reason = "test-only id")]
            fn id(&self) -> ExtId {
                ExtId::new("f16_ext")
            }

            fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
                vortex_bail!("not implemented")
            }

            fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
                vortex_bail!("not implemented")
            }

            fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
                Ok(())
            }

            fn unpack_native<'a>(
                _ext_dtype: &'a ExtDType<Self>,
                _storage_value: &'a ScalarValue,
            ) -> VortexResult<Self::NativeValue<'a>> {
                Ok("")
            }
        }

        let storage_dtype = DType::Primitive(PType::F16, Nullability::NonNullable);
        let ext_dtype = ExtDType::<F16Ext>::try_new(0, storage_dtype).unwrap();

        // Test f16 value stored as u64 gets coerced through extension type
        let f16_value = f16::from_f32(0.42);
        let u64_bits = f16_value.to_bits() as u64;

        let scalar = Scalar::new(
            DType::Extension(ext_dtype.erased()),
            Some(ScalarValue::Primitive(PValue::U64(u64_bits))),
        );

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
        impl ExtVTable for StructExt {
            type Metadata = usize;
            type NativeValue<'a> = &'a str;

            #[expect(clippy::disallowed_methods, reason = "test-only id")]
            fn id(&self) -> ExtId {
                ExtId::new("struct_ext")
            }

            fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
                vortex_bail!("not implemented")
            }

            fn deserialize_metadata(&self, _data: &[u8]) -> VortexResult<Self::Metadata> {
                vortex_bail!("not implemented")
            }

            fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
                Ok(())
            }

            fn unpack_native<'a>(
                _ext_dtype: &'a ExtDType<Self>,
                _storage_value: &'a ScalarValue,
            ) -> VortexResult<Self::NativeValue<'a>> {
                Ok("")
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
        let ext_dtype = ExtDType::<StructExt>::try_new(0, struct_dtype).unwrap();

        // Create struct value with f16 stored as u64
        let f16_value = f16::from_f32(1.5);
        let field_values = vec![
            Some(ScalarValue::Primitive(PValue::U32(123))),
            Some(ScalarValue::Primitive(PValue::U64(
                f16_value.to_bits() as u64
            ))),
        ];

        let scalar = Scalar::new(
            DType::Extension(ext_dtype.erased()),
            Some(ScalarValue::Tuple(field_values)),
        );

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

    #[test]
    fn union_casts_apply_inner_nullability() -> VortexResult<()> {
        let source_variants = UnionVariants::try_new(
            ["int", "string"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
            ],
            vec![5, 9],
        )?;

        let target_variants = UnionVariants::try_new(
            ["int", "string"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::Nullable),
            ],
            vec![5, 9],
        )?;
        let target_dtype = DType::Union(target_variants, Nullability::NonNullable);

        let unchanged_child = Scalar::union(
            source_variants.clone(),
            5,
            Scalar::primitive(42_i32, Nullability::NonNullable),
            Nullability::NonNullable,
        )?;
        let changed_child = Scalar::union(
            source_variants,
            9,
            Scalar::utf8("hello", Nullability::NonNullable),
            Nullability::NonNullable,
        )?;

        let unchanged_child = unchanged_child.cast(&target_dtype)?;
        let changed_child = changed_child.cast(&target_dtype)?;

        assert_eq!(unchanged_child.dtype(), &target_dtype);
        assert_eq!(
            unchanged_child
                .as_union()
                .value()
                .map(|scalar| scalar.dtype().clone()),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable))
        );
        assert_eq!(changed_child.dtype(), &target_dtype);
        assert_eq!(
            changed_child
                .as_union()
                .value()
                .map(|scalar| scalar.dtype().clone()),
            Some(DType::Utf8(Nullability::Nullable))
        );

        Ok(())
    }

    #[test]
    fn union_cast_rejects_null_for_non_nullable_child() -> VortexResult<()> {
        let source_variants = UnionVariants::try_new(
            ["int"].into(),
            vec![DType::Primitive(PType::I32, Nullability::Nullable)],
            vec![5],
        )?;
        let target_variants = UnionVariants::try_new(
            ["int"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![5],
        )?;
        let scalar = Scalar::union(
            source_variants,
            5,
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            Nullability::NonNullable,
        )?;

        assert!(
            scalar
                .cast(&DType::Union(target_variants, Nullability::NonNullable))
                .is_err()
        );

        Ok(())
    }

    #[test]
    fn union_cast_widens_outer_nullability_with_selected_inner_null() -> VortexResult<()> {
        let variants = UnionVariants::try_new(
            ["int"].into(),
            vec![DType::Primitive(PType::I32, Nullability::Nullable)],
            vec![5],
        )?;
        let scalar = Scalar::union(
            variants.clone(),
            5,
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            Nullability::NonNullable,
        )?;
        let target_dtype = DType::Union(variants, Nullability::Nullable);

        let cast = scalar.cast(&target_dtype)?;
        let selected_child = cast
            .as_union()
            .value()
            .vortex_expect("present union must have a selected child");

        assert_eq!(cast.dtype(), &target_dtype);
        assert!(selected_child.is_null());
        assert_eq!(
            selected_child.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        Ok(())
    }

    #[test]
    fn union_nullability_cast_recurses_inside_list() -> VortexResult<()> {
        let source_variants = UnionVariants::try_new(
            ["int"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![5],
        )?;
        let source_union_dtype = DType::Union(source_variants.clone(), Nullability::NonNullable);
        let scalar = Scalar::list(
            source_union_dtype,
            vec![Scalar::union(
                source_variants,
                5,
                Scalar::primitive(42_i32, Nullability::NonNullable),
                Nullability::NonNullable,
            )?],
            Nullability::NonNullable,
        );

        let target_variants = UnionVariants::try_new(
            ["int"].into(),
            vec![DType::Primitive(PType::I32, Nullability::Nullable)],
            vec![5],
        )?;
        let target_union_dtype = DType::Union(target_variants, Nullability::NonNullable);
        let target_dtype = DType::List(Arc::new(target_union_dtype.clone()), Nullability::Nullable);

        let cast = scalar.cast(&target_dtype)?;
        let union = cast
            .as_list()
            .element(0)
            .vortex_expect("list must contain its union child");

        assert_eq!(cast.dtype(), &target_dtype);
        assert_eq!(union.dtype(), &target_union_dtype);
        assert_eq!(
            union.as_union().value(),
            Some(Scalar::primitive(42_i32, Nullability::Nullable))
        );

        Ok(())
    }

    #[test]
    fn union_cast_rejects_schema_changes() -> VortexResult<()> {
        let source_variants = UnionVariants::try_new(
            ["int"].into(),
            vec![DType::Primitive(PType::I32, Nullability::Nullable)],
            vec![5],
        )?;
        let scalar = Scalar::union(
            source_variants,
            5,
            Scalar::primitive(42_i32, Nullability::Nullable),
            Nullability::NonNullable,
        )?;

        let changed_name = UnionVariants::try_new(
            ["integer"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![5],
        )?;
        let changed_type_id = UnionVariants::try_new(
            ["int"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![6],
        )?;
        let changed_dtype = UnionVariants::try_new(
            ["int"].into(),
            vec![DType::Primitive(PType::I64, Nullability::Nullable)],
            vec![5],
        )?;

        for variants in [changed_name, changed_type_id, changed_dtype] {
            assert!(
                scalar
                    .cast(&DType::Union(variants, Nullability::NonNullable))
                    .is_err()
            );
        }

        Ok(())
    }

    #[test]
    fn cast_changes_only_outer_union_nullability() -> VortexResult<()> {
        let variants = UnionVariants::try_new(
            ["int", "string"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
            ],
            vec![5, 9],
        )?;
        let scalar = Scalar::union(
            variants.clone(),
            5,
            Scalar::primitive(42_i32, Nullability::NonNullable),
            Nullability::NonNullable,
        )?;

        let expected = DType::Union(variants, Nullability::Nullable);
        let nullable = scalar.cast(&expected)?;

        assert_eq!(nullable.dtype(), &expected);
        assert!(nullable.dtype().is_nullable());
        assert_eq!(nullable.as_union().type_id(), Some(5));
        assert_eq!(
            nullable
                .as_union()
                .value()
                .map(|scalar| scalar.dtype().clone()),
            Some(DType::Primitive(PType::I32, Nullability::NonNullable))
        );

        Ok(())
    }
}
