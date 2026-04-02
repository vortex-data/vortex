// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use arrow_array::StructArray;
    use arrow_buffer::NullBuffer;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant::VariantBuilderExt;
    use parquet_variant_compute::VariantArrayBuilder;
    use vortex_array::ArrayRef;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::expr::root;
    use vortex_array::expr::variant_get;
    use vortex_error::VortexResult;

    use crate::ParquetVariantData;

    /// Apply variant_get and execute through the full pipeline (including execute_parent).
    fn apply_variant_get(arr: &ArrayRef, field: &str) -> VortexResult<ArrayRef> {
        let expr = variant_get(field, root());
        let lazy = arr.clone().apply(&expr)?;
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        lazy.execute::<ArrayRef>(&mut ctx)
    }

    /// Build a VariantArray of objects: [{"a": 1, "b": "x"}, {"a": 2, "c": true}, {"b": "y"}]
    fn make_object_array() -> VortexResult<ArrayRef> {
        let mut builder = VariantArrayBuilder::new(3);

        builder
            .new_object()
            .with_field("a", 1i32)
            .with_field("b", "x")
            .finish();

        builder
            .new_object()
            .with_field("a", 2i32)
            .with_field("c", true)
            .finish();

        builder.new_object().with_field("b", "y").finish();

        ParquetVariantData::from_arrow_variant(&builder.build())
    }

    /// Build a nullable VariantArray: [{"a": 10}, NULL, {"a": 30}]
    fn make_nullable_object_array() -> VortexResult<ArrayRef> {
        let mut builder = VariantArrayBuilder::new(3);

        builder.new_object().with_field("a", 10i32).finish();

        builder.new_object().with_field("a", 20i32).finish();

        builder.new_object().with_field("a", 30i32).finish();

        let inner = builder.build().into_inner();
        let null_struct = StructArray::try_new(
            inner.fields().clone(),
            inner.columns().to_vec(),
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .unwrap();
        let arrow_variant = parquet_variant_compute::VariantArray::try_new(&null_struct).unwrap();
        ParquetVariantData::from_arrow_variant(&arrow_variant)
    }

    #[test]
    fn test_variant_get_basic() -> VortexResult<()> {
        let arr = make_object_array()?;
        let result = apply_variant_get(&arr, "a")?;

        assert_eq!(result.len(), 3);

        // Row 0: {"a": 1, ...} → variant(1)
        let s0 = result.scalar_at(0)?;
        assert!(!s0.is_null());
        let inner0 = s0.as_variant().value().unwrap();
        assert_eq!(*inner0, 1i32.into());

        // Row 1: {"a": 2, ...} → variant(2)
        let s1 = result.scalar_at(1)?;
        assert!(!s1.is_null());
        let inner1 = s1.as_variant().value().unwrap();
        assert_eq!(*inner1, 2i32.into());

        // Row 2: {"b": "y"} → null (field "a" missing)
        let s2 = result.scalar_at(2)?;
        assert!(s2.is_null());

        Ok(())
    }

    #[test]
    fn test_variant_get_missing_field() -> VortexResult<()> {
        let arr = make_object_array()?;
        let result = apply_variant_get(&arr, "nonexistent")?;

        assert_eq!(result.len(), 3);
        for i in 0..3 {
            assert!(result.scalar_at(i)?.is_null(), "row {i} should be null");
        }

        Ok(())
    }

    #[test]
    fn test_variant_get_null_input() -> VortexResult<()> {
        let arr = make_nullable_object_array()?;
        let result = apply_variant_get(&arr, "a")?;

        assert_eq!(result.len(), 3);

        // Row 0: {"a": 10} → variant(10)
        assert!(!result.scalar_at(0)?.is_null());

        // Row 1: NULL → null
        assert!(result.scalar_at(1)?.is_null());

        // Row 2: {"a": 30} → variant(30)
        assert!(!result.scalar_at(2)?.is_null());

        Ok(())
    }

    #[test]
    fn test_variant_get_non_object() -> VortexResult<()> {
        // Array of primitive variants (not objects)
        let mut builder = VariantArrayBuilder::new(2);
        builder.append_variant(PqVariant::from(42i32));
        builder.append_variant(PqVariant::from("hello"));
        let arr = ParquetVariantData::from_arrow_variant(&builder.build())?;

        let result = apply_variant_get(&arr, "a")?;

        assert_eq!(result.len(), 2);
        assert!(result.scalar_at(0)?.is_null());
        assert!(result.scalar_at(1)?.is_null());

        Ok(())
    }

    #[test]
    fn test_variant_get_different_field() -> VortexResult<()> {
        let arr = make_object_array()?;
        let result = apply_variant_get(&arr, "b")?;

        assert_eq!(result.len(), 3);

        // Row 0: {"a": 1, "b": "x"} → variant("x")
        assert!(!result.scalar_at(0)?.is_null());

        // Row 1: {"a": 2, "c": true} → null (no "b")
        assert!(result.scalar_at(1)?.is_null());

        // Row 2: {"b": "y"} → variant("y")
        assert!(!result.scalar_at(2)?.is_null());

        Ok(())
    }
}
