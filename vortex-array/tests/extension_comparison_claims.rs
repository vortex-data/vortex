// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Comparison claims for extension types defined outside `vortex-array`.

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::lt;
    use vortex_array::expr::root;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::builtin::compare::Compare;
    use vortex_array::scalar_fn::builtin::operators::CompareOperator;
    use vortex_array::test_harness::ExpectedOutcome;
    use vortex_array::test_harness::assert_execution_paths;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_functions::between::BetweenOptions;
    use vortex_functions::between::StrictComparison;
    use vortex_functions::expr::between;
    use vortex_json::Json;
    use vortex_session::VortexSession;
    use vortex_tensor::vector::Vector;
    use vortex_tensor::vector::VectorMetadata;
    use vortex_uuid::Uuid;
    use vortex_uuid::UuidMetadata;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn json_dtype() -> VortexResult<DType> {
        Ok(DType::Extension(
            ExtDType::<Json>::try_new(EmptyMetadata, NonNullable)?.erased(),
        ))
    }

    fn vector_dtype() -> VortexResult<DType> {
        let metadata = VectorMetadata::try_new(PType::F32, 2)?;
        Ok(DType::Extension(
            ExtDType::<Vector>::try_new(metadata, NonNullable)?.erased(),
        ))
    }

    #[track_caller]
    fn assert_missing_claim(expr: Expression, dtype: &DType, function: &str, claim: &str) {
        let error = expr.bind(dtype).expect_err("binding requires a claim");
        let expected =
            format!("{function} requires the {claim} claim, but {dtype} does not hold it");
        assert_eq!(error.to_string().lines().next(), Some(expected.as_str()));
    }

    #[rstest]
    #[case::json(json_dtype, CompareOperator::Eq, "Eq")]
    #[case::vector(vector_dtype, CompareOperator::Lt, "Ord")]
    fn compare_rejects_types_without_claim(
        #[case] dtype: fn() -> VortexResult<DType>,
        #[case] operator: CompareOperator,
        #[case] claim: &str,
    ) -> VortexResult<()> {
        let expr = Compare.new_expr(operator, [root(), root()]);
        assert_missing_claim(expr, &dtype()?, &operator.to_string(), claim);

        Ok(())
    }

    #[test]
    fn vector_accepts_equality() -> VortexResult<()> {
        let result = eq(root(), root()).bind(&vector_dtype()?)?;
        assert_eq!(result.dtype(), &DType::Bool(NonNullable));

        Ok(())
    }

    #[test]
    fn compare_nested_extension_needs_claim() -> VortexResult<()> {
        let document = DType::struct_([("doc", json_dtype()?)], NonNullable);
        assert_missing_claim(eq(root(), root()), &document, "=", "Eq");

        let mut ctx = SESSION.create_execution_ctx();
        let lhs = TemporalArray::new_timestamp(
            buffer![1000i64, 2000].into_array(),
            TimeUnit::Milliseconds,
            None,
        );
        let rhs = TemporalArray::new_timestamp(
            buffer![2000i64, 1000].into_array(),
            TimeUnit::Milliseconds,
            None,
        );
        let input = StructArray::from_fields(&[
            (
                "lhs",
                StructArray::from_fields(&[("ts", lhs.into_array())])?.into_array(),
            ),
            (
                "rhs",
                StructArray::from_fields(&[("ts", rhs.into_array())])?.into_array(),
            ),
        ])?
        .into_array();

        let result = input.apply(&lt(col("lhs"), col("rhs")))?;
        assert_arrays_eq!(result, BoolArray::from_iter([true, false]), &mut ctx);

        Ok(())
    }

    #[test]
    fn compare_accepts_claimed_types_on_every_path() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = ExtDType::<Uuid>::try_new(UuidMetadata::default(), NonNullable)?.erased();
        let bytes = Buffer::from_iter([0u8; 16].into_iter().chain([1u8; 16])).into_array();
        let storage = FixedSizeListArray::try_new(bytes, 16, Validity::NonNullable, 2)?;
        let lhs = ExtensionArray::new(dtype, storage.into_array()).into_array();
        let rhs = ConstantArray::new(lhs.execute_scalar(0, &mut ctx)?, lhs.len()).into_array();

        assert_execution_paths(
            &[lhs, rhs],
            |inputs| inputs[0].compare(inputs[1].clone(), CompareOperator::Eq),
            ExpectedOutcome::Array(BoolArray::from_iter([true, false]).into_array()),
            &mut ctx,
        );

        Ok(())
    }

    #[test]
    fn between_requires_ord() -> VortexResult<()> {
        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };
        let expr = between(root(), root(), root(), options);
        assert_missing_claim(expr, &vector_dtype()?, "between", "Ord");

        Ok(())
    }
}
