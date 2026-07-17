// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use onpair::search;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::array::dict_view;
use crate::decode::collect_codes_window;

impl CompareKernel for OnPair {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let needle = match constant.dtype() {
            DType::Utf8(_) => constant
                .as_utf8()
                .value()
                .map(|value| value.as_bytes().to_vec()),
            DType::Binary(_) => constant
                .as_binary()
                .value()
                .map(|value| value.as_slice().to_vec()),
            _ => return Ok(None),
        };
        let Some(needle) = needle else {
            return Ok(None);
        };

        let buffer = if needle.is_empty() {
            let lengths = lhs.uncompressed_lengths();
            match operator {
                // every value is greater than an empty string
                CompareOperator::Gte => BitBuffer::new_set(lhs.len()),
                // no value is less than an empty string
                CompareOperator::Lt => BitBuffer::new_unset(lhs.len()),
                _ => lengths
                    .binary(
                        ConstantArray::new(Scalar::zero_value(lengths.dtype()), lengths.len())
                            .into_array(),
                        operator.into(),
                    )?
                    .execute(ctx)?,
            }
        } else {
            if !matches!(operator, CompareOperator::Eq | CompareOperator::NotEq) {
                return Ok(None);
            }

            let dict = dict_view(lhs, ctx)?;
            let query = search::tokenize(&needle, dict);
            let window = collect_codes_window(lhs, ctx)?;

            let negated = operator == CompareOperator::NotEq;
            BitBuffer::collect_bool(lhs.len(), |i| {
                search::equals(window.row(i), &query) != negated
            })
        };

        Ok(Some(
            BoolArray::new(
                buffer,
                lhs.validity()?
                    .union_nullability(constant.dtype().nullability()),
            )
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::OnPair;
    use crate::compress::DEFAULT_DICT12_CONFIG;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[cfg_attr(miri, ignore)]
    #[rstest]
    #[case(Operator::Eq, [true, false, true, false])]
    #[case(Operator::NotEq, [false, true, false, true])]
    #[case(Operator::Gt, [false, true, false, true])]
    #[case(Operator::Gte, [true, true, true, true])]
    #[case(Operator::Lt, [false, false, false, false])]
    #[case(Operator::Lte, [true, false, true, false])]
    fn compare_empty_string(#[case] op: Operator, #[case] expected: [bool; 4]) -> VortexResult<()> {
        let input = VarBinArray::from_iter(
            [Some(""), Some("a"), Some(""), Some("bbb")],
            DType::Utf8(Nullability::NonNullable),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let arr = onpair_compress(input.as_array(), DEFAULT_DICT12_CONFIG, &mut ctx)?.into_array();

        let result = arr
            .binary(ConstantArray::new("", input.len()).into_array(), op)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn compare_empty_string_nullable() -> VortexResult<()> {
        let input = VarBinArray::from_iter(
            [Some(""), None, Some("x")],
            DType::Utf8(Nullability::Nullable),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let arr = onpair_compress(input.as_array(), DEFAULT_DICT12_CONFIG, &mut ctx)?.into_array();

        let eq_empty = arr
            .clone()
            .binary(ConstantArray::new("", arr.len()).into_array(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(
            &eq_empty,
            &BoolArray::from_iter([Some(true), None, Some(false)]),
            &mut ctx
        );

        let null_rhs =
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), arr.len());
        let eq_null = arr
            .binary(null_rhs.into_array(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(
            &eq_null,
            &BoolArray::from_iter([None::<bool>, None, None]),
            &mut ctx
        );
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn compare_nonempty_string_nullable() -> VortexResult<()> {
        let input = VarBinArray::from_iter(
            [Some("hello"), None, Some("world"), Some("hello")],
            DType::Utf8(Nullability::Nullable),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let arr = onpair_compress(input.as_array(), DEFAULT_DICT12_CONFIG, &mut ctx)?.into_array();
        let rhs = ConstantArray::new("hello", arr.len()).into_array();

        let eq = arr
            .binary(rhs.clone(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(
            &eq,
            &BoolArray::from_iter([Some(true), None, Some(false), Some(true)]),
            &mut ctx
        );

        let neq = arr
            .binary(rhs, Operator::NotEq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(
            &neq,
            &BoolArray::from_iter([Some(false), None, Some(true), Some(false)]),
            &mut ctx
        );
        Ok(())
    }

    /// Call `CompareKernel::compare` directly and verify it returns `Some`
    /// (i.e. the kernel handles the constant needle rather than silently
    /// falling back to canonical decompression), and that ordering against a
    /// non-empty needle declines — code sequences are not order-preserving.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn compare_kernel_handles_constant_needle() -> VortexResult<()> {
        let input = VarBinArray::from_iter(
            [Some("hello"), Some("world"), Some("hello"), Some("he")],
            DType::Utf8(Nullability::NonNullable),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let arr = onpair_compress(input.as_array(), DEFAULT_DICT12_CONFIG, &mut ctx)?
            .try_downcast::<OnPair>()
            .map_err(|array| vortex_err!("expected OnPair array, got {}", array.encoding_id()))?;
        let rhs = ConstantArray::new("hello", arr.len()).into_array();

        let eq =
            <OnPair as CompareKernel>::compare(arr.as_view(), &rhs, CompareOperator::Eq, &mut ctx)?
                .expect("OnPair CompareKernel should handle a constant needle");
        assert_arrays_eq!(
            eq,
            BoolArray::from_iter([true, false, true, false]),
            &mut ctx
        );

        let lt =
            <OnPair as CompareKernel>::compare(arr.as_view(), &rhs, CompareOperator::Lt, &mut ctx)?;
        assert!(
            lt.is_none(),
            "ordering compare against a non-empty needle must fall back to canonical"
        );
        Ok(())
    }

    /// The equality path resolves row windows relative to `codes_offsets[0]`,
    /// which is nonzero for a sliced array. Exercise the kernel directly on a
    /// slice so the windowed row arithmetic is covered.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn compare_kernel_on_sliced_array() -> VortexResult<()> {
        let input = VarBinArray::from_iter(
            [
                Some("aardvark"),
                Some("hello"),
                Some("world"),
                Some("hello"),
                Some("zebra"),
            ],
            DType::Utf8(Nullability::NonNullable),
        );
        let mut ctx = SESSION.create_execution_ctx();
        let arr = onpair_compress(input.as_array(), DEFAULT_DICT12_CONFIG, &mut ctx)?.into_array();
        let sliced = arr.slice(1..4)?;
        assert!(sliced.is::<OnPair>(), "slice dropped OnPair encoding");
        let sliced = sliced
            .try_downcast::<OnPair>()
            .map_err(|_| vortex_err!("sliced array was not OnPair"))?;

        let rhs = ConstantArray::new("hello", sliced.len()).into_array();
        let eq = <OnPair as CompareKernel>::compare(
            sliced.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut ctx,
        )?
        .expect("OnPair CompareKernel should handle a constant needle");
        assert_arrays_eq!(eq, BoolArray::from_iter([true, false, true]), &mut ctx);
        Ok(())
    }
}
