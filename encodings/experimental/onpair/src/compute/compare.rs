// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::uncompressed_lengths::UncompressedLengthsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArraySlotsExt;

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
        let is_empty = match constant.dtype() {
            DType::Utf8(_) => constant.as_utf8().is_empty(),
            DType::Binary(_) => constant.as_binary().is_empty(),
            _ => return Ok(None),
        };
        if is_empty != Some(true) {
            return OnPair::compare_to_empty(lhs, operator, constant.dtype().nullability(), ctx)
                .map(Some);
        }

        if !matches!(operator, CompareOperator::Eq | CompareOperator::NotEq) {
            return Ok(None);
        }

        // This part is the issue
        let compressor = lhs.compressor();
        let encoded_buffer = match lhs.dtype() {
            DType::Utf8(_) => {
                let value = constant
                    .as_utf8()
                    .value()
                    .vortex_expect("Expected non-null scalar");
                ByteBuffer::from(compressor.compress(value.as_bytes()))
            }
            DType::Binary(_) => {
                let value = constant
                    .as_binary()
                    .value()
                    .vortex_expect("Expected non-null scalar");
                ByteBuffer::from(compressor.compress(value.as_slice()))
            }
            _ => unreachable!("OnPairArray can only have string or binary data type"),
        };

        let encoded_scalar = Scalar::binary(
            encoded_buffer,
            lhs.dtype().nullability() | constant.dtype().nullability(),
        );

        let rhs = ConstantArray::new(encoded_scalar, lhs.len());
        lhs.codes()
            .binary(rhs.into_array(), Operator::from(operator))
            .map(Some)
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
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::compress::DEFAULT_DICT12_CONFIG;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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
        let arr = onpair_compress(&input, input.len(), input.dtype(), DEFAULT_DICT12_CONFIG)?
            .into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let result = arr
            .binary(ConstantArray::new("", input.len()).into_array(), op)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(&result, &BoolArray::from_iter(expected));
        Ok(())
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn compare_empty_string_nullable() -> VortexResult<()> {
        let input = VarBinArray::from_iter(
            [Some(""), None, Some("x")],
            DType::Utf8(Nullability::Nullable),
        );
        let arr = onpair_compress(&input, input.len(), input.dtype(), DEFAULT_DICT12_CONFIG)?
            .into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let eq_empty = arr
            .clone()
            .binary(ConstantArray::new("", arr.len()).into_array(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(
            &eq_empty,
            &BoolArray::from_iter([Some(true), None, Some(false)])
        );

        let null_rhs =
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), arr.len());
        let eq_null = arr
            .binary(null_rhs.into_array(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(&eq_null, &BoolArray::from_iter([None::<bool>, None, None]));
        Ok(())
    }
}
