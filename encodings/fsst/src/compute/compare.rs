// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::FSST;
use crate::FSSTArrayExt;
impl CompareKernel for FSST {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match rhs.as_constant() {
            Some(constant) => compare_fsst_constant(lhs, &constant, operator, ctx),
            // Otherwise, fall back to the default comparison behavior.
            _ => Ok(None),
        }
    }
}

/// Specialized compare function implementation used when performing against a constant
fn compare_fsst_constant(
    left: ArrayView<'_, FSST>,
    right: &Scalar,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let is_rhs_empty = match right.dtype() {
        DType::Binary(_) => right
            .as_binary()
            .is_empty()
            .vortex_expect("RHS should not be null"),
        DType::Utf8(_) => right
            .as_utf8()
            .is_empty()
            .vortex_expect("RHS should not be null"),
        _ => vortex_bail!("VarBinArray can only have type of Binary or Utf8"),
    };
    if is_rhs_empty {
        let buffer = match operator {
            // Every possible value is gte ""
            CompareOperator::Gte => BitBuffer::new_set(left.len()),
            // No value is lt ""
            CompareOperator::Lt => BitBuffer::new_unset(left.len()),
            _ => left
                .uncompressed_lengths()
                .binary(
                    ConstantArray::new(
                        Scalar::zero_value(left.uncompressed_lengths().dtype()),
                        left.uncompressed_lengths().len(),
                    )
                    .into_array(),
                    operator.into(),
                )?
                .execute(ctx)?,
        };

        return Ok(Some(
            BoolArray::new(
                buffer,
                left.array()
                    .validity()?
                    .union_nullability(right.dtype().nullability()),
            )
            .into_array(),
        ));
    }

    // The following section only supports Eq/NotEq
    if !matches!(operator, CompareOperator::Eq | CompareOperator::NotEq) {
        return Ok(None);
    }

    let compressor = left.compressor();
    let encoded_buffer = match left.dtype() {
        DType::Utf8(_) => {
            let value = right
                .as_utf8()
                .value()
                .vortex_expect("Expected non-null scalar");
            ByteBuffer::from(compressor.compress(value.as_bytes()))
        }
        DType::Binary(_) => {
            let value = right
                .as_binary()
                .value()
                .vortex_expect("Expected non-null scalar");
            ByteBuffer::from(compressor.compress(value.as_slice()))
        }
        _ => unreachable!("FSSTArray can only have string or binary data type"),
    };

    let encoded_scalar = Scalar::binary(
        encoded_buffer,
        left.dtype().nullability() | right.dtype().nullability(),
    );

    let rhs = ConstantArray::new(encoded_scalar, left.len());
    left.codes()
        .into_array()
        .binary(rhs.into_array(), Operator::from(operator))
        .map(Some)
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
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

    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_fsst() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let lhs = VarBinArray::from_iter(
            [
                Some("hello"),
                None,
                Some("world"),
                None,
                Some("this is a very long string"),
            ],
            DType::Utf8(Nullability::Nullable),
        );
        let compressor = fsst_train_compressor(&lhs);
        let len = lhs.len();
        let dtype = lhs.dtype().clone();
        let lhs = fsst_compress(lhs, len, &dtype, &compressor, &mut ctx);

        let rhs = ConstantArray::new("world", lhs.len());

        // Ensure fastpath for Eq exists, and returns correct answer
        let equals = lhs
            .clone()
            .into_array()
            .binary(rhs.clone().into_array(), Operator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();

        assert_eq!(equals.dtype(), &DType::Bool(Nullability::Nullable));

        assert_arrays_eq!(
            &equals,
            &BoolArray::from_iter([Some(false), None, Some(true), None, Some(false)])
        );

        // Ensure fastpath for Eq exists, and returns correct answer
        let not_equals = lhs
            .clone()
            .into_array()
            .binary(rhs.into_array(), Operator::NotEq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();

        assert_eq!(not_equals.dtype(), &DType::Bool(Nullability::Nullable));
        assert_arrays_eq!(
            &not_equals,
            &BoolArray::from_iter([Some(true), None, Some(false), None, Some(true)])
        );

        // Ensure null constants are handled correctly.
        let null_rhs =
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), lhs.len());
        let equals_null = lhs
            .clone()
            .into_array()
            .binary(null_rhs.clone().into_array(), Operator::Eq)
            .unwrap();
        assert_arrays_eq!(
            &equals_null,
            &BoolArray::from_iter([None::<bool>, None, None, None, None])
        );

        let noteq_null = lhs
            .into_array()
            .binary(null_rhs.into_array(), Operator::NotEq)
            .unwrap();
        assert_arrays_eq!(
            &noteq_null,
            &BoolArray::from_iter([None::<bool>, None, None, None, None])
        );
    }
}
