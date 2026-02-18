// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::compute::Operator;
use vortex_array::compute::compare;
use vortex_array::compute::compare_lengths_to_empty;
use vortex_array::expr::CompareKernel;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::FSSTArray;
use crate::FSSTVTable;

impl CompareKernel for FSSTVTable {
    fn compare(
        lhs: &FSSTArray,
        rhs: &dyn Array,
        operator: Operator,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match rhs.as_constant() {
            Some(constant) => compare_fsst_constant(lhs, &constant, operator),
            // Otherwise, fall back to the default comparison behavior.
            _ => Ok(None),
        }
    }
}

/// Specialized compare function implementation used when performing against a constant
fn compare_fsst_constant(
    left: &FSSTArray,
    right: &Scalar,
    operator: Operator,
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
            Operator::Gte => BitBuffer::new_set(left.len()),
            // No value is lt ""
            Operator::Lt => BitBuffer::new_unset(left.len()),
            _ => {
                let uncompressed_lengths = left.uncompressed_lengths().to_primitive();
                match_each_integer_ptype!(uncompressed_lengths.ptype(), |P| {
                    compare_lengths_to_empty(
                        uncompressed_lengths.as_slice::<P>().iter().copied(),
                        operator,
                    )
                })
            }
        };

        return Ok(Some(
            BoolArray::new(
                buffer,
                Validity::copy_from_array(left.as_ref())?
                    .union_nullability(right.dtype().nullability()),
            )
            .into_array(),
        ));
    }

    // The following section only supports Eq/NotEq
    if !matches!(operator, Operator::Eq | Operator::NotEq) {
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
    compare(left.codes().as_ref(), rhs.as_ref(), operator).map(Some)
}

#[cfg(test)]
mod tests {
    use vortex_array::Array;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::Operator;
    use vortex_array::compute::compare;
    use vortex_array::scalar::Scalar;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;

    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_fsst() {
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
        let lhs = fsst_compress(lhs, &compressor);

        let rhs = ConstantArray::new("world", lhs.len());

        // Ensure fastpath for Eq exists, and returns correct answer
        let equals = compare(lhs.as_ref(), rhs.as_ref(), Operator::Eq)
            .unwrap()
            .to_bool();

        assert_eq!(equals.dtype(), &DType::Bool(Nullability::Nullable));

        assert_arrays_eq!(
            &equals,
            &BoolArray::from_iter([Some(false), None, Some(true), None, Some(false)])
        );

        // Ensure fastpath for Eq exists, and returns correct answer
        let not_equals = compare(lhs.as_ref(), rhs.as_ref(), Operator::NotEq)
            .unwrap()
            .to_bool();

        assert_eq!(not_equals.dtype(), &DType::Bool(Nullability::Nullable));
        assert_arrays_eq!(
            &not_equals,
            &BoolArray::from_iter([Some(true), None, Some(false), None, Some(true)])
        );

        // Ensure null constants are handled correctly.
        let null_rhs =
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), lhs.len());
        let equals_null = compare(lhs.as_ref(), null_rhs.as_ref(), Operator::Eq).unwrap();
        assert_arrays_eq!(
            &equals_null,
            &BoolArray::from_iter([None::<bool>, None, None, None, None])
        );

        let noteq_null = compare(lhs.as_ref(), null_rhs.as_ref(), Operator::NotEq).unwrap();
        assert_arrays_eq!(
            &noteq_null,
            &BoolArray::from_iter([None::<bool>, None, None, None, None])
        );
    }
}
