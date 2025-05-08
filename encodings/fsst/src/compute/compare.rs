use vortex_array::arrays::{BoolArray, BooleanBuffer, ConstantArray};
use vortex_array::compute::{
    CompareKernel, CompareKernelAdapter, Operator, compare, compare_lengths_to_empty,
};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayRef, ToCanonical, register_kernel};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::{FSSTArray, FSSTEncoding};

impl CompareKernel for FSSTEncoding {
    fn compare(
        &self,
        lhs: &FSSTArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        match rhs.as_constant() {
            Some(constant) => {
                compare_fsst_constant(lhs, &ConstantArray::new(constant, lhs.len()), operator)
            }
            // Otherwise, fall back to the default comparison behavior.
            _ => Ok(None),
        }
    }
}

register_kernel!(CompareKernelAdapter(FSSTEncoding).lift());

/// Specialized compare function implementation used when performing against a constant
fn compare_fsst_constant(
    left: &FSSTArray,
    right: &ConstantArray,
    operator: Operator,
) -> VortexResult<Option<ArrayRef>> {
    let rhs_scalar = right.scalar();
    let is_rhs_empty = match rhs_scalar.dtype() {
        DType::Binary(_) => rhs_scalar
            .as_binary()
            .is_empty()
            .vortex_expect("RHS should not be null"),
        DType::Utf8(_) => rhs_scalar
            .as_utf8()
            .is_empty()
            .vortex_expect("RHS should not be null"),
        _ => vortex_bail!("VarBinArray can only have type of Binary or Utf8"),
    };
    if is_rhs_empty {
        let buffer = match operator {
            // Every possible value is gte ""
            Operator::Gte => BooleanBuffer::new_set(left.len()),
            // No value is lt ""
            Operator::Lt => BooleanBuffer::new_unset(left.len()),
            _ => {
                let uncompressed_lengths = left.uncompressed_lengths().to_primitive()?;
                match_each_native_ptype!(uncompressed_lengths.ptype(), |$P| {
                    compare_lengths_to_empty(uncompressed_lengths.as_slice::<$P>().iter().copied(), operator)
                })
            }
        };

        return Ok(Some(
            BoolArray::new(
                buffer,
                Validity::copy_from_array(left)?.union_nullability(right.dtype().nullability()),
            )
            .into_array(),
        ));
    }

    // The following section only supports Eq/NotEq
    if !matches!(operator, Operator::Eq | Operator::NotEq) {
        return Ok(None);
    }

    let compressor = fsst::Compressor::rebuild_from(left.symbols(), left.symbol_lengths());

    let encoded_buffer = match left.dtype() {
        DType::Utf8(_) => {
            let value = right
                .scalar()
                .as_utf8()
                .value()
                .vortex_expect("Expected non-null scalar");
            ByteBuffer::from(compressor.compress(value.as_bytes()))
        }
        DType::Binary(_) => {
            let value = right
                .scalar()
                .as_binary()
                .value()
                .vortex_expect("Expected non-null scalar");
            ByteBuffer::from(compressor.compress(value.as_slice()))
        }
        _ => unreachable!("FSSTArray can only have string or binary data type"),
    };

    let encoded_scalar = Scalar::new(
        DType::Binary(left.dtype().nullability() | right.dtype().nullability()),
        encoded_buffer.into(),
    );

    let rhs = ConstantArray::new(encoded_scalar, left.len());
    compare(left.codes(), &rhs, operator).map(Some)
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{ConstantArray, VarBinArray};
    use vortex_array::compute::{Operator, compare};
    use vortex_array::{Array, ToCanonical};
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use crate::{fsst_compress, fsst_train_compressor};

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
        let compressor = fsst_train_compressor(&lhs).unwrap();
        let lhs = fsst_compress(&lhs, &compressor).unwrap();

        let rhs = ConstantArray::new("world", lhs.len());

        // Ensure fastpath for Eq exists, and returns correct answer
        let equals = compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .to_bool()
            .unwrap();

        assert_eq!(equals.dtype(), &DType::Bool(Nullability::Nullable));

        assert_eq!(
            equals.boolean_buffer().into_iter().collect::<Vec<_>>(),
            vec![false, false, true, false, false]
        );

        // Ensure fastpath for Eq exists, and returns correct answer
        let not_equals = compare(&lhs, &rhs, Operator::NotEq)
            .unwrap()
            .to_bool()
            .unwrap();

        assert_eq!(not_equals.dtype(), &DType::Bool(Nullability::Nullable));
        assert_eq!(
            not_equals.boolean_buffer().into_iter().collect::<Vec<_>>(),
            vec![true, true, false, true, true]
        );

        // Ensure null constants are handled correctly.
        let null_rhs =
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), lhs.len());
        let equals_null = compare(&lhs, &null_rhs, Operator::Eq).unwrap();
        for idx in 0..lhs.len() {
            assert!(equals_null.scalar_at(idx).unwrap().is_null());
        }

        let noteq_null = compare(&lhs, &null_rhs, Operator::NotEq).unwrap();
        for idx in 0..lhs.len() {
            assert!(noteq_null.scalar_at(idx).unwrap().is_null());
        }
    }
}
