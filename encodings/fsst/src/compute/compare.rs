use vortex_array::arrays::{BoolArray, BooleanBuffer, ConstantArray};
use vortex_array::compute::{CompareFn, Operator, compare, compare_lengths_to_empty};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::{FSSTArray, FSSTEncoding};

impl CompareFn<&FSSTArray> for FSSTEncoding {
    fn compare(
        &self,
        lhs: &FSSTArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        match rhs.as_constant() {
            Some(constant) => compare_fsst_constant(lhs, constant, operator),
            // Otherwise, fall back to the default comparison behavior.
            _ => Ok(None),
        }
    }
}

/// Specialized compare function implementation used when performing against a constant
fn compare_fsst_constant(
    left: &FSSTArray,
    right: Scalar,
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
            BoolArray::new(buffer, Validity::copy_from_array(left)?).into_array(),
        ));
    }

    // The following section only supports Eq/NotEq
    if !matches!(operator, Operator::Eq | Operator::NotEq) {
        return Ok(None);
    }

    let symbols = left.symbols();
    let symbol_lens = left.symbol_lengths();

    let mut compressor = fsst::CompressorBuilder::new();
    for (symbol, symbol_len) in symbols.iter().zip(symbol_lens.iter()) {
        compressor.insert(*symbol, *symbol_len as usize);
    }
    let compressor = compressor.build();

    let encoded_scalar = match left.dtype() {
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

    let rhs = ConstantArray::new(encoded_scalar, left.len());
    compare(left.codes(), &rhs, operator).map(Some)
}

#[cfg(test)]
mod tests {
    use fsst::{CompressorBuilder, Decompressor, Symbol};
    use vortex_array::arrays::{ConstantArray, VarBinArray};
    use vortex_array::compute::{Operator, compare, scalar_at};
    use vortex_array::{Array, ToCanonical};
    use vortex_buffer::Buffer;
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
        let equals: Vec<bool> = compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .to_bool()
            .unwrap()
            .boolean_buffer()
            .into_iter()
            .collect();

        assert_eq!(equals, vec![false, false, true, false, false]);

        // Ensure fastpath for Eq exists, and returns correct answer
        let not_equals: Vec<bool> = compare(&lhs, &rhs, Operator::NotEq)
            .unwrap()
            .to_bool()
            .unwrap()
            .boolean_buffer()
            .into_iter()
            .collect();

        assert_eq!(not_equals, vec![true, true, false, true, true]);

        // Ensure null constants are handled correctly.
        let null_rhs =
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), lhs.len());
        let equals_null = compare(&lhs, &null_rhs, Operator::Eq).unwrap();
        for idx in 0..lhs.len() {
            assert!(scalar_at(&equals_null, idx).unwrap().is_null());
        }

        let noteq_null = compare(&lhs, &null_rhs, Operator::NotEq).unwrap();
        for idx in 0..lhs.len() {
            assert!(scalar_at(&noteq_null, idx).unwrap().is_null());
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_random_string() {
        let value = "AAtttttttttttHHHHHHHHHHHHHHttttttttttttttttttttttttttttttttt,tttttttttttttttttttttHHHHHH\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}0tttttttttttttttttttttttttHHHHHH\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}0\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\0\0\0\0\0\0\0\u{8}\u{18}\u{18}))))))\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\u{18}\u{18}\u{18}\u{18}HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHttttttttttttttttttttt|tttttttttttttttttttttttttttttttttHHHHHH\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}0\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\u{18}\0\0\0\0\0\0\0\u{8}\u{18}";
        let symbols = Buffer::from_iter(vec![
            Symbol::from_slice(&[116, 116, 0, 0, 0, 0, 0, 0]),
            Symbol::from_slice(&[24, 0, 0, 0, 0, 0, 0, 0]),
            Symbol::from_slice(&[116, 116, 116, 116, 116, 116, 116, 116]),
            Symbol::from_slice(&[24, 24, 24, 24, 24, 24, 24, 24]),
            Symbol::from_slice(&[116, 0, 0, 0, 0, 0, 0, 0]),
            Symbol::from_slice(&[24, 0, 0, 0, 0, 0, 0, 0]),
            Symbol::from_slice(&[72, 0, 0, 0, 0, 0, 0, 0]),
            Symbol::from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]),
            Symbol::from_slice(&[72, 0, 0, 0, 0, 0, 0, 0]),
        ]);
        let lengths = Buffer::from_iter(vec![2, 1, 8, 8, 1, 1, 1, 1, 1]);
        let mut compressor = CompressorBuilder::new();
        for (symbol, symbol_len) in symbols.iter().zip(lengths.iter()) {
            compressor.insert(*symbol, *symbol_len as usize);
        }
        let compressor = compressor.build();

        let compressed = compressor.compress(value.as_bytes());
        let expected = vec![
            255u8, 65, 255, 65, 2, 0, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2, 2, 2, 4,
            255, 44, 2, 2, 0, 0, 4, 8, 8, 8, 8, 8, 8, 3, 3, 1, 5, 255, 48, 2, 2, 2, 4, 8, 8, 8, 8,
            8, 8, 3, 3, 1, 5, 255, 48, 1, 1, 1, 5, 7, 7, 7, 7, 7, 7, 7, 255, 8, 1, 255, 41, 255,
            41, 255, 41, 255, 41, 255, 41, 255, 41, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 2, 2, 0, 0, 4, 255, 124, 2, 2, 2, 2, 4, 8, 8, 8, 8, 8, 8, 3, 3, 1, 5, 255, 48, 1,
            1, 1, 5, 7, 7, 7, 7, 7, 7, 7, 255, 8, 5,
        ];
        assert_eq!(Decompressor::new(&symbols, &lengths).decompress(&compressed), value.as_bytes());
        assert_eq!(compressed, expected);
    }
}
