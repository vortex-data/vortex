use vortex_array::arrays::{BoolArray, BooleanBuffer, ConstantArray};
use vortex_array::compute::{CompareFn, Operator, compare, compare_lengths_to_empty};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::{FSSTArray, FSSTEncoding};

impl CompareFn<&FSSTArray> for FSSTEncoding {
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

    let built_symbols: &[u64] = unsafe { std::mem::transmute(compressor.symbol_table()) };
    let expected_symbols: &[u64] = unsafe { std::mem::transmute(symbols.as_slice()) };
    assert_eq!(
        built_symbols, expected_symbols,
        "built symbol table does not match expected"
    );
    assert_eq!(compressor.symbol_lengths(), symbol_lens.as_slice());

    let encoded_scalar = match left.dtype() {
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

    let rhs = ConstantArray::new(encoded_scalar, left.len());
    compare(left.codes(), &rhs, operator).map(Some)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use fsst::{CompressorBuilder, Symbol};
    use vortex_array::arrays::{ConstantArray, VarBinArray};
    use vortex_array::compute::{Operator, compare, scalar_at};
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
    fn test_symbols_bad() {
        let symbols: &[u64] = &[
            24931, 25698, 25442, 25699, 25186, 25444, 24932, 25188, 25185, 25441, 25697, 25700,
            24929, 24930, 25443, 25187, 6513249, 6512995, 6578786, 6513761, 6513507, 6382434,
            6579042, 6512994, 6447460, 6447969, 6382178, 6579041, 6512993, 6448226, 6513250,
            6579297, 6513506, 6447459, 6513764, 6447458, 6578529, 6382180, 6513762, 6447714,
            6579299, 6513508, 6382436, 6513763, 6578532, 6381924, 6448228, 6579300, 6381921,
            6382690, 6382179, 6447713, 6447972, 6513505, 6447457, 6382692, 6513252, 6578785,
            6578787, 6578531, 6448225, 6382177, 6382433, 6578530, 6448227, 6381922, 6578788,
            6579044, 6382691, 6512996, 6579043, 6579298, 6447970, 6447716, 6447971, 6381923,
            6447715, 97, 98, 100, 99, 97, 98, 99, 100,
        ];
        let lens: Vec<u8> = iter::repeat_n(2u8, 16)
            .chain(iter::repeat_n(3u8, 61))
            .chain(iter::repeat_n(1u8, 8))
            .collect();

        let mut builder = CompressorBuilder::new();
        for (symbol, len) in symbols.iter().zip(lens.iter()) {
            let symbol = Symbol::from_slice(&symbol.to_le_bytes());
            builder.insert(symbol, *len as usize);
        }
        let compressor = builder.build();
        let built_symbols: &[u64] = unsafe { std::mem::transmute(compressor.symbol_table()) };
        assert_eq!(built_symbols, symbols);
    }
}
