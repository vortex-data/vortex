use fsst::Symbol;
use vortex_array::array::{varbin_scalar, ConstantArray};
use vortex_array::compute::unary::{scalar_at_unchecked, ScalarAtFn};
use vortex_array::compute::{
    compare, filter, slice, take, ArrayCompute, FilterFn, MaybeCompareFn, Operator, SliceFn, TakeFn,
};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult, VortexUnwrap};
use vortex_scalar::Scalar;

use crate::FSSTArray;

impl ArrayCompute for FSSTArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }

    fn filter(&self) -> Option<&dyn FilterFn> {
        Some(self)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl MaybeCompareFn for FSSTArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        match (ConstantArray::try_from(other), operator) {
            (Ok(constant_array), Operator::Eq | Operator::NotEq) => Some(compare_fsst_constant(
                self,
                &constant_array,
                operator == Operator::Eq,
            )),
            _ => None,
        }
    }
}

/// Specialized compare function implementation used when performing equals or not equals against
/// a constant.
fn compare_fsst_constant(
    left: &FSSTArray,
    right: &ConstantArray,
    equal: bool,
) -> VortexResult<ArrayData> {
    let symbols = left.symbols().into_primitive()?;
    let symbols_u64 = symbols.maybe_null_slice::<u64>();

    let symbol_lens = left.symbol_lengths().into_primitive()?;
    let symbol_lens_u8 = symbol_lens.maybe_null_slice::<u8>();

    let mut compressor = fsst::CompressorBuilder::new();
    for (symbol, symbol_len) in symbols_u64.iter().zip(symbol_lens_u8.iter()) {
        compressor.insert(Symbol::from_slice(&symbol.to_le_bytes()), *symbol_len as _);
    }
    let compressor = compressor.build();

    let encoded_scalar = match left.dtype() {
        DType::Utf8(_) => right
            .scalar_value()
            .as_buffer_string()?
            .map(|scalar| Buffer::from(compressor.compress(scalar.as_bytes()))),
        DType::Binary(_) => right
            .scalar_value()
            .as_buffer()?
            .map(|scalar| Buffer::from(compressor.compress(scalar.as_slice()))),

        _ => unreachable!("FSSTArray can only have string or binary data type"),
    };

    match encoded_scalar {
        None => {
            // Eq and NotEq on null values yield nulls, per the Arrow behavior.
            Ok(right.clone().into_array())
        }
        Some(encoded_scalar) => {
            let rhs = ConstantArray::new(encoded_scalar, left.len());

            compare(
                left.codes(),
                rhs,
                if equal { Operator::Eq } else { Operator::NotEq },
            )
        }
    }
}

impl SliceFn for FSSTArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        // Slicing an FSST array leaves the symbol table unmodified,
        // only slicing the `codes` array.
        Ok(Self::try_new(
            self.dtype().clone(),
            self.symbols(),
            self.symbol_lengths(),
            slice(self.codes(), start, stop)?,
            slice(self.uncompressed_lengths(), start, stop)?,
        )?
        .into_array())
    }
}

impl TakeFn for FSSTArray {
    // Take on an FSSTArray is a simple take on the codes array.
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        Ok(Self::try_new(
            self.dtype().clone(),
            self.symbols(),
            self.symbol_lengths(),
            take(self.codes(), indices)?,
            take(self.uncompressed_lengths(), indices)?,
        )?
        .into_array())
    }
}

impl ScalarAtFn for FSSTArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let compressed = scalar_at_unchecked(self.codes(), index);
        let binary_datum = compressed
            .value()
            .as_buffer()?
            .ok_or_else(|| vortex_err!("Expected a binary scalar, found {}", compressed.dtype()))?;

        self.with_decompressor(|decompressor| {
            let decoded_buffer: Buffer = decompressor.decompress(binary_datum.as_slice()).into();
            Ok(varbin_scalar(decoded_buffer, self.dtype()))
        })
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        <Self as ScalarAtFn>::scalar_at(self, index).vortex_unwrap()
    }
}

impl FilterFn for FSSTArray {
    // Filtering an FSSTArray filters the codes array, leaving the symbols array untouched
    fn filter(&self, predicate: &ArrayData) -> VortexResult<ArrayData> {
        Ok(Self::try_new(
            self.dtype().clone(),
            self.symbols(),
            self.symbol_lengths(),
            filter(self.codes(), predicate)?,
            filter(self.uncompressed_lengths(), predicate)?,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::{ConstantArray, VarBinArray};
    use vortex_array::compute::unary::scalar_at_unchecked;
    use vortex_array::compute::{MaybeCompareFn, Operator};
    use vortex_array::{IntoArrayData, IntoArrayVariant};
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
        )
        .into_array();
        let compressor = fsst_train_compressor(&lhs).unwrap();
        let lhs = fsst_compress(&lhs, &compressor).unwrap();

        let rhs = ConstantArray::new("world", lhs.len()).into_array();

        // Ensure fastpath for Eq exists, and returns correct answer
        let equals: Vec<bool> = MaybeCompareFn::maybe_compare(&lhs, &rhs, Operator::Eq)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .into_iter()
            .collect();

        assert_eq!(equals, vec![false, false, true, false, false]);

        // Ensure fastpath for Eq exists, and returns correct answer
        let not_equals: Vec<bool> = MaybeCompareFn::maybe_compare(&lhs, &rhs, Operator::NotEq)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap()
            .boolean_buffer()
            .into_iter()
            .collect();

        assert_eq!(not_equals, vec![true, true, false, true, true]);

        // Ensure null constants are handled correctly.
        let null_rhs =
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), lhs.len());
        let equals_null = MaybeCompareFn::maybe_compare(&lhs, null_rhs.as_ref(), Operator::Eq)
            .unwrap()
            .unwrap();
        for idx in 0..lhs.len() {
            assert!(scalar_at_unchecked(&equals_null, idx).is_null());
        }

        let noteq_null = MaybeCompareFn::maybe_compare(&lhs, null_rhs.as_ref(), Operator::NotEq)
            .unwrap()
            .unwrap();
        for idx in 0..lhs.len() {
            assert!(scalar_at_unchecked(&noteq_null, idx).is_null());
        }
    }
}
