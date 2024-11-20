use fsst::Symbol;
use vortex_array::array::{varbin_scalar, ConstantArray};
use vortex_array::compute::unary::{scalar_at_unchecked, ScalarAtFn};
use vortex_array::compute::{
    compare, filter, slice, take, ArrayCompute, ComputeVTable, FilterFn, FilterMask,
    MaybeCompareFn, Operator, SliceFn, TakeFn, TakeOptions,
};
use vortex_array::{ArrayDType, ArrayData, ArrayLen, IntoArrayData, IntoArrayVariant, ToArrayData};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult, VortexUnwrap};
use vortex_scalar::Scalar;

use crate::{FSSTArray, FSSTEncoding};

impl ArrayCompute for FSSTArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }
}

impl ComputeVTable for FSSTEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl MaybeCompareFn for FSSTArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        match (other.as_constant(), operator) {
            (Some(constant_array), Operator::Eq | Operator::NotEq) => Some(compare_fsst_constant(
                self,
                &ConstantArray::new(constant_array, self.len()),
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
            Ok(right.to_array())
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

impl SliceFn<FSSTArray> for FSSTEncoding {
    fn slice(&self, array: &FSSTArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        // Slicing an FSST array leaves the symbol table unmodified,
        // only slicing the `codes` array.
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols(),
            array.symbol_lengths(),
            slice(array.codes(), start, stop)?,
            slice(array.uncompressed_lengths(), start, stop)?,
        )?
        .into_array())
    }
}

impl TakeFn<FSSTArray> for FSSTEncoding {
    // Take on an FSSTArray is a simple take on the codes array.
    fn take(
        &self,
        array: &FSSTArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols(),
            array.symbol_lengths(),
            take(array.codes(), indices, options)?,
            take(array.uncompressed_lengths(), indices, options)?,
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

impl FilterFn<FSSTArray> for FSSTEncoding {
    // Filtering an FSSTArray filters the codes array, leaving the symbols array untouched
    fn filter(&self, array: &FSSTArray, mask: FilterMask) -> VortexResult<ArrayData> {
        Ok(FSSTArray::try_new(
            array.dtype().clone(),
            array.symbols(),
            array.symbol_lengths(),
            filter(&array.codes(), mask.clone())?,
            filter(&array.uncompressed_lengths(), mask)?,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::{ConstantArray, VarBinArray};
    use vortex_array::compute::unary::scalar_at_unchecked;
    use vortex_array::compute::{MaybeCompareFn, Operator};
    use vortex_array::{ArrayLen, IntoArrayData, IntoArrayVariant};
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
