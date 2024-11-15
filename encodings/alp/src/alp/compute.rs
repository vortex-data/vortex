use vortex_array::array::{BoolArray, ConstantArray};
use vortex_array::compute::unary::{scalar_at, scalar_at_unchecked, ScalarAtFn};
use vortex_array::compute::{
    compare, filter, slice, take, ArrayCompute, FilterFn, MaybeCompareFn, Operator, SliceFn, TakeFn,
};
use vortex_array::stats::{ArrayStatistics, Stat};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::{PValue, Scalar};

use crate::{match_each_alp_float_ptype, ALPArray, ALPFloat};

impl ArrayCompute for ALPArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }

    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }

    fn filter(&self) -> Option<&dyn FilterFn> {
        Some(self)
    }
}

impl ScalarAtFn for ALPArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(self.scalar_at_unchecked(index))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        if let Some(patches) = self.patches() {
            if patches.with_dyn(|a| a.is_valid(index)) {
                // We need to make sure the value is actually in the patches array
                return scalar_at_unchecked(&patches, index);
            }
        }

        let encoded_val = scalar_at_unchecked(self.encoded(), index);

        match_each_alp_float_ptype!(self.ptype(), |$T| {
            let encoded_val: <$T as ALPFloat>::ALPInt = encoded_val.as_ref().try_into().unwrap();
            Scalar::primitive(<$T as ALPFloat>::decode_single(
                encoded_val,
                self.exponents(),
            ), self.dtype().nullability())
        })
    }
}

impl TakeFn for ALPArray {
    fn take(&self, indices: &ArrayData) -> VortexResult<ArrayData> {
        // TODO(ngates): wrap up indices in an array that caches decompression?
        Ok(Self::try_new(
            take(self.encoded(), indices)?,
            self.exponents(),
            self.patches().map(|p| take(&p, indices)).transpose()?,
        )?
        .into_array())
    }
}

impl SliceFn for ALPArray {
    fn slice(&self, start: usize, end: usize) -> VortexResult<ArrayData> {
        Ok(Self::try_new(
            slice(self.encoded(), start, end)?,
            self.exponents(),
            self.patches().map(|p| slice(&p, start, end)).transpose()?,
        )?
        .into_array())
    }
}

impl FilterFn for ALPArray {
    fn filter(&self, predicate: &ArrayData) -> VortexResult<ArrayData> {
        Ok(Self::try_new(
            filter(self.encoded(), predicate)?,
            self.exponents(),
            self.patches().map(|p| filter(&p, predicate)).transpose()?,
        )?
        .into_array())
    }
}

impl MaybeCompareFn for ALPArray {
    fn maybe_compare(
        &self,
        array: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        if ConstantArray::try_from(array).is_ok()
            || array
                .statistics()
                .get_as::<bool>(Stat::IsConstant)
                .unwrap_or_default()
        {
            let rhs = scalar_at(array, 0).vortex_expect("should be scalar");
            let pvalue = rhs
                .value()
                .as_pvalue()
                .vortex_expect("Expected primitive value");

            match pvalue {
                Some(PValue::F32(f)) => Some(alp_scalar_compare(self, f, operator)),
                Some(PValue::F64(f)) => Some(alp_scalar_compare(self, f, operator)),
                Some(_) | None => Some(Ok(BoolArray::from_vec(
                    vec![false; self.len()],
                    Validity::AllValid,
                )
                .into_array())),
            }
        } else {
            None
        }
    }
}

fn alp_scalar_compare<F: ALPFloat + Into<Scalar>>(
    alp: &ALPArray,
    value: F,
    operator: Operator,
) -> VortexResult<ArrayData>
where
    F::ALPInt: Into<Scalar>,
{
    let encoded = F::encode_single(value, alp.exponents());
    match encoded {
        Ok(encoded) => {
            let s = ConstantArray::new(encoded, alp.len());
            compare(alp.encoded(), s.as_ref(), operator)
        }
        Err(exception) => {
            if let Some(patches) = alp.patches().as_ref() {
                let s = ConstantArray::new(exception, alp.len());
                compare(patches, s.as_ref(), operator)
            } else {
                Ok(BoolArray::from_vec(vec![false; alp.len()], Validity::AllValid).into_array())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::array::PrimitiveArray;
    use vortex_array::IntoArrayVariant;
    use vortex_dtype::{DType, Nullability, PType};

    use super::*;
    use crate::alp_encode;

    #[test]
    fn basic_comparison_test() {
        let array = PrimitiveArray::from(vec![1.234f32; 1025]);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().as_primitive().maybe_null_slice::<i32>(),
            vec![1234; 1025]
        );

        let r = alp_scalar_compare(&encoded, 1.3_f32, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        for v in r.boolean_buffer().iter() {
            assert!(!v);
        }

        let r = alp_scalar_compare(&encoded, 1.234f32, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        for v in r.boolean_buffer().iter() {
            assert!(v);
        }
    }

    #[test]
    fn compare_with_patches() {
        let array =
            PrimitiveArray::from(vec![1.234f32, 1.5, 19.0, std::f32::consts::E, 1_000_000.9]);
        let encoded = alp_encode(&array).unwrap();
        assert!(encoded.patches().is_some());

        let r = alp_scalar_compare(&encoded, 1_000_000.9_f32, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        let buffer = r.boolean_buffer();
        assert!(buffer.value(buffer.len() - 1));
    }

    #[test]
    fn compare_to_null() {
        let array = PrimitiveArray::from(vec![1.234f32; 1025]);
        let encoded = alp_encode(&array).unwrap();

        let other = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::F32, Nullability::Nullable)),
            array.len(),
        );

        let r = encoded
            .maybe_compare(other.as_ref(), Operator::Eq)
            .unwrap()
            .unwrap()
            .into_bool()
            .unwrap();

        for v in r.boolean_buffer().iter() {
            assert!(!v);
        }
    }
}
