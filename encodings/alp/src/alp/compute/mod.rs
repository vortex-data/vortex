mod compare;

use std::fmt::Debug;

use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{
    BetweenFn, BetweenOptions, CompareFn, FilterFn, ScalarAtFn, SliceFn, StrictComparison, TakeFn,
    between, filter, scalar_at, slice, take,
};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayRef};
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::{Scalar, ScalarType};

use crate::{ALPArray, ALPEncoding, ALPFloat, match_each_alp_float_ptype};

impl ComputeVTable for ALPEncoding {
    fn between_fn(&self) -> Option<&dyn BetweenFn<&dyn Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<&dyn Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}

impl ScalarAtFn<&ALPArray> for ALPEncoding {
    fn scalar_at(&self, array: &ALPArray, index: usize) -> VortexResult<Scalar> {
        if !array.encoded().is_valid(index)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        if let Some(patches) = array.patches() {
            if let Some(patch) = patches.get_patched(index)? {
                return patch.cast(array.dtype());
            }
        }

        let encoded_val = scalar_at(array.encoded(), index)?;

        Ok(match_each_alp_float_ptype!(array.ptype(), |$T| {
            let encoded_val: <$T as ALPFloat>::ALPInt = encoded_val.as_ref().try_into().unwrap();
            Scalar::primitive(<$T as ALPFloat>::decode_single(
                encoded_val,
                array.exponents(),
            ), array.dtype().nullability())
        }))
    }
}

impl TakeFn<&ALPArray> for ALPEncoding {
    fn take(&self, array: &ALPArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_encoded = take(array.encoded(), indices)?;
        let taken_patches = array
            .patches()
            .map(|p| p.take(indices))
            .transpose()?
            .flatten()
            .map(|p| {
                p.cast_values(
                    &array
                        .dtype()
                        .with_nullability(taken_encoded.dtype().nullability()),
                )
            })
            .transpose()?;
        Ok(ALPArray::try_new(taken_encoded, array.exponents(), taken_patches)?.into_array())
    }
}

impl SliceFn<&ALPArray> for ALPEncoding {
    fn slice(&self, array: &ALPArray, start: usize, end: usize) -> VortexResult<ArrayRef> {
        Ok(ALPArray::try_new(
            slice(array.encoded(), start, end)?,
            array.exponents(),
            array
                .patches()
                .map(|p| p.slice(start, end))
                .transpose()?
                .flatten(),
        )?
        .into_array())
    }
}

impl FilterFn<&ALPArray> for ALPEncoding {
    fn filter(&self, array: &ALPArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let patches = array
            .patches()
            .map(|p| p.filter(mask))
            .transpose()?
            .flatten();

        Ok(
            ALPArray::try_new(filter(array.encoded(), mask)?, array.exponents(), patches)?
                .into_array(),
        )
    }
}

impl BetweenFn<&ALPArray> for ALPEncoding {
    fn between(
        &self,
        array: &ALPArray,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant()) else {
            return Ok(None);
        };

        if array.patches().is_some() {
            return Ok(None);
        }

        match_each_alp_float_ptype!(array.ptype(), |$F| {
            between_impl::<$F>(array, $F::try_from(lower)?, $F::try_from(upper)?, options)
        })
        .map(Some)
    }
}

fn between_impl<T: NativePType + ALPFloat>(
    array: &ALPArray,
    lower: T,
    upper: T,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef>
where
    Scalar: From<T::ALPInt>,
    <T as ALPFloat>::ALPInt: ScalarType + Debug,
{
    let exponents = array.exponents();

    // There are always compared
    // the below bound is `value {< | <=} x`, either value encodes into the ALPInt domain
    // in which case we can leave the comparison unchanged `enc(value) {< | <=} x` or it doesn't
    // and we encode into value below enc_below(value) < value < x, in which case the comparison
    // becomes enc(value) < x. See `alp_scalar_compare` for more details.
    // note that if the value doesn't encode than value != x, so must use strict comparison.
    let (lower_enc, lower_strict) = T::encode_single(lower, exponents)
        .map(|x| (x, options.lower_strict))
        .unwrap_or_else(|| (T::encode_below(lower, exponents), StrictComparison::Strict));

    // the upper value `x { < | <= } value` similarly encodes or `x < value < enc_above(value())`
    let (upper_enc, upper_strict) = T::encode_single(upper, exponents)
        .map(|x| (x, options.upper_strict))
        .unwrap_or_else(|| (T::encode_above(upper, exponents), StrictComparison::Strict));

    let options = BetweenOptions {
        lower_strict,
        upper_strict,
    };

    between(
        array.encoded(),
        &ConstantArray::new(lower_enc, array.len()),
        &ConstantArray::new(upper_enc, array.len()),
        &options,
    )
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::{BetweenOptions, StrictComparison};

    use crate::ALPArray;
    use crate::alp::compute::between_impl;

    fn between_test(arr: &ALPArray, lower: f32, upper: f32, options: &BetweenOptions) -> bool {
        let res = between_impl(arr, lower, upper, options)
            .unwrap()
            .to_bool()
            .unwrap()
            .boolean_buffer()
            .iter()
            .collect_vec();
        assert_eq!(res.len(), 1);

        res[0]
    }

    #[test]
    fn comparison_range() {
        let value = 0.0605_f32;
        let array = PrimitiveArray::from_iter([value; 1]);
        let encoded = crate::alp::compress::alp_encode(&array).unwrap();
        assert!(encoded.patches().is_none());
        assert_eq!(
            encoded.encoded().to_primitive().unwrap().as_slice::<i32>(),
            vec![605; 1]
        );

        assert!(between_test(
            &encoded,
            0.0605_f32,
            0.0605,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        ));

        assert!(!between_test(
            &encoded,
            0.0605_f32,
            0.0605,
            &BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
        ));

        assert!(!between_test(
            &encoded,
            0.0605_f32,
            0.0605,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
        ));

        assert!(between_test(
            &encoded,
            0.060499_f32,
            0.06051,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        ));

        assert!(between_test(
            &encoded,
            0.06_f32,
            0.06051,
            &BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
        ))
    }
}
