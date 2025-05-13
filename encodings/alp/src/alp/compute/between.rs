use std::fmt::Debug;

use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{
    BetweenKernel, BetweenKernelAdapter, BetweenOptions, StrictComparison, between,
};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_dtype::{NativePType, Nullability};
use vortex_error::VortexResult;
use vortex_scalar::{Scalar, ScalarType};

use crate::{ALPArray, ALPFloat, ALPVTable, match_each_alp_float_ptype};

impl BetweenKernel for ALPVTable {
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

        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();

        match_each_alp_float_ptype!(array.ptype(), |$F| {
            between_impl::<$F>(array, $F::try_from(lower)?, $F::try_from(upper)?, nullability, options)
        })
            .map(Some)
    }
}

register_kernel!(BetweenKernelAdapter(ALPVTable).lift());

fn between_impl<T: NativePType + ALPFloat>(
    array: &ALPArray,
    lower: T,
    upper: T,
    nullability: Nullability,
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
        ConstantArray::new(Scalar::primitive(lower_enc, nullability), array.len()).as_ref(),
        ConstantArray::new(Scalar::primitive(upper_enc, nullability), array.len()).as_ref(),
        &options,
    )
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::{BetweenOptions, StrictComparison};
    use vortex_dtype::Nullability;

    use crate::alp::compute::between::between_impl;
    use crate::{ALPArray, alp_encode};

    fn between_test(arr: &ALPArray, lower: f32, upper: f32, options: &BetweenOptions) -> bool {
        let res = between_impl(arr, lower, upper, Nullability::Nullable, options)
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
        let encoded = alp_encode(&array, None).unwrap();
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
