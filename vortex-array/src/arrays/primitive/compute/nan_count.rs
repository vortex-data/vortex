use vortex_dtype::{NativePType, match_each_float_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::compute::{NaNCountKernel, NaNCountKernelAdapter};
use crate::register_kernel;

impl NaNCountKernel for PrimitiveVTable {
    fn nan_count(&self, array: &PrimitiveArray) -> VortexResult<usize> {
        Ok(match_each_float_ptype!(array.ptype(), |$F| {
            compute_nan_count_with_validity(array.as_slice::<$F>(), array.validity_mask()?)
        }))
    }
}

register_kernel!(NaNCountKernelAdapter(PrimitiveVTable).lift());

#[inline]
fn compute_nan_count_with_validity<T: NativePType>(values: &[T], validity: Mask) -> usize {
    match validity {
        Mask::AllTrue(_) => values.iter().filter(|v| v.is_nan()).count(),
        Mask::AllFalse(_) => 0,
        Mask::Values(v) => values
            .iter()
            .zip(v.boolean_buffer().iter())
            .filter_map(|(v, m)| m.then_some(v))
            .filter(|v| v.is_nan())
            .count(),
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::arrays::PrimitiveArray;
    use crate::compute::nan_count;
    use crate::validity::Validity;

    #[test]
    fn primitive_nan_count() {
        let p = PrimitiveArray::new(
            buffer![
                -f32::NAN,
                f32::NAN,
                0.1,
                1.1,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY
            ],
            Validity::NonNullable,
        );
        assert_eq!(nan_count(p.as_ref()).unwrap(), 2);
    }
}
