use vortex_dtype::match_each_integer_ptype;
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::arrays::NullVTable;
use crate::arrays::null::NullArray;
use crate::compute::{
    FilterKernel, FilterKernelAdapter, MaskKernel, MaskKernelAdapter, MinMaxKernel,
    MinMaxKernelAdapter, MinMaxResult, TakeKernel, TakeKernelAdapter,
};
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl FilterKernel for NullVTable {
    fn filter(&self, _array: &Self::Array, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(NullArray::new(mask.true_count()).into_array())
    }
}

register_kernel!(FilterKernelAdapter(NullVTable).lift());

impl MaskKernel for NullVTable {
    fn mask(&self, array: &NullArray, _mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(array.to_array())
    }
}

register_kernel!(MaskKernelAdapter(NullVTable).lift());

impl TakeKernel for NullVTable {
    fn take(&self, array: &NullArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let indices = indices.to_primitive()?;

        // Enforce all indices are valid
        match_each_integer_ptype!(indices.ptype(), |$T| {
            for index in indices.as_slice::<$T>() {
                if !((*index as usize) < array.len()) {
                    vortex_bail!(OutOfBounds: *index as usize, 0, array.len());
                }
            }
        });

        Ok(NullArray::new(indices.len()).into_array())
    }
}

register_kernel!(TakeKernelAdapter(NullVTable).lift());

impl MinMaxKernel for NullVTable {
    fn min_max(&self, _array: &NullArray) -> VortexResult<Option<MinMaxResult>> {
        Ok(None)
    }
}

register_kernel!(MinMaxKernelAdapter(NullVTable).lift());

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_mask::Mask;

    use crate::arrays::null::NullArray;
    use crate::compute::take;
    use crate::{IntoArray, ToCanonical};

    #[test]
    fn test_slice_nulls() {
        let nulls = NullArray::new(10);
        let sliced = nulls.slice(0, 4).unwrap().to_null().unwrap();

        assert_eq!(sliced.len(), 4);
        assert!(matches!(sliced.validity_mask().unwrap(), Mask::AllFalse(4)));
    }

    #[test]
    fn test_take_nulls() {
        let nulls = NullArray::new(10);
        let taken = take(nulls.as_ref(), &buffer![0u64, 2, 4, 6, 8].into_array())
            .unwrap()
            .to_null()
            .unwrap();

        assert_eq!(taken.len(), 5);
        assert!(matches!(taken.validity_mask().unwrap(), Mask::AllFalse(5)));
    }

    #[test]
    fn test_scalar_at_nulls() {
        let nulls = NullArray::new(10);

        let scalar = nulls.scalar_at(0).unwrap();
        assert!(scalar.is_null());
        assert_eq!(scalar.dtype().clone(), DType::Null);
    }
}
