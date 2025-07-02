#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx2;

#[cfg(feature = "nightly")]
mod portable;

use std::sync::LazyLock;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_dtype::{DType, NativePType, match_each_integer_ptype, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::PrimitiveVTable;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::{TakeKernel, TakeKernelAdapter, cast};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

// Kernel selection happens on the first call to `take` and uses a combination of compile-time
// and runtime feature detection to infer the best kernel for the platform.
static PRIMITIVE_TAKE_KERNEL: LazyLock<&'static dyn TakeImpl> = LazyLock::new(|| {
    cfg_if::cfg_if! {
        if #[cfg(feature = "nightly")] {
            // nightly codepath: use portable_simd kernel
            &portable::TakeKernelPortableSimd
        } else if #[cfg(target_arch = "x86_64")] {
            // stable x86_64 path: use the optimized AVX2 kernel when available, falling
            // back to scalar when not.
            if is_x86_feature_detected!("avx2") {
                &avx2::TakeKernelAVX2
            } else {
                &TakeKernelScalar
            }
        } else {
            // stable all other platforms: scalar kernel
            &TakeKernelScalar
        }
    }
});

trait TakeImpl: Send + Sync {
    fn take(
        &self,
        array: &PrimitiveArray,
        indices: &PrimitiveArray,
        validity: Validity,
    ) -> VortexResult<ArrayRef>;
}

#[allow(unused)]
struct TakeKernelScalar;

impl TakeImpl for TakeKernelScalar {
    fn take(
        &self,
        array: &PrimitiveArray,
        indices: &PrimitiveArray,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        match_each_native_ptype!(array.ptype(), |T| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                let values = take_primitive_scalar(array.as_slice::<T>(), indices.as_slice::<I>());
                Ok(PrimitiveArray::new(values, validity).into_array())
            })
        })
    }
}

impl TakeKernel for PrimitiveVTable {
    fn take(&self, array: &PrimitiveArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let unsigned_indices = match indices.dtype() {
            DType::Primitive(p, n) => {
                if p.is_unsigned_int() {
                    indices.to_primitive()?
                } else {
                    // This will fail if all values cannot be converted to unsigned
                    cast(indices, &DType::Primitive(p.to_unsigned(), *n))?.to_primitive()?
                }
            }
            _ => vortex_bail!("Invalid indices dtype: {}", indices.dtype()),
        };
        let validity = array.validity().take(unsigned_indices.as_ref())?;
        // Delegate to the best kernel based on the target CPU
        PRIMITIVE_TAKE_KERNEL.take(array, &unsigned_indices, validity)
    }
}

register_kernel!(TakeKernelAdapter(PrimitiveVTable).lift());

// Compiler may see this as unused based on enabled features
#[allow(unused)]
#[inline(always)]
fn take_primitive_scalar<T: NativePType, I: NativePType + AsPrimitive<usize>>(
    array: &[T],
    indices: &[I],
) -> Buffer<T> {
    indices.iter().map(|idx| array[idx.as_()]).collect()
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::arrays::primitive::compute::take::take_primitive_scalar;
    use crate::arrays::{BoolArray, PrimitiveArray};
    use crate::compute::take;
    use crate::validity::Validity;
    use crate::{Array, IntoArray};

    #[test]
    fn test_take() {
        let a = vec![1i32, 2, 3, 4, 5];
        let result = take_primitive_scalar(&a, &[0, 0, 4, 2]);
        assert_eq!(result.as_slice(), &[1i32, 1, 5, 3]);
    }

    #[test]
    fn test_take_with_null_indices() {
        let values = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5],
            Validity::Array(BoolArray::from_iter([true, true, false, false, true]).into_array()),
        );
        let indices = PrimitiveArray::new(
            buffer![0, 3, 4],
            Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
        );
        let actual = take(values.as_ref(), indices.as_ref()).unwrap();
        assert_eq!(actual.scalar_at(0).unwrap(), Scalar::from(Some(1)));
        // position 3 is null
        assert_eq!(actual.scalar_at(1).unwrap(), Scalar::null_typed::<i32>());
        // the third index is null
        assert_eq!(actual.scalar_at(2).unwrap(), Scalar::null_typed::<i32>());
    }
}
