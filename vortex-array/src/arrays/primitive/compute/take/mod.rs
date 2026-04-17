// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx2;

use std::sync::LazyLock;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::validity::Validity;

// Kernel selection happens on the first call to `take` and uses a combination of compile-time
// and runtime feature detection to infer the best kernel for the platform.
static PRIMITIVE_TAKE_KERNEL: LazyLock<&'static dyn TakeImpl> = LazyLock::new(|| {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            &avx2::TakeKernelAVX2
        } else {
            &TakeKernelScalar
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        &TakeKernelScalar
    }
});

trait TakeImpl: Send + Sync {
    fn take(
        &self,
        array: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef>;
}

struct TakeKernelScalar;

impl TakeImpl for TakeKernelScalar {
    fn take(
        &self,
        array: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
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

impl TakeExecute for Primitive {
    fn take(
        array: ArrayView<'_, Primitive>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let DType::Primitive(ptype, null) = indices.dtype() else {
            vortex_bail!("Invalid indices dtype: {}", indices.dtype())
        };

        let unsigned_indices = if ptype.is_unsigned_int() {
            indices.clone().execute::<PrimitiveArray>(ctx)?
        } else {
            // This will fail if all values cannot be converted to unsigned
            indices
                .clone()
                .cast(DType::Primitive(ptype.to_unsigned(), *null))?
                .execute::<PrimitiveArray>(ctx)?
        };

        let validity = array
            .validity()?
            .take(&unsigned_indices.clone().into_array())?;
        // Delegate to the best kernel based on the target CPU
        {
            let unsigned_indices = unsigned_indices.as_view();
            PRIMITIVE_TAKE_KERNEL
                .take(array, unsigned_indices, validity)
                .map(Some)
        }
    }
}

// Compiler may see this as unused based on enabled features
#[inline(always)]
fn take_primitive_scalar<T: NativePType, I: IntegerPType>(
    buffer: &[T],
    indices: &[I],
) -> Buffer<T> {
    // NB: The simpler `indices.iter().map(|idx| buffer[idx.as_()]).collect()` generates suboptimal
    // assembly where the buffer length is repeatedly loaded from the stack on each iteration.

    let mut result = BufferMut::with_capacity(indices.len());
    let ptr = result.spare_capacity_mut().as_mut_ptr().cast::<T>();

    // This explicit loop with pointer writes keeps the length in a register and avoids per-element
    // capacity checks from `push()`.
    for (i, idx) in indices.iter().enumerate() {
        // SAFETY: We reserved `indices.len()` capacity, so `ptr.add(i)` is valid.
        unsafe { ptr.add(i).write(buffer[idx.as_()]) };
    }

    // SAFETY: We just wrote exactly `indices.len()` elements.
    unsafe { result.set_len(indices.len()) };
    result.freeze()
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::primitive::compute::take::take_primitive_scalar;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

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
        let actual = values.take(indices.into_array()).unwrap();
        assert_eq!(
            actual
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("no fail"),
            Scalar::from(Some(1))
        );
        // position 3 is null
        assert_eq!(
            actual
                .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("no fail"),
            Scalar::null_native::<i32>()
        );
        // the third index is null
        assert_eq!(
            actual
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .vortex_expect("no fail"),
            Scalar::null_native::<i32>()
        );
    }

    #[rstest]
    #[case(PrimitiveArray::new(buffer![42i32], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1, 2, 3, 4, 5, 6, 7], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid))]
    #[case(PrimitiveArray::new(
        buffer![0, 1, 2, 3, 4, 5],
        Validity::Array(BoolArray::from_iter([true, false, true, false, true, true]).into_array()),
    ))]
    #[case(PrimitiveArray::from_option_iter([Some(1), None, Some(3), Some(4), None]))]
    fn test_take_primitive_conformance(#[case] array: PrimitiveArray) {
        test_take_conformance(&array.into_array());
    }
}
