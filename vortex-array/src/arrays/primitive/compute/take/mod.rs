// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

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
#[cfg(target_arch = "x86_64")]
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::validity::Validity;

// Kernel selection happens on the first call to `take` and uses a combination of compile-time
// and runtime feature detection to infer the best kernel for the platform.
static PRIMITIVE_TAKE_KERNEL: LazyLock<&'static dyn TakeImpl> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        // Benchmarks (single-process, back-to-back) show AVX2 gather is fastest for 32-bit
        // values while AVX-512 gather is fastest for 64-bit values, so dispatch per value width
        // when both are available.
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx2")
        {
            return &TakeKernelWidthAware;
        }
        if is_x86_feature_detected!("avx2") {
            return &avx2::TakeKernelAVX2;
        }
        &TakeKernelScalar
    }

    #[cfg(all(target_arch = "x86", not(target_arch = "x86_64")))]
    {
        if is_x86_feature_detected!("avx2") {
            return &avx2::TakeKernelAVX2;
        }
        return &TakeKernelScalar;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        &TakeKernelScalar
    }
});

/// Routes by value width: AVX-512 gather for 64-bit values taken with `u32` indices (its sweet
/// spot), AVX2 gather for everything else (which itself falls back to scalar for unsupported
/// widths). Only selected when both AVX-512 and AVX2 are available at runtime.
#[cfg(target_arch = "x86_64")]
struct TakeKernelWidthAware;

#[cfg(target_arch = "x86_64")]
impl TakeImpl for TakeKernelWidthAware {
    #[inline(always)]
    fn take(
        &self,
        array: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        let use_avx512 = matches!(array.ptype(), PType::I64 | PType::U64 | PType::F64)
            && indices.ptype() == PType::U32;
        if use_avx512 {
            avx512::TakeKernelAVX512.take(array, indices, validity)
        } else {
            avx2::TakeKernelAVX2.take(array, indices, validity)
        }
    }
}

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

    // Single-process, back-to-back three-way comparison of the scalar, AVX2, and AVX-512 take
    // kernels on identical inputs. Run with:
    //   cargo test --release -p vortex-array -- --ignored --nocapture three_way_kernel_timing
    #[test]
    #[ignore = "manual benchmark, run explicitly"]
    #[allow(clippy::cast_possible_truncation)]
    fn three_way_kernel_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        use super::TakeImpl;
        use super::TakeKernelScalar;
        use super::avx2::TakeKernelAVX2;
        use super::avx512::TakeKernelAVX512;

        const N_INDICES: usize = 100_000;
        const ROUNDS: usize = 7;

        fn time_kernel(
            name: &str,
            kernel: &dyn TakeImpl,
            values: &PrimitiveArray,
            indices: &PrimitiveArray,
            iters: usize,
            rounds: usize,
        ) {
            // Warm up; black_box prevents the loop from being optimized away.
            let reference = kernel
                .take(values.as_view(), indices.as_view(), Validity::NonNullable)
                .unwrap();
            black_box(&reference);

            let mut best = f64::INFINITY;
            for _ in 0..rounds {
                let start = Instant::now();
                for _ in 0..iters {
                    let out = kernel
                        .take(values.as_view(), indices.as_view(), Validity::NonNullable)
                        .unwrap();
                    black_box(out);
                }
                let per_iter_us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
                best = best.min(per_iter_us);
            }
            println!(
                "  {name:>8}: {best:7.2} us/iter (best of {rounds} rounds, {iters} iters each)"
            );
        }

        // (values_len, iters): in-cache (L1/L2, dict-decode hot path) and memory-bound (>L3).
        for &(values_len, iters) in &[(4096u32, 1000usize), (4_000_000u32, 200usize)] {
            // Deterministic pseudo-random in-bounds indices.
            let idx: Vec<u32> = (0..N_INDICES as u32)
                .map(|i| i.wrapping_mul(2_654_435_761) % values_len)
                .collect();
            let indices = PrimitiveArray::from_iter(idx);

            let values32 = PrimitiveArray::from_iter((0..values_len).collect::<Vec<u32>>());
            let values64 = PrimitiveArray::from_iter((0..values_len as u64).collect::<Vec<u64>>());

            let regime = if values_len <= 4096 {
                "in-cache"
            } else {
                "memory-bound"
            };
            println!(
                "\n=== take: {N_INDICES} indices into {values_len}-element values ({regime}) ==="
            );
            println!("u32 values:");
            time_kernel(
                "scalar",
                &TakeKernelScalar,
                &values32,
                &indices,
                iters,
                ROUNDS,
            );
            time_kernel("avx2", &TakeKernelAVX2, &values32, &indices, iters, ROUNDS);
            time_kernel(
                "avx512",
                &TakeKernelAVX512,
                &values32,
                &indices,
                iters,
                ROUNDS,
            );
            println!("u64 values:");
            time_kernel(
                "scalar",
                &TakeKernelScalar,
                &values64,
                &indices,
                iters,
                ROUNDS,
            );
            time_kernel("avx2", &TakeKernelAVX2, &values64, &indices, iters, ROUNDS);
            time_kernel(
                "avx512",
                &TakeKernelAVX512,
                &values64,
                &indices,
                iters,
                ROUNDS,
            );
        }
    }
}
