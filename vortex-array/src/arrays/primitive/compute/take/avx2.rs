//! An AVX2 implementation of take operation using gather instructions.
//!
//! Only enabled for x86_64 hosts and it is gated at runtime behind feature detection to
//! ensure AVX2 instructions are available.

use std::arch::x86_64::*;

use num_traits::AsPrimitive;
use vortex_buffer::{Alignment, Buffer, BufferMut};
use vortex_dtype::{
    NativePType, PType, match_each_integer_ptype, match_each_native_ptype,
    match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexResult, vortex_panic};

use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::primitive::compute::take::{TakeImpl, take_primitive_scalar};
use crate::validity::Validity;
use crate::{ArrayRef, IntoArray};

pub(super) struct TakeKernelAVX2;

impl TakeImpl for TakeKernelAVX2 {
    #[allow(clippy::cognitive_complexity)]
    #[inline(always)]
    fn take(
        &self,
        values: &PrimitiveArray,
        indices: &PrimitiveArray,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        if values.ptype() != PType::F16
            && indices.dtype().is_unsigned_int()
            && indices.all_valid()?
            && values.all_valid()?
        {
            match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
                match_each_native_ptype!(values.ptype(), |V| {
                    // SAFETY: this kernel is only selected when avx2 cpu-feature is detected
                    Ok(unsafe {
                        take_primitive_avx2(
                            indices.as_slice::<I>(),
                            values.as_slice::<V>(),
                            validity,
                        )
                    }
                    .into_array())
                })
            })
        } else {
            match_each_native_ptype!(values.ptype(), |T| {
                match_each_integer_ptype!(indices.ptype(), |I| {
                    // NOTE: we don't need to pre-scan the indices here, because
                    //  the slice indexing inside of take_primitive_scalar will do bounds checks.
                    let result =
                        take_primitive_scalar(values.as_slice::<T>(), indices.as_slice::<I>());
                    Ok(PrimitiveArray::new(result, validity).into_array())
                })
            })
        }
    }
}

/// The main gather function that is used by the inner loop kernel for AVX2 gather.
pub(crate) trait GatherFn<Idx, Values> {
    /// The number of data elements that are written to the `dst` on each loop iteration.
    const WIDTH: usize;
    /// The number of indices read from `indices` on each loop iteration.
    /// Depending on the available instructions and bit-width we may stride by a larger amount
    /// than we actually end up reading from `src` (governed by the `WIDTH` parameter).
    const STRIDE: usize = Self::WIDTH;

    /// Gather values from `src` into the `dst` using the `indices`, optionally using
    /// SIMD instructions.
    ///
    /// # Safety
    ///
    /// This function can read up to `STRIDE` elements through `indices`, and read/write up to
    /// `WIDTH` elements through `src` and `dst` respectively.
    unsafe fn gather(indices: *const Idx, src: *const Values, dst: *mut Values);
}

/// AVX2 version of GatherFn defined for 32- and 64-bit value types.
enum AVX2Gather {}

#[inline(always)]
unsafe fn identity<T>(input: T) -> T {
    input
}

macro_rules! impl_gather {
    ($idx:ty, $({$value:ty => load: $load:ident, extend: $extend:ident, gather: $gather:ident, store: $store:ident, WIDTH = $WIDTH:literal, STRIDE = $STRIDE:literal }),+) => {
        $(
            impl GatherFn<$idx, $value> for AVX2Gather {
                const WIDTH: usize = $WIDTH;
                const STRIDE: usize = $STRIDE;

                #[allow(clippy::cast_possible_truncation)]
                #[inline(always)]
                unsafe fn gather(indices: *const $idx, src: *const $value, dst: *mut $value) {
                    const {
                        assert!($WIDTH <= $STRIDE, "dst cannot advance by more than the stride");
                    }

                    const SCALE: i32 = std::mem::size_of::<$value>() as i32;

                    let indices_vec = unsafe { $load(indices.cast()) };
                    // Extend indices to fill vector register
                    let indices_vec = unsafe { $extend(indices_vec) };
                    // Gather the values into vector register
                    let values_vec = unsafe { $gather::<SCALE>(src.cast(), indices_vec) };

                    // Write the vec out to dst.
                    unsafe { $store(dst.cast(), values_vec) };
                }
            }
        )*
    };
}

impl_gather!(u8,
    // 32-bit values, loaded 8 at a time
    { u32 => load: _mm_loadu_si128, extend: _mm256_cvtepu8_epi32, gather: _mm256_i32gather_epi32, store: _mm256_storeu_si256, WIDTH = 8, STRIDE = 8 },
    { i32 => load: _mm_loadu_si128, extend: _mm256_cvtepu8_epi32, gather: _mm256_i32gather_epi32, store: _mm256_storeu_si256, WIDTH = 8, STRIDE = 8 },

    // 64-bit values, loaded 4 at a time
    { u64 => load: _mm_loadu_si128, extend: _mm256_cvtepu8_epi64, gather: _mm256_i64gather_epi64, store: _mm256_storeu_si256, WIDTH = 4, STRIDE = 4 },
    { i64 => load: _mm_loadu_si128, extend: _mm256_cvtepu8_epi64, gather: _mm256_i64gather_epi64, store: _mm256_storeu_si256, WIDTH = 4, STRIDE = 4 }
);

impl_gather!(u16,
    // 32-bit values. 8x indices loaded at a time and 8x values written at a time
    { u32 => load: _mm_loadu_si128, extend: _mm256_cvtepu16_epi32, gather: _mm256_i32gather_epi32, store: _mm256_storeu_si256, WIDTH = 8, STRIDE = 8 },
    { i32 => load: _mm_loadu_si128, extend: _mm256_cvtepu16_epi32, gather: _mm256_i32gather_epi32, store: _mm256_storeu_si256, WIDTH = 8, STRIDE = 8 },

    // 64-bit values. 8x indices loaded at a time and 4x values loaded at a time.
    { u64 => load: _mm_loadu_si128, extend: _mm256_cvtepu16_epi64, gather: _mm256_i64gather_epi64, store: _mm256_storeu_si256, WIDTH = 4, STRIDE = 8 },
    { i64 => load: _mm_loadu_si128, extend: _mm256_cvtepu16_epi64, gather: _mm256_i64gather_epi64, store: _mm256_storeu_si256, WIDTH = 4, STRIDE = 8 }
);

impl_gather!(u32,
    // 32-bit values. 8x indices loaded at a time and 8x values written
    { u32 => load: _mm256_loadu_si256, extend: identity, gather: _mm256_i32gather_epi32, store: _mm256_storeu_si256, WIDTH = 8, STRIDE = 8 },
    { i32 => load: _mm256_loadu_si256, extend: identity, gather: _mm256_i32gather_epi32, store: _mm256_storeu_si256, WIDTH = 8, STRIDE = 8 },

    // 64-bit values, 4x indices loaded at a time and 4x values written
    { u64 => load: _mm_loadu_si128, extend: identity, gather: _mm256_i32gather_epi64, store: _mm256_storeu_si256, WIDTH = 4, STRIDE = 4 },
    { i64 => load: _mm_loadu_si128, extend: identity, gather: _mm256_i32gather_epi64, store: _mm256_storeu_si256, WIDTH = 4, STRIDE = 4 }
);

impl_gather!(u64,
    // 64-bit values. Gathered 4 at a time (4x u64 indices fit into an mm256 register)
    { u32 => load: _mm256_loadu_si256, extend: identity, gather: _mm256_i64gather_epi32, store: _mm_storeu_si128, WIDTH = 4, STRIDE = 4 },
    { i32 => load: _mm256_loadu_si256, extend: identity, gather: _mm256_i64gather_epi32, store: _mm_storeu_si128, WIDTH = 4, STRIDE = 4 },

    // 64-bit values. Gathered 4 at a time (4x u64 indices fit into an mm256 register)
    { u64 => load: _mm256_loadu_si256, extend: identity, gather: _mm256_i64gather_epi64, store: _mm256_storeu_si256, WIDTH = 4, STRIDE = 4 },
    { i64 => load: _mm256_loadu_si256, extend: identity, gather: _mm256_i64gather_epi64, store: _mm256_storeu_si256, WIDTH = 4, STRIDE = 4 }
);

/// AVX2 core inner loop for certain `Idx` and `Value` type.
#[inline(always)]
fn exec_take<Idx, Value, Gather>(indices: &[Idx], values: &[Value]) -> Buffer<Value>
where
    Idx: Copy + AsPrimitive<usize>,
    Value: Copy,
    Gather: GatherFn<Idx, Value>,
{
    let indices_len = indices.len();
    let mut buffer =
        BufferMut::<Value>::with_capacity_aligned(indices_len, Alignment::of::<__m256i>());
    let buf_uninit = buffer.spare_capacity_mut();

    let mut offset = 0;
    // Loop terminates STRIDE elements before end of the indices array because the GatherFn
    // might read up to STRIDE src elements at a time, even though it only advances WIDTH elements
    // in the dst.
    while offset + Gather::STRIDE < indices_len {
        // SAFETY: gather_simd preconditions satisfied:
        //  1. `(indices + offset)..(indices + offset + STRIDE)` is in-bounds for indices allocation
        //  2. `buffer` has same len as indices so `buffer + offset + STRIDE` is always valid.
        unsafe {
            Gather::gather(
                indices.as_ptr().add(offset),
                values.as_ptr(),
                buf_uninit.as_mut_ptr().add(offset).cast(),
            )
        };
        offset += Gather::WIDTH;
    }

    // Remainder
    while offset < indices_len {
        buf_uninit[offset].write(values[indices[offset].as_()]);
        offset += 1;
    }

    assert_eq!(offset, indices_len);

    // SAFETY: all elements have been initialized.
    unsafe { buffer.set_len(indices_len) };

    buffer.freeze()
}

/// AVX2-optimized take operation dispatch.
///
/// This returns None if the AVX2 feature is not detected at runtime, signalling to the caller
/// that it should fall back to the scalar implementation.
///
/// If AVX2 is available, this returns a PrimitiveArray containing the result of the take operation
/// accelerated using AVX2 instructions.
///
/// # Panics
///
/// This function panics if any of the provided `indices` are out of bounds for `values`.
#[target_feature(enable = "avx2")]
#[allow(clippy::cognitive_complexity)]
pub(crate) fn take_primitive_avx2<I, V>(
    indices: &[I],
    values: &[V],
    validity: Validity,
) -> PrimitiveArray
where
    I: NativePType + AsPrimitive<usize>,
    V: NativePType,
{
    macro_rules! dispatch_avx2 {
        ($indices:ty, $values:ty) => {{
            let indices = unsafe { std::mem::transmute::<&[I], &[$indices]>(indices) };
            let values = unsafe { std::mem::transmute::<&[V], &[$values]>(values) };

            // bounds check indices
            for &idx in indices {
                let idx: usize = idx.as_();
                if idx >= values.len() {
                    vortex_panic!(
                        "cannot take with index {} on array with length {}",
                        idx,
                        values.len()
                    );
                }
            }

            let result = exec_take::<$indices, $values, AVX2Gather>(indices, values);
            PrimitiveArray::new(
                unsafe { std::mem::transmute::<Buffer<$values>, Buffer<V>>(result) },
                validity,
            )
        }};
    }

    match (I::PTYPE, V::PTYPE) {
        (PType::U8, PType::I32) => dispatch_avx2!(u8, i32),
        (PType::U8, PType::U32) => dispatch_avx2!(u8, u32),
        (PType::U8, PType::I64) => dispatch_avx2!(u8, i64),
        (PType::U8, PType::U64) => dispatch_avx2!(u8, u64),
        (PType::U16, PType::I32) => dispatch_avx2!(u16, i32),
        (PType::U16, PType::U32) => dispatch_avx2!(u16, u32),
        (PType::U16, PType::I64) => dispatch_avx2!(u16, i64),
        (PType::U16, PType::U64) => dispatch_avx2!(u16, u64),
        (PType::U32, PType::I32) => dispatch_avx2!(u32, i32),
        (PType::U32, PType::U32) => dispatch_avx2!(u32, u32),
        (PType::U32, PType::I64) => dispatch_avx2!(u32, i64),
        (PType::U32, PType::U64) => dispatch_avx2!(u32, u64),

        // Scalar fallback for unsupported value types.
        _ => {
            let result = take_primitive_scalar(values, indices);
            PrimitiveArray::new(result, validity)
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn take_primitive_avx2<I, V>(
    _indices: &[I],
    _values: &[V],
    _nullability: Nullability,
) -> Option<PrimitiveArray>
where
    I: NativePType + AsPrimitive<usize>,
    V: NativePType,
{
    None
}

#[cfg(test)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod tests {
    use super::*;

    #[test]
    fn test_take_i32_u32() {
        let values = [10i32, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let indices = [0u32, 2, 4, 6, 8, 1, 3, 5, 7, 9];

        let result = unsafe { take_primitive_avx2(&indices, &values, Validity::NonNullable) };
        let expected = [10i32, 30, 50, 70, 90, 20, 40, 60, 80, 100];
        assert_eq!(result.as_slice::<i32>(), &expected);
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn test_take_f32_u32() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices = [7u32, 0, 3, 2, 1, 4, 6, 5];

        let result = unsafe { take_primitive_avx2(&indices, &values, Validity::NonNullable) };

        let expected = [8.0f32, 1.0, 4.0, 3.0, 2.0, 5.0, 7.0, 6.0];
        assert_eq!(result.as_slice::<f32>(), &expected);
    }

    #[test]
    fn test_take_i64_u64() {
        let values = [100i64, 200, 300, 400, 500];
        let indices = [4u64, 0, 2, 1, 3];

        let result = unsafe { take_primitive_avx2(&indices, &values, Validity::NonNullable) };

        let expected = [500i64, 100, 300, 200, 400];
        assert_eq!(result.as_slice::<i64>(), &expected);
    }

    #[test]
    fn test_take_f64_u64() {
        let values = [1.1f64, 2.2, 3.3, 4.4, 5.5, 6.6];
        let indices = [0u64, 5, 2, 1, 4, 3];

        let result = unsafe { take_primitive_avx2(&indices, &values, Validity::NonNullable) };

        let expected = [1.1f64, 6.6, 3.3, 2.2, 5.5, 4.4];
        assert_eq!(result.as_slice::<f64>(), &expected);
    }

    #[test]
    fn test_empty_arrays() {
        let values: [i32; 0] = [];
        let indices: [u32; 0] = [];

        let result = unsafe { take_primitive_avx2(&indices, &values, Validity::NonNullable) };
        assert_eq!(result.as_slice::<i32>().len(), 0);
    }

    #[test]
    fn test_single_element() {
        let values = [42i32];
        let indices = [0u32];

        let result = unsafe { take_primitive_avx2(&indices, &values, Validity::NonNullable) };
        assert_eq!(result.as_slice::<i32>(), &[42i32]);
    }
}
