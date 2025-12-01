// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An AVX2 implementation of take operation using gather instructions.
//!
//! Only enabled for x86_64 hosts and it is gated at runtime behind feature detection to
//! ensure AVX2 instructions are available.

use vortex_compute::take::slice::avx2;
use vortex_dtype::NativePType;
use vortex_dtype::UnsignedPType;
use vortex_dtype::match_each_native_ptype;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::primitive::compute::take::TakeImpl;
use crate::validity::Validity;

#[allow(unused)]
pub(super) struct TakeKernelAVX2;

impl TakeImpl for TakeKernelAVX2 {
    #[inline(always)]
    fn take(
        &self,
        values: &PrimitiveArray,
        indices: &PrimitiveArray,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        assert!(indices.ptype().is_unsigned_int());

        Ok(match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            match_each_native_ptype!(values.ptype(), |V| {
                // SAFETY: This kernel is only selected when avx2 cpu-feature is detected.
                unsafe {
                    take_primitive_avx2(values.as_slice::<V>(), indices.as_slice::<I>(), validity)
                }
            })
        })
        .into_array())
    }
}

/// # Safety
///
/// The caller must ensure that if the validity has a length, it is the same length as the indices,
/// and that the `avx2` feature is enabled.
#[target_feature(enable = "avx2")]
unsafe fn take_primitive_avx2<V, I>(
    values: &[V],
    indices: &[I],
    validity: Validity,
) -> PrimitiveArray
where
    V: NativePType,
    I: UnsignedPType,
{
    // SAFETY: The caller guarantees that the `avx2` feature is enabled.
    let buffer = unsafe { avx2::take_avx2(values, indices) };

    debug_assert!(
        validity
            .maybe_len()
            .is_none_or(|validity_len| validity_len == buffer.len())
    );

    // SAFETY: The caller ensures that the validity and indices have the same length, so the taken
    // buffer and the validity must have the same length.
    unsafe { PrimitiveArray::new_unchecked(buffer, validity) }
}
