// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An implementation of the Take kernel for primitive Arrays that uses
//! the nightly-only `portable_simd` feature.
//!
//! This is only enabled on non-x86_64 platforms and when using the nightly compiler for builds.

#![allow(unused)]

use std::mem::MaybeUninit;
use std::mem::transmute;
use std::simd;
use std::simd::num::SimdUint;

use multiversion::multiversion;
use num_traits::AsPrimitive;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::take::slice::portable;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::match_each_native_simd_ptype;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::compute::take::TakeImpl;
use crate::validity::Validity;

pub(super) struct TakeKernelPortableSimd;

// SIMD types larger than the SIMD register size are beneficial for
// performance as this leads to better instruction level parallelism.
const SIMD_WIDTH: usize = 64;

impl TakeImpl for TakeKernelPortableSimd {
    fn take(
        &self,
        array: &PrimitiveArray,
        unsigned_indices: &PrimitiveArray,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        if array.ptype() == PType::F16 {
            // Special handling for f16 to treat as opaque u16
            let decoded = match_each_unsigned_integer_ptype!(unsigned_indices.ptype(), |C| {
                portable::take_portable_simd::<u16, C, SIMD_WIDTH>(
                    array.reinterpret_cast(PType::U16).as_slice(),
                    unsigned_indices.as_slice(),
                )
            });
            Ok(PrimitiveArray::new(decoded, validity)
                .reinterpret_cast(PType::F16)
                .into_array())
        } else {
            match_each_unsigned_integer_ptype!(unsigned_indices.ptype(), |C| {
                match_each_native_simd_ptype!(array.ptype(), |V| {
                    let decoded = portable::take_portable_simd::<V, C, SIMD_WIDTH>(
                        array.as_slice(),
                        unsigned_indices.as_slice(),
                    );
                    Ok(PrimitiveArray::new(decoded, validity).into_array())
                })
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::take_portable_simd;

    #[test]
    fn test_take_out_of_bounds() {
        let indices = vec![2_000_000u32; 64];
        let values = vec![1i32];

        let result = take_portable_simd::<u32, i32, 64>(&indices, &values);
        assert_eq!(result.as_slice(), [0i32; 64]);
    }
}
