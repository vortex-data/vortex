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
                portable::take_portable_simd::<u16, C, { portable::SIMD_WIDTH }>(
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
                    let decoded = portable::take_portable_simd::<V, C, { portable::SIMD_WIDTH }>(
                        array.as_slice(),
                        unsigned_indices.as_slice(),
                    );
                    Ok(PrimitiveArray::new(decoded, validity).into_array())
                })
            })
        }
    }
}
