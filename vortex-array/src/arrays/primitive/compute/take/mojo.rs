// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Mojo SIMD take kernel, AOT-compiled from `kernels/take.mojo`.

use std::mem::size_of;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::TakeImpl;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::vtable::Primitive;
use crate::dtype::IntegerPType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::validity::Validity;

// ---------------------------------------------------------------------------
// FFI declarations — 16 take symbols + 4 filter symbols
// ---------------------------------------------------------------------------

unsafe extern "C" {
    // 1-byte values
    fn vortex_take_1byte_u8idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_1byte_u16idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_1byte_u32idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_1byte_u64idx(src: usize, idx: usize, dst: usize, n: usize);

    // 2-byte values
    fn vortex_take_2byte_u8idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_2byte_u16idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_2byte_u32idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_2byte_u64idx(src: usize, idx: usize, dst: usize, n: usize);

    // 4-byte values
    fn vortex_take_4byte_u8idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_4byte_u16idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_4byte_u32idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_4byte_u64idx(src: usize, idx: usize, dst: usize, n: usize);

    // 8-byte values
    fn vortex_take_8byte_u8idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_8byte_u16idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_8byte_u32idx(src: usize, idx: usize, dst: usize, n: usize);
    fn vortex_take_8byte_u64idx(src: usize, idx: usize, dst: usize, n: usize);
}

// ---------------------------------------------------------------------------
// Kernel implementation
// ---------------------------------------------------------------------------

pub(super) struct TakeKernelMojo;

impl TakeImpl for TakeKernelMojo {
    fn take(
        &self,
        array: ArrayView<'_, Primitive>,
        indices: ArrayView<'_, Primitive>,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        Ok(match_each_native_ptype!(array.ptype(), |V| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                let values = take_mojo::<V, I>(array.as_slice::<V>(), indices.as_slice::<I>());
                PrimitiveArray::new(values, validity).into_array()
            })
        }))
    }
}

/// Dispatch to the appropriate Mojo take kernel based on value size and index type.
fn take_mojo<V: NativePType, I: IntegerPType>(src: &[V], idx: &[I]) -> Buffer<V> {
    let n = idx.len();
    let mut dst = BufferMut::<V>::with_capacity(n);
    let dst_ptr = dst.spare_capacity_mut().as_mut_ptr() as usize;
    let src_ptr = src.as_ptr() as usize;
    let idx_ptr = idx.as_ptr() as usize;

    type TakeFn = unsafe extern "C" fn(usize, usize, usize, usize);

    let func: TakeFn = match (size_of::<V>(), I::PTYPE) {
        (1, PType::U8) => vortex_take_1byte_u8idx,
        (1, PType::U16) => vortex_take_1byte_u16idx,
        (1, PType::U32) => vortex_take_1byte_u32idx,
        (1, PType::U64) => vortex_take_1byte_u64idx,
        (2, PType::U8) => vortex_take_2byte_u8idx,
        (2, PType::U16) => vortex_take_2byte_u16idx,
        (2, PType::U32) => vortex_take_2byte_u32idx,
        (2, PType::U64) => vortex_take_2byte_u64idx,
        (4, PType::U8) => vortex_take_4byte_u8idx,
        (4, PType::U16) => vortex_take_4byte_u16idx,
        (4, PType::U32) => vortex_take_4byte_u32idx,
        (4, PType::U64) => vortex_take_4byte_u64idx,
        (8, PType::U8) => vortex_take_8byte_u8idx,
        (8, PType::U16) => vortex_take_8byte_u16idx,
        (8, PType::U32) => vortex_take_8byte_u32idx,
        (8, PType::U64) => vortex_take_8byte_u64idx,
        _ => unreachable!("unsupported value size / index type combination"),
    };

    // SAFETY: Mojo kernel reads `n` elements from `src` at offsets given by `idx` and writes
    // `n` elements to `dst`. The caller guarantees the slices are valid and large enough.
    unsafe {
        func(src_ptr, idx_ptr, dst_ptr, n);
        dst.set_len(n);
    }

    dst.freeze()
}
