// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[path = "avx2.rs"]
mod avx2;
#[path = "avx512.rs"]
mod avx512;

use std::mem::align_of;
use std::mem::size_of;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use super::TakeImpl;
use super::TakeKernelScalar;
use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::UnsignedPType;
use crate::validity::Validity;

pub(super) fn select_take_impl() -> &'static dyn TakeImpl {
    if is_x86_feature_detected!("avx512f") {
        &avx512::TakeKernelAVX512
    } else if is_x86_feature_detected!("avx2") {
        &avx2::TakeKernelAVX2
    } else {
        &TakeKernelScalar
    }
}

/// # Safety
///
/// The caller must ensure that if the validity has a length, it is the same length as the
/// indices.
#[inline(always)]
unsafe fn take_primitive_with_validity<V, I>(
    values: &[V],
    indices: &[I],
    validity: Validity,
    take: impl FnOnce(&[V], &[I]) -> Buffer<V>,
) -> PrimitiveArray
where
    V: NativePType,
    I: UnsignedPType,
{
    let buffer = take(values, indices);

    debug_assert!(
        validity
            .maybe_len()
            .is_none_or(|validity_len| validity_len == buffer.len())
    );

    unsafe { PrimitiveArray::new_unchecked(buffer, validity) }
}

#[inline(always)]
fn new_simd_buffer<T>(len: usize, alignment: Alignment) -> BufferMut<T> {
    BufferMut::with_capacity_aligned(len, alignment)
}

#[inline(always)]
fn finish_simd_buffer<T>(mut buffer: BufferMut<T>, len: usize) -> Buffer<T> {
    unsafe { buffer.set_len(len) };
    buffer = buffer.aligned(Alignment::of::<T>());
    buffer.freeze()
}

#[inline(always)]
unsafe fn cast_slice<T, U>(slice: &[T]) -> &[U] {
    debug_assert_eq!(size_of::<T>(), size_of::<U>());
    debug_assert_eq!(align_of::<T>(), align_of::<U>());
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<U>(), slice.len()) }
}

#[inline(always)]
unsafe fn take_reinterpreted<V, I, U>(
    values: &[V],
    indices: &[I],
    take: impl FnOnce(&[U], &[I]) -> Buffer<U>,
) -> Buffer<V>
where
    V: NativePType,
    I: UnsignedPType,
    U: NativePType,
{
    let values = unsafe { cast_slice::<V, U>(values) };
    let taken = take(values, indices);
    unsafe { taken.transmute::<V>() }
}
