// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An AVX2 implementation of take operation using gather instructions.
//!
//! Only enabled for x86_64 hosts and it is gated at runtime behind feature detection to
//! ensure AVX2 instructions are available.

#![cfg(any(target_arch = "x86_64", target_arch = "x86"))]

mod gather;
#[cfg(test)]
mod tests;

use std::arch::x86_64::__m256i;
use std::arch::x86_64::_mm256_and_si256;
use std::arch::x86_64::_mm256_movemask_epi8;
use std::arch::x86_64::_mm256_set1_epi32;

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use self::gather::Avx2Gather;
use self::gather::GatherFn;
use super::FixedWidthTakeValue;
use super::take_values_scalar;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::match_each_unsigned_integer_ptype;

/// Takes the specified indices into a new [`Buffer`] using AVX2 SIMD.
///
/// An AVX2 gather only moves raw bytes, so signedness and float-ness are irrelevant — only the
/// byte width of `V` matters. Any 4-byte value rides the gather through the `u32` lane and any
/// 8-byte value through the `u64` lane, regardless of its actual type. Values 1 or 2 bytes wide
/// (AVX2 has no sub-32-bit gather) and wider than 8 bytes fall back to the scalar kernel.
///
/// [`FixedWidthTakeValue`] guarantees that the complete representation is initialized before it is
/// read through an integer lane.
///
/// # Safety
///
/// The caller must ensure the `avx2` feature is enabled.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn take_avx2<V: FixedWidthTakeValue, I: UnsignedPType>(
    buffer: &[V],
    indices: &[I],
) -> Buffer<V> {
    if buffer.is_empty() {
        assert!(
            indices.is_empty(),
            "cannot take a non-empty set of indices from an empty buffer"
        );
        return Buffer::empty();
    }

    // Dispatch on the gather lane width. The index type must still be concretized to select the
    // right `GatherFn` impl, so re-dispatch it with `match_each_unsigned_integer_ptype!`.
    macro_rules! dispatch {
        ($lane:ty) => {{
            match_each_unsigned_integer_ptype!(I::PTYPE, |Idx| {
                // SAFETY: `Idx` has the same `PTYPE` as `I`, so this is a no-op reinterpret of the
                // index slice into the concrete type the gather impl is keyed on.
                let indices = unsafe { std::mem::transmute::<&[I], &[Idx]>(indices) };
                exec_take::<V, $lane, Idx, Avx2Gather>(buffer, indices)
            })
        }};
    }

    match size_of::<V>() {
        // The i32 gather interprets u32 lanes as signed offsets. A valid high u32 index needs the
        // scalar path when the values slice exceeds the non-negative i32 addressable range.
        4 if I::PTYPE == PType::U32 && !i32_gather_can_address(buffer.len()) => {
            take_values_scalar(buffer, indices)
        }
        4 => dispatch!(u32),
        8 => dispatch!(u64),
        // 1/2-byte and >8-byte values have no AVX2 gather lane, so fall back to scalar.
        _ => take_values_scalar(buffer, indices),
    }
}

const fn i32_gather_can_address(values_len: usize) -> bool {
    values_len <= i32::MAX as usize + 1
}

/// AVX2 core inner loop for a given index type `Idx`, output element type `Out`, and gather
/// `Lane` type.
///
/// `Out` is the element type written to the output buffer; `Lane` (`u32` or `u64`) is the
/// integer type the gather intrinsics operate on. The caller must pair them so that
/// `size_of::<Out>() == size_of::<Lane>()` (the only caller, [`take_avx2`], picks `Lane` from
/// `size_of::<Out>()`). Each valid lane copies the initialized representation of an existing
/// `Out`; invalid lanes are masked and cause a panic before the output buffer is initialized.
/// Gather instructions tolerate the source's potentially weaker alignment.
#[inline(always)]
fn exec_take<Out, Lane, Idx, Gather>(values: &[Out], indices: &[Idx]) -> Buffer<Out>
where
    Out: FixedWidthTakeValue,
    Idx: UnsignedPType,
    Gather: GatherFn<Idx, Lane>,
{
    assert_eq!(
        size_of::<Out>(),
        size_of::<Lane>(),
        "gather lane and output element must have the same size"
    );

    let indices_len = indices.len();
    // The length is an exclusive upper bound on valid indices. `None` means the bound does not
    // fit in the index type, so every representable index is in-bounds.
    let max_index = Idx::from(values.len());
    let mut buffer =
        BufferMut::<Out>::with_capacity_aligned(indices_len, Alignment::of::<__m256i>());
    let buf_uninit = buffer.spare_capacity_mut();

    let mut offset = 0;
    // SAFETY: `exec_take` is only called by `take_avx2`, whose caller guarantees AVX2 support.
    let mut all_indices_valid = unsafe { _mm256_set1_epi32(-1) };
    // Loop terminates STRIDE elements before end of the indices array because the `GatherFn`
    // might read up to STRIDE src elements at a time, even though it only advances WIDTH elements
    // in the dst.
    while offset + Gather::STRIDE < indices_len {
        // SAFETY: `gather` preconditions satisfied:
        //  1. `(indices + offset)..(indices + offset + STRIDE)` is in-bounds for indices
        //     allocation.
        //  2. `buffer` has same len as indices so `buffer + offset + WIDTH` is always valid.
        //  3. `size_of::<Out>() == size_of::<Lane>()` (asserted above), so the `Lane`-typed
        //     pointers address the same bytes as the `Out`-typed `values`/`buffer` allocations.
        let valid_mask = unsafe {
            Gather::gather(
                indices.as_ptr().add(offset),
                max_index,
                values.as_ptr().cast::<Lane>(),
                buf_uninit.as_mut_ptr().add(offset).cast::<Lane>(),
            )
        };
        // SAFETY: `exec_take` is only called by `take_avx2`, whose caller guarantees AVX2 support.
        all_indices_valid = unsafe { _mm256_and_si256(all_indices_valid, valid_mask) };
        offset += Gather::WIDTH;
    }

    // Invalid lanes were masked before gathering, so it is safe to defer the bounds failure until
    // after the SIMD loop and avoid a conditional branch on every iteration.
    assert!(
        // SAFETY: `exec_take` is only called by `take_avx2`, whose caller guarantees AVX2 support.
        unsafe { _mm256_movemask_epi8(all_indices_valid) } == -1,
        "take index out of bounds"
    );

    // Remainder.
    while offset < indices_len {
        buf_uninit[offset].write(values[indices[offset].as_()]);
        offset += 1;
    }

    assert_eq!(offset, indices_len);

    // SAFETY: All elements have been initialized.
    unsafe { buffer.set_len(indices_len) };

    // Do not expose the temporary SIMD over-alignment as part of the returned buffer.
    buffer = buffer.aligned(Alignment::of::<Out>());

    buffer.freeze()
}
