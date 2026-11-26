// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Views of memory as values of a primitive type, so the rest of the block runtime holds blocks
//! as borrowed bytes without `unsafe`.
//!
//! Every native type is plain old data: it has no padding, and every bit pattern of its size is a
//! value. Initialized bytes that are aligned for a type and a multiple of its size are therefore a
//! slice of it, and values of it are initialized bytes.

use std::mem::MaybeUninit;

use crate::dtype::NativePType;

/// Check that `ptr` and `len` bytes can be viewed as values of `T`.
fn check<T>(ptr: *const u8, len: usize) {
    assert!(ptr.cast::<T>().is_aligned(), "bytes misaligned for values");
    assert!(
        len.is_multiple_of(size_of::<T>()),
        "bytes are not whole values"
    );
}

/// The bytes of `values`.
pub(super) fn bytes<T: NativePType>(values: &[T]) -> &[u8] {
    // SAFETY: values of a native type are initialized bytes, and `u8` has alignment 1.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

/// `bytes` as values of `T`.
///
/// # Panics
///
/// Panics if `bytes` is misaligned for `T` or not a whole number of values.
pub(super) fn values<T: NativePType>(bytes: &[u8]) -> &[T] {
    check::<T>(bytes.as_ptr(), bytes.len());
    // SAFETY: checked above, and any initialized bytes are values of a native type.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / size_of::<T>()) }
}

/// `bytes` as values of `T`, to overwrite.
///
/// # Panics
///
/// Panics if `bytes` is misaligned for `T` or not a whole number of values.
pub(super) fn values_mut<T: NativePType>(bytes: &mut [u8]) -> &mut [T] {
    check::<T>(bytes.as_ptr(), bytes.len());
    // SAFETY: checked above; any value written is initialized bytes, and any bytes are a value.
    unsafe {
        std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len() / size_of::<T>())
    }
}

/// The bytes of uninitialized room for values of `T`.
pub(super) fn uninit_bytes_mut<T: NativePType>(
    values: &mut [MaybeUninit<T>],
) -> &mut [MaybeUninit<u8>] {
    // SAFETY: the room is `size_of_val` bytes, and `u8` has alignment 1.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), size_of_val(values)) }
}

/// Uninitialized `bytes` as room for values of `T`.
///
/// # Panics
///
/// Panics if `bytes` is misaligned for `T` or not a whole number of values.
pub(super) fn uninit_values_mut<T: NativePType>(
    bytes: &mut [MaybeUninit<u8>],
) -> &mut [MaybeUninit<T>] {
    check::<T>(bytes.as_ptr().cast(), bytes.len());
    // SAFETY: checked above, and uninitialized room needs no valid values.
    unsafe {
        std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len() / size_of::<T>())
    }
}

/// Initialized `bytes` as room to write.
///
/// # Safety
///
/// The caller must write only initialized bytes through the result, so that `bytes` stays
/// initialized.
pub(super) unsafe fn as_uninit_mut(bytes: &mut [u8]) -> &mut [MaybeUninit<u8>] {
    // SAFETY: `MaybeUninit<u8>` has the layout of `u8`, and the caller keeps the bytes initialized.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len()) }
}

/// `bytes`, once written, as initialized bytes.
///
/// # Safety
///
/// Every byte must be initialized.
pub(super) unsafe fn assume_init_mut(bytes: &mut [MaybeUninit<u8>]) -> &mut [u8] {
    // SAFETY: the caller guarantees every byte is initialized.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len()) }
}

/// The bits of `value` as a value of `S`, a native type of the same width: how an op that keeps
/// the width but changes the type writes its output over its input in place.
///
/// # Panics
///
/// Panics if `T` and `S` differ in size.
#[inline]
pub(super) fn bitcast<T: NativePType, S: NativePType>(value: T) -> S {
    assert_eq!(size_of::<T>(), size_of::<S>(), "a bit cast keeps the width");
    // SAFETY: both are native types of the same size, so the bits of a `T` are an `S`.
    unsafe { std::mem::transmute_copy(&value) }
}
