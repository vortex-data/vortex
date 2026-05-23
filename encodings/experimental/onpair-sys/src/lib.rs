// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Unsafe FFI bindings to the OnPair C++ compression library.
//!
//! The public surface is intentionally minimal: a [`Column`] owning handle
//! plus the C-ABI functions defined in `cxx/onpair_shim.h`. Safe wrappers and
//! the Vortex array implementation live in the `vortex-onpair` crate.

#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::ptr::NonNull;

pub mod ffi {
    #[repr(C)]
    pub struct OnPairColumnHandle {
        _opaque: [u8; 0],
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum OnPairStatus {
        Ok = 0,
        InvalidArg = 1,
        BadFormat = 2,
        OutOfRange = 3,
        Oom = 4,
        Internal = 99,
    }

    impl OnPairStatus {
        pub fn from_raw(raw: u32) -> Self {
            match raw {
                0 => OnPairStatus::Ok,
                1 => OnPairStatus::InvalidArg,
                2 => OnPairStatus::BadFormat,
                3 => OnPairStatus::OutOfRange,
                4 => OnPairStatus::Oom,
                _ => OnPairStatus::Internal,
            }
        }
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct OnPairTrainingConfig {
        pub bits: u32,
        pub threshold: f64,
        pub seed: u64,
    }

    unsafe extern "C" {
        pub fn onpair_column_compress(
            bytes: *const u8,
            offsets: *const u64,
            n: usize,
            config: OnPairTrainingConfig,
            out_handle: *mut *mut OnPairColumnHandle,
        ) -> u32;

        pub fn onpair_column_deserialize(
            data: *const u8,
            len: usize,
            out_handle: *mut *mut OnPairColumnHandle,
        ) -> u32;

        pub fn onpair_column_serialize(
            handle: *const OnPairColumnHandle,
            out_data: *mut *mut u8,
            out_len: *mut usize,
        ) -> u32;

        pub fn onpair_column_free(handle: *mut OnPairColumnHandle);
        pub fn onpair_buffer_free(data: *mut u8, len: usize);

        pub fn onpair_column_len(handle: *const OnPairColumnHandle) -> usize;
        pub fn onpair_column_bits(handle: *const OnPairColumnHandle) -> u32;
        pub fn onpair_column_dict_size(handle: *const OnPairColumnHandle) -> usize;
        pub fn onpair_column_decompress_capacity(handle: *const OnPairColumnHandle) -> usize;
        pub fn onpair_column_dict_bytes(handle: *const OnPairColumnHandle) -> usize;

        pub fn onpair_column_decompress(
            handle: *const OnPairColumnHandle,
            row_id: usize,
            out_buf: *mut u8,
            out_capacity: usize,
            out_len: *mut usize,
        ) -> u32;

        pub fn onpair_column_equals_into(
            handle: *const OnPairColumnHandle,
            needle: *const u8,
            needle_len: usize,
            out_bits: *mut u8,
        ) -> u32;

        pub fn onpair_column_starts_with_into(
            handle: *const OnPairColumnHandle,
            needle: *const u8,
            needle_len: usize,
            out_bits: *mut u8,
        ) -> u32;

        pub fn onpair_column_contains_into(
            handle: *const OnPairColumnHandle,
            needle: *const u8,
            needle_len: usize,
            out_bits: *mut u8,
        ) -> u32;

        pub fn onpair_column_dict_copy(
            handle: *const OnPairColumnHandle,
            out_bytes: *mut u8,
            bytes_capacity: usize,
            out_offsets: *mut u64,
        ) -> u32;

        pub fn onpair_column_parts(
            handle: *const OnPairColumnHandle,
            out_parts: *mut OnPairColumnParts,
        ) -> u32;
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct OnPairColumnParts {
        pub dict_bytes: *const u8,
        pub dict_bytes_len: usize,
        pub dict_offsets: *const u32,
        pub dict_offsets_len: usize,
        pub codes_packed: *const u64,
        pub codes_packed_u64_len: usize,
        pub codes_boundaries: *const u32,
        pub codes_boundaries_len: usize,
        pub bits: u32,
        pub num_rows: usize,
    }
}

pub use ffi::*;

/// The "dict-12" preset: 12-bit packed token codes.
pub const DEFAULT_DICT12_CONFIG: OnPairTrainingConfig = OnPairTrainingConfig {
    bits: 12,
    threshold: 0.5,
    seed: 0,
};

/// Error type returned by the safe wrappers.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Error {
    InvalidArg,
    BadFormat,
    OutOfRange,
    Oom,
    Internal,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Error::InvalidArg => "OnPair: invalid argument",
            Error::BadFormat => "OnPair: bad serialized format",
            Error::OutOfRange => "OnPair: row index out of range",
            Error::Oom => "OnPair: out of memory or buffer too small",
            Error::Internal => "OnPair: internal error",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for Error {}

impl Error {
    fn check(status: u32) -> Result<(), Self> {
        match OnPairStatus::from_raw(status) {
            OnPairStatus::Ok => Ok(()),
            OnPairStatus::InvalidArg => Err(Error::InvalidArg),
            OnPairStatus::BadFormat => Err(Error::BadFormat),
            OnPairStatus::OutOfRange => Err(Error::OutOfRange),
            OnPairStatus::Oom => Err(Error::Oom),
            OnPairStatus::Internal => Err(Error::Internal),
        }
    }
}

/// Owning handle around a `OnPairColumn`. Send + Sync because the C++ object
/// is immutable once constructed and the predicate methods are read-only.
pub struct Column {
    handle: NonNull<OnPairColumnHandle>,
}

unsafe impl Send for Column {}
unsafe impl Sync for Column {}

impl Column {
    /// Compress `n` byte strings described by a flat `bytes` blob and an
    /// `offsets` array of length `n + 1`.
    pub fn compress(
        bytes: &[u8],
        offsets: &[u64],
        config: OnPairTrainingConfig,
    ) -> Result<Self, Error> {
        if offsets.is_empty() || offsets.len() - 1 > offsets.len() {
            return Err(Error::InvalidArg);
        }
        let n = offsets.len() - 1;
        let mut out: *mut OnPairColumnHandle = std::ptr::null_mut();
        let status = unsafe {
            onpair_column_compress(bytes.as_ptr(), offsets.as_ptr(), n, config, &raw mut out)
        };
        Error::check(status)?;
        let handle = NonNull::new(out).ok_or(Error::Internal)?;
        Ok(Self { handle })
    }

    /// Reconstruct a column from a previously-serialised byte blob.
    pub fn from_bytes(data: &[u8]) -> Result<Self, Error> {
        let mut out: *mut OnPairColumnHandle = std::ptr::null_mut();
        let status = unsafe { onpair_column_deserialize(data.as_ptr(), data.len(), &raw mut out) };
        Error::check(status)?;
        let handle = NonNull::new(out).ok_or(Error::Internal)?;
        Ok(Self { handle })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut data: *mut u8 = std::ptr::null_mut();
        let mut len: usize = 0;
        let status =
            unsafe { onpair_column_serialize(self.handle.as_ptr(), &raw mut data, &raw mut len) };
        Error::check(status)?;
        let out = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
        unsafe { onpair_buffer_free(data, len) };
        Ok(out)
    }

    pub fn len(&self) -> usize {
        unsafe { onpair_column_len(self.handle.as_ptr()) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bits(&self) -> u32 {
        unsafe { onpair_column_bits(self.handle.as_ptr()) }
    }

    pub fn dict_size(&self) -> usize {
        unsafe { onpair_column_dict_size(self.handle.as_ptr()) }
    }

    pub fn max_decompress_capacity(&self) -> usize {
        unsafe { onpair_column_decompress_capacity(self.handle.as_ptr()) }
    }

    /// Decompress a single row, growing `out` as needed.
    pub fn decompress_row(&self, row_id: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        let capacity = self.max_decompress_capacity().max(64);
        out.clear();
        out.reserve(capacity);
        let mut written: usize = 0;
        let status = unsafe {
            onpair_column_decompress(
                self.handle.as_ptr(),
                row_id,
                out.as_mut_ptr(),
                out.capacity(),
                &raw mut written,
            )
        };
        Error::check(status)?;
        unsafe { out.set_len(written) };
        Ok(())
    }

    pub fn dict_bytes(&self) -> usize {
        unsafe { onpair_column_dict_bytes(self.handle.as_ptr()) }
    }

    /// Materialise the dictionary as `(bytes, offsets)`. `offsets` has length
    /// `dict_size + 1`.
    pub fn dict(&self) -> Result<(Vec<u8>, Vec<u64>), Error> {
        let dict_size = self.dict_size();
        let bytes_len = self.dict_bytes();
        let mut bytes = vec![0u8; bytes_len];
        let mut offsets = vec![0u64; dict_size + 1];
        let status = unsafe {
            onpair_column_dict_copy(
                self.handle.as_ptr(),
                bytes.as_mut_ptr(),
                bytes.len(),
                offsets.as_mut_ptr(),
            )
        };
        Error::check(status)?;
        Ok((bytes, offsets))
    }

    fn run_predicate(
        &self,
        f: unsafe extern "C" fn(*const OnPairColumnHandle, *const u8, usize, *mut u8) -> u32,
        needle: &[u8],
    ) -> Result<Vec<u8>, Error> {
        let n = self.len();
        let mut bits = vec![0u8; n.div_ceil(8)];
        let status = unsafe {
            f(
                self.handle.as_ptr(),
                needle.as_ptr(),
                needle.len(),
                bits.as_mut_ptr(),
            )
        };
        Error::check(status)?;
        Ok(bits)
    }

    pub fn equals_bitmap(&self, needle: &[u8]) -> Result<Vec<u8>, Error> {
        self.run_predicate(onpair_column_equals_into, needle)
    }

    pub fn starts_with_bitmap(&self, needle: &[u8]) -> Result<Vec<u8>, Error> {
        self.run_predicate(onpair_column_starts_with_into, needle)
    }

    pub fn contains_bitmap(&self, needle: &[u8]) -> Result<Vec<u8>, Error> {
        self.run_predicate(onpair_column_contains_into, needle)
    }

    /// Raw handle exposed for higher-level wrappers that need to pass the
    /// pointer to their own FFI calls.
    ///
    /// # Safety
    ///
    /// The returned pointer is owned by `self`; callers must not free it,
    /// must not dereference it through any FFI other than the `onpair_*`
    /// functions, and must not let it outlive this [`Column`].
    pub unsafe fn raw(&self) -> *const c_void {
        self.handle.as_ptr() as *const c_void
    }
}

impl Column {
    /// Borrow the column's raw decomposition: dictionary, bit-packed token
    /// stream, and per-row boundaries. The returned pointers reference memory
    /// owned by `self` and remain valid for as long as the column does.
    pub fn parts(&self) -> Result<Parts<'_>, Error> {
        let mut raw = OnPairColumnParts {
            dict_bytes: std::ptr::null(),
            dict_bytes_len: 0,
            dict_offsets: std::ptr::null(),
            dict_offsets_len: 0,
            codes_packed: std::ptr::null(),
            codes_packed_u64_len: 0,
            codes_boundaries: std::ptr::null(),
            codes_boundaries_len: 0,
            bits: 0,
            num_rows: 0,
        };
        let status = unsafe { onpair_column_parts(self.handle.as_ptr(), &raw mut raw) };
        Error::check(status)?;
        // SAFETY: the C side returns pointers into vectors owned by `self`
        // (the underlying `OnPairColumn`); they remain valid for `&self`.
        Ok(unsafe { Parts::from_raw(raw) })
    }
}

impl Drop for Column {
    fn drop(&mut self) {
        unsafe { onpair_column_free(self.handle.as_ptr()) }
    }
}

/// Borrowed view over a column's raw arrays. See [`Column::parts`].
#[derive(Copy, Clone)]
pub struct Parts<'a> {
    /// Concatenated dictionary entry bytes (unpadded).
    pub dict_bytes: &'a [u8],
    /// Length `dict_size + 1`; entry `i` spans `dict_bytes[dict_offsets[i]..dict_offsets[i + 1]]`.
    pub dict_offsets: &'a [u32],
    /// LSB-first bit-packed token stream, packed `bits` bits per token.
    pub codes_packed: &'a [u64],
    /// Length `num_rows + 1`; row `r` spans tokens `codes_boundaries[r]..codes_boundaries[r + 1]`.
    pub codes_boundaries: &'a [u32],
    /// Bits per token (9..=16).
    pub bits: u32,
    pub num_rows: usize,
}

impl<'a> Parts<'a> {
    /// # Safety
    /// Caller must guarantee the pointers in `raw` are valid for `'a`.
    unsafe fn from_raw(raw: OnPairColumnParts) -> Self {
        unsafe {
            Self {
                dict_bytes: slice_or_empty(raw.dict_bytes, raw.dict_bytes_len),
                dict_offsets: slice_or_empty(raw.dict_offsets, raw.dict_offsets_len),
                codes_packed: slice_or_empty(raw.codes_packed, raw.codes_packed_u64_len),
                codes_boundaries: slice_or_empty(raw.codes_boundaries, raw.codes_boundaries_len),
                bits: raw.bits,
                num_rows: raw.num_rows,
            }
        }
    }
}

#[inline]
unsafe fn slice_or_empty<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// Read `bits` (1..=16) bits from `packed` starting at LSB-first bit position
/// `bit_pos`. Matches OnPair's `BitWriter` layout.
#[inline]
pub fn read_bits_lsb(packed: &[u64], bit_pos: usize, bits: u32) -> u16 {
    debug_assert!((1..=16).contains(&bits));
    let word_idx = bit_pos / 64;
    // SAFETY of cast: `bit_pos % 64` is always in `0..64`, which fits in u32.
    #[allow(clippy::cast_possible_truncation)]
    let bit_off = (bit_pos % 64) as u32;
    let mask: u64 = (1u64 << bits) - 1;
    let low = packed[word_idx] >> bit_off;
    let combined = if bit_off + bits <= 64 {
        low & mask
    } else {
        let high = packed[word_idx + 1] << (64 - bit_off);
        (low | high) & mask
    };
    // SAFETY of cast: `combined` has been masked to at most `bits` (<=16) bits.
    #[allow(clippy::cast_possible_truncation)]
    let value = combined as u16;
    value
}

/// Decompress an LSB-first bit-packed token stream into a flat `Vec<u16>`,
/// one element per token. Each `u16` only uses its low `bits` bits.
pub fn unpack_codes_to_u16(packed: &[u64], total_tokens: usize, bits: u32) -> Vec<u16> {
    assert!((9..=16).contains(&bits), "bits must be in [9, 16]");
    let mut out = Vec::with_capacity(total_tokens);
    for t in 0..total_tokens {
        out.push(read_bits_lsb(packed, t * bits as usize, bits));
    }
    out
}
