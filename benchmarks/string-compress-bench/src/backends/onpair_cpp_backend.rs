// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Backend bound to the upstream C++ `onpair_cpp` library.
//!
//! Unlike the Rust port, this backend can do true compressed-domain equality
//! and substring search via the library's KMP / EqAutomaton scanners.

use std::os::raw::c_char;

use super::onpair_cpp_ffi::*;
use super::{Backend, Pushdown};

pub struct OnPairCppBackend {
    handle: *mut OnPairHandle,
    /// Per-row uncompressed lengths from the input; we need them only to
    /// drop overshoot from the decompress buffer.
    lengths: Vec<usize>,
    decode_padding: usize,
}

unsafe impl Send for OnPairCppBackend {}
unsafe impl Sync for OnPairCppBackend {}

impl OnPairCppBackend {
    pub fn train_and_compress(strings: &[Vec<u8>], bits: u8, seed: u32) -> Self {
        // Build a flat byte buffer + offsets prefix-sum, the canonical input
        // shape the C++ API exposes.
        let mut offsets: Vec<u32> = Vec::with_capacity(strings.len() + 1);
        offsets.push(0);
        let mut data: Vec<u8> = Vec::new();
        for s in strings {
            data.extend_from_slice(s);
            offsets.push(u32::try_from(data.len()).expect("offsets fit in u32"));
        }

        let handle = unsafe {
            onpair_compress(data.as_ptr(), offsets.as_ptr(), strings.len(), bits, seed)
        };
        assert!(!handle.is_null(), "onpair_compress returned null");

        let decode_padding = unsafe { onpair_decompress_padding() };
        let lengths = strings.iter().map(|s| s.len()).collect();
        Self { handle, lengths, decode_padding }
    }
}

impl Drop for OnPairCppBackend {
    fn drop(&mut self) {
        unsafe { onpair_destroy(self.handle) };
    }
}

impl Backend for OnPairCppBackend {
    fn name(&self) -> &'static str {
        "onpair-cpp"
    }

    fn compressed_payload_bytes(&self) -> usize {
        unsafe { onpair_bytes_used(self.handle) }
    }

    fn total_compressed_bytes(&self) -> usize {
        // `bytes_used` already covers the store + dictionary; serialised
        // offsets add `n+1` u32 entries.
        self.compressed_payload_bytes() + (self.lengths.len() + 1) * size_of::<u32>()
    }

    fn decompress_all(&self) -> Vec<Vec<u8>> {
        let total: usize = self.lengths.iter().sum();
        let cap = total + self.decode_padding;
        let mut buf = vec![0u8; cap];
        let mut offsets = vec![0u32; self.lengths.len() + 1];
        let written = unsafe {
            onpair_decompress_all(
                self.handle,
                buf.as_mut_ptr(),
                offsets.as_mut_ptr(),
            )
        };
        debug_assert!(written <= buf.len());

        let mut out = Vec::with_capacity(self.lengths.len());
        for w in offsets.windows(2) {
            let lo = w[0] as usize;
            let hi = w[1] as usize;
            out.push(buf[lo..hi].to_vec());
        }
        out
    }
}

impl Pushdown for OnPairCppBackend {
    fn equals(&self, needle: &[u8]) -> Vec<usize> {
        let v = unsafe {
            onpair_equals(
                self.handle,
                needle.as_ptr().cast::<c_char>(),
                needle.len(),
            )
        };
        take_indices(v)
    }

    fn contains(&self, needle: &[u8]) -> Vec<usize> {
        let v = unsafe {
            onpair_contains(
                self.handle,
                needle.as_ptr().cast::<c_char>(),
                needle.len(),
            )
        };
        take_indices(v)
    }

    fn starts_with(&self, prefix: &[u8]) -> Vec<usize> {
        let v = unsafe {
            onpair_starts_with(
                self.handle,
                prefix.as_ptr().cast::<c_char>(),
                prefix.len(),
            )
        };
        take_indices(v)
    }
}

/// Copy the indices out of the C-allocated buffer and immediately free the
/// original. Keeps the Rust side from learning C++ allocator semantics.
fn take_indices(v: OnPairIndexVec) -> Vec<usize> {
    let out = if v.data.is_null() || v.len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(v.data, v.len).to_vec() }
    };
    unsafe { onpair_free_indices(v) };
    out
}
