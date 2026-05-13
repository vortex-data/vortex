// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Raw FFI bindings for `cpp/onpair_cpp_wrapper.cpp`.

use std::ffi::c_void;
use std::os::raw::{c_char, c_uchar};

#[repr(C)]
pub struct OnPairIndexVec {
    pub data: *mut usize,
    pub len: usize,
}

pub type OnPairHandle = c_void;

unsafe extern "C" {
    pub fn onpair_compress(
        data: *const c_uchar,
        offsets: *const u32,
        n_strings: usize,
        bits: u8,
        seed: u32,
    ) -> *mut OnPairHandle;

    pub fn onpair_destroy(h: *mut OnPairHandle);

    pub fn onpair_bytes_used(h: *const OnPairHandle) -> usize;

    pub fn onpair_num_strings(h: *const OnPairHandle) -> usize;

    pub fn onpair_decompress_all(
        h: *const OnPairHandle,
        out_data: *mut c_uchar,
        out_offsets: *mut u32,
    ) -> usize;

    pub fn onpair_equals(
        h: *const OnPairHandle,
        needle: *const c_char,
        needle_len: usize,
    ) -> OnPairIndexVec;

    pub fn onpair_contains(
        h: *const OnPairHandle,
        needle: *const c_char,
        needle_len: usize,
    ) -> OnPairIndexVec;

    pub fn onpair_starts_with(
        h: *const OnPairHandle,
        prefix: *const c_char,
        prefix_len: usize,
    ) -> OnPairIndexVec;

    pub fn onpair_free_indices(v: OnPairIndexVec);

    pub fn onpair_decompress_padding() -> usize;
}
