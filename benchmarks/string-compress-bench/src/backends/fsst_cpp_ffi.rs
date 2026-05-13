// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Raw FFI bindings for the vendored C++ FSST library.
//!
//! Both the 8-bit and 12-bit symbol-table variants are linked into the same
//! binary by renaming their `extern "C"` symbols (see `cpp/fsst8_wrapper.cpp`
//! and `cpp/fsst12_wrapper.cpp`). The two `fsst_decoder_t` shapes differ in
//! layout, so each variant exposes its own struct.

use std::ffi::c_void;
use std::os::raw::{c_int, c_uchar};

/// Opaque encoder handle. Owned by the C++ library; freed with `*_destroy`.
pub type EncoderHandle = *mut c_void;

/// FSST-8 decoder layout. Matches the typedef in `vendor/fsst_cpp/fsst.h`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Fsst8Decoder {
    pub version: u64,
    pub zero_terminated: u8,
    pub len: [u8; 255],
    pub symbol: [u64; 255],
}

impl Fsst8Decoder {
    pub fn zeroed() -> Self {
        Self { version: 0, zero_terminated: 0, len: [0; 255], symbol: [0; 255] }
    }
}

/// FSST-12 decoder layout. Matches the typedef in `vendor/fsst_cpp/fsst12.h`
/// (it stores ~4096 symbols).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct Fsst12Decoder {
    pub version: u64,
    pub zero_terminated: u8,
    pub len: [u8; 4096],
    pub symbol: [u64; 4096],
}

impl Fsst12Decoder {
    pub fn zeroed() -> Self {
        Self { version: 0, zero_terminated: 0, len: [0; 4096], symbol: [0; 4096] }
    }
}

unsafe extern "C" {
    // ----- FSST-8 -----
    pub fn fsst8_create(
        n: usize,
        len_in: *const usize,
        str_in: *const *const c_uchar,
        zero_terminated: c_int,
    ) -> EncoderHandle;

    pub fn fsst8_destroy(encoder: EncoderHandle);

    pub fn fsst8_decoder(encoder: EncoderHandle) -> Fsst8Decoder;

    pub fn fsst8_compress(
        encoder: EncoderHandle,
        n_strings: usize,
        len_in: *const usize,
        str_in: *const *const c_uchar,
        out_size: usize,
        output: *mut c_uchar,
        len_out: *mut usize,
        str_out: *mut *mut c_uchar,
    ) -> usize;

    pub fn fsst8_decompress_export(
        decoder: *const Fsst8Decoder,
        len_in: usize,
        str_in: *const c_uchar,
        size: usize,
        output: *mut c_uchar,
    ) -> usize;

    // ----- FSST-12 -----
    pub fn fsst12_create(
        n: usize,
        len_in: *const usize,
        str_in: *const *const c_uchar,
        zero_terminated: c_int,
    ) -> EncoderHandle;

    pub fn fsst12_destroy(encoder: EncoderHandle);

    pub fn fsst12_decoder(encoder: EncoderHandle) -> Fsst12Decoder;

    pub fn fsst12_compress(
        encoder: EncoderHandle,
        n_strings: usize,
        len_in: *const usize,
        str_in: *const *const c_uchar,
        out_size: usize,
        output: *mut c_uchar,
        len_out: *mut usize,
        str_out: *mut *mut c_uchar,
    ) -> usize;

    pub fn fsst12_decompress_export(
        decoder: *const Fsst12Decoder,
        len_in: usize,
        str_in: *const c_uchar,
        size: usize,
        output: *mut c_uchar,
    ) -> usize;
}
