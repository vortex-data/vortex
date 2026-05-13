// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Backend bound to the upstream C++ FSST-12 implementation (12-bit codes,
//! up to ~4096 symbols, 1.5 bytes per code in the stream).

use std::os::raw::c_uchar;

use super::fsst_cpp_ffi::*;
use super::{Backend, Pushdown};

pub struct FsstCpp12Backend {
    decoder: Box<Fsst12Decoder>,
    codes: Vec<Vec<u8>>,
    encoder: EncoderHandle,
}

unsafe impl Send for FsstCpp12Backend {}
unsafe impl Sync for FsstCpp12Backend {}

impl FsstCpp12Backend {
    pub fn train_and_compress(strings: &[Vec<u8>]) -> Self {
        let n = strings.len();
        let len_in: Vec<usize> = strings.iter().map(|s| s.len()).collect();
        let str_in: Vec<*const c_uchar> = strings.iter().map(|s| s.as_ptr()).collect();

        unsafe {
            let encoder = fsst12_create(n, len_in.as_ptr(), str_in.as_ptr(), 0);
            assert!(!encoder.is_null(), "fsst12_create returned null");
            // The 4096-entry decoder is large (~32 KB); keep it heap-allocated
            // so we do not blow the bench-thread stack.
            let decoder = Box::new(fsst12_decoder(encoder));
            let codes = compress_all_fsst12(encoder, strings);
            Self { decoder, codes, encoder }
        }
    }
}

impl Drop for FsstCpp12Backend {
    fn drop(&mut self) {
        unsafe { fsst12_destroy(self.encoder) };
    }
}

impl Backend for FsstCpp12Backend {
    fn name(&self) -> &'static str {
        "fsst-cpp-12"
    }

    fn compressed_payload_bytes(&self) -> usize {
        self.codes.iter().map(|c| c.len()).sum()
    }

    fn total_compressed_bytes(&self) -> usize {
        // The serialised header is much smaller than the in-memory decoder
        // (per upstream: `8 + 1 + 8 + 2048 + 1` = 2066 bytes is the
        // serialisation limit for the table). Use that figure here.
        const FSST_MAXHEADER: usize = 8 + 1 + 8 + 2048 + 1;
        self.compressed_payload_bytes()
            + FSST_MAXHEADER
            + self.codes.len() * size_of::<u32>()
    }

    fn decompress_all(&self) -> Vec<Vec<u8>> {
        self.codes
            .iter()
            .map(|c| {
                let cap = c.len().saturating_mul(12) + 32;
                let mut out = vec![0u8; cap];
                let n = unsafe {
                    fsst12_decompress_export(
                        &*self.decoder,
                        c.len(),
                        c.as_ptr(),
                        out.len(),
                        out.as_mut_ptr(),
                    )
                };
                out.truncate(n);
                out
            })
            .collect()
    }
}

impl Pushdown for FsstCpp12Backend {
    fn equals(&self, needle: &[u8]) -> Vec<usize> {
        let target = compress_one_fsst12(self.encoder, needle);
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| (c.as_slice() == target.as_slice()).then_some(i))
            .collect()
    }

    fn contains(&self, needle: &[u8]) -> Vec<usize> {
        let decoder = &*self.decoder;
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let cap = c.len().saturating_mul(12) + 32;
                let mut buf = vec![0u8; cap];
                let n = unsafe {
                    fsst12_decompress_export(
                        decoder,
                        c.len(),
                        c.as_ptr(),
                        buf.len(),
                        buf.as_mut_ptr(),
                    )
                };
                buf.truncate(n);
                buf.windows(needle.len()).any(|w| w == needle).then_some(i)
            })
            .collect()
    }

    fn starts_with(&self, prefix: &[u8]) -> Vec<usize> {
        let decoder = &*self.decoder;
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let cap = c.len().saturating_mul(12) + 32;
                let mut buf = vec![0u8; cap];
                let n = unsafe {
                    fsst12_decompress_export(
                        decoder,
                        c.len(),
                        c.as_ptr(),
                        buf.len(),
                        buf.as_mut_ptr(),
                    )
                };
                buf.truncate(n);
                (buf.len() >= prefix.len() && &buf[..prefix.len()] == prefix).then_some(i)
            })
            .collect()
    }
}

unsafe fn compress_all_fsst12(encoder: EncoderHandle, strings: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let n = strings.len();
    let len_in: Vec<usize> = strings.iter().map(|s| s.len()).collect();
    let str_in: Vec<*const c_uchar> = strings.iter().map(|s| s.as_ptr()).collect();
    let total_in: usize = len_in.iter().sum();
    // 12-bit codes pack roughly 1.5 bytes per token; budget generously.
    let cap = total_in.saturating_mul(3) + 32 * n.max(1) + 64;
    let mut output = vec![0u8; cap];
    let mut len_out = vec![0usize; n];
    let mut str_out = vec![std::ptr::null_mut::<c_uchar>(); n];

    let written = unsafe {
        fsst12_compress(
            encoder,
            n,
            len_in.as_ptr(),
            str_in.as_ptr(),
            output.len(),
            output.as_mut_ptr(),
            len_out.as_mut_ptr(),
            str_out.as_mut_ptr(),
        )
    };
    assert_eq!(written, n, "fsst12_compress only consumed {written}/{n} strings");

    let mut codes = Vec::with_capacity(n);
    for i in 0..n {
        let len = len_out[i];
        if len == 0 {
            codes.push(Vec::new());
        } else {
            let p = str_out[i];
            assert!(!p.is_null());
            let slice = unsafe { std::slice::from_raw_parts(p, len) };
            codes.push(slice.to_vec());
        }
    }
    codes
}

fn compress_one_fsst12(encoder: EncoderHandle, s: &[u8]) -> Vec<u8> {
    let len_in = [s.len()];
    let ptr = [s.as_ptr()];
    let cap = s.len() * 3 + 64;
    let mut out = vec![0u8; cap];
    let mut len_out = [0usize; 1];
    let mut str_out = [std::ptr::null_mut::<c_uchar>(); 1];
    let written = unsafe {
        fsst12_compress(
            encoder,
            1,
            len_in.as_ptr(),
            ptr.as_ptr(),
            out.len(),
            out.as_mut_ptr(),
            len_out.as_mut_ptr(),
            str_out.as_mut_ptr(),
        )
    };
    assert_eq!(written, 1);
    let n = len_out[0];
    if n == 0 {
        return Vec::new();
    }
    let p = str_out[0];
    unsafe { std::slice::from_raw_parts(p, n).to_vec() }
}
