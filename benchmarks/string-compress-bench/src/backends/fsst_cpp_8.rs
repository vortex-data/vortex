// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Backend bound to the upstream C++ FSST-8 implementation (8-bit codes,
//! 255 symbols + 1 escape).

use std::os::raw::c_uchar;

use super::fsst_cpp_ffi::*;
use super::{Backend, Pushdown};

pub struct FsstCpp8Backend {
    decoder: Fsst8Decoder,
    codes: Vec<Vec<u8>>,
    /// Cached compressor handle so equality pushdown can re-encode needles
    /// without re-training.
    encoder: EncoderHandle,
}

// The encoder is an opaque C++ object; the bench harness only calls into it
// from a single thread per benchmark, so these unsafe impls are sound. divan
// requires `Sync` on bench closures so the harness can shard benches across
// threads; each FsstCpp8Backend instance still serves at most one thread at
// a time.
unsafe impl Send for FsstCpp8Backend {}
unsafe impl Sync for FsstCpp8Backend {}

impl FsstCpp8Backend {
    pub fn train_and_compress(strings: &[Vec<u8>]) -> Self {
        let n = strings.len();
        let len_in: Vec<usize> = strings.iter().map(|s| s.len()).collect();
        let str_in: Vec<*const c_uchar> = strings.iter().map(|s| s.as_ptr()).collect();

        unsafe {
            let encoder = fsst8_create(n, len_in.as_ptr(), str_in.as_ptr(), 0);
            assert!(!encoder.is_null(), "fsst8_create returned null");
            let decoder = fsst8_decoder(encoder);
            let codes = compress_one_at_a_time_fsst8(encoder, strings);
            Self { decoder, codes, encoder }
        }
    }
}

impl Drop for FsstCpp8Backend {
    fn drop(&mut self) {
        unsafe { fsst8_destroy(self.encoder) };
    }
}

impl Backend for FsstCpp8Backend {
    fn name(&self) -> &'static str {
        "fsst-cpp-8"
    }

    fn compressed_payload_bytes(&self) -> usize {
        self.codes.iter().map(|c| c.len()).sum()
    }

    fn total_compressed_bytes(&self) -> usize {
        // Codes + serialised decoder (FSST_MAXHEADER ~ 2066) + per-row offsets.
        self.compressed_payload_bytes()
            + size_of::<Fsst8Decoder>()
            + self.codes.len() * size_of::<u32>()
    }

    fn decompress_all(&self) -> Vec<Vec<u8>> {
        self.codes
            .iter()
            .map(|c| {
                // 7 + 2*lenIn is the worst-case decompressed size for an
                // FSST-encoded string (each escape costs 2 bytes in, 1 out).
                let cap = c.len().saturating_mul(8) + 32;
                let mut out = vec![0u8; cap];
                let n = unsafe {
                    fsst8_decompress_export(
                        &self.decoder,
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

impl Pushdown for FsstCpp8Backend {
    fn equals(&self, needle: &[u8]) -> Vec<usize> {
        let target = compress_one_fsst8(self.encoder, needle);
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| (c.as_slice() == target.as_slice()).then_some(i))
            .collect()
    }

    fn contains(&self, needle: &[u8]) -> Vec<usize> {
        // FSST has no substring-on-compressed primitive, so decompress per
        // row. This mirrors the fallback path callers see in production
        // engines.
        let decoder = &self.decoder;
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let cap = c.len().saturating_mul(8) + 32;
                let mut buf = vec![0u8; cap];
                let n = unsafe {
                    fsst8_decompress_export(
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
        let decoder = &self.decoder;
        self.codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let cap = c.len().saturating_mul(8) + 32;
                let mut buf = vec![0u8; cap];
                let n = unsafe {
                    fsst8_decompress_export(
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

unsafe fn compress_one_at_a_time_fsst8(
    encoder: EncoderHandle,
    strings: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let n = strings.len();
    let len_in: Vec<usize> = strings.iter().map(|s| s.len()).collect();
    let str_in: Vec<*const c_uchar> = strings.iter().map(|s| s.as_ptr()).collect();
    // Conservative output buffer: 7 + 2*lenIn per string is the upper bound.
    let total_in: usize = len_in.iter().sum();
    let cap = total_in.saturating_mul(2) + 7 * n.max(1) + 64;
    let mut output = vec![0u8; cap];
    let mut len_out = vec![0usize; n];
    let mut str_out = vec![std::ptr::null_mut::<c_uchar>(); n];

    let written = unsafe {
        fsst8_compress(
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
    assert_eq!(written, n, "fsst8_compress only consumed {written}/{n} strings");

    let mut codes = Vec::with_capacity(n);
    for i in 0..n {
        let len = len_out[i];
        if len == 0 {
            codes.push(Vec::new());
        } else {
            let p = str_out[i];
            assert!(!p.is_null());
            // Copy out: the caller can drop `output` once we return.
            let slice = unsafe { std::slice::from_raw_parts(p, len) };
            codes.push(slice.to_vec());
        }
    }
    codes
}

fn compress_one_fsst8(encoder: EncoderHandle, s: &[u8]) -> Vec<u8> {
    let len_in = [s.len()];
    let ptr = [s.as_ptr()];
    let cap = s.len() * 2 + 64;
    let mut out = vec![0u8; cap];
    let mut len_out = [0usize; 1];
    let mut str_out = [std::ptr::null_mut::<c_uchar>(); 1];
    let written = unsafe {
        fsst8_compress(
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
