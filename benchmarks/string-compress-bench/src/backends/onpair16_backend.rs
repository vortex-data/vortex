// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Backend wrapping `onpair_rs::OnPair16`. Same API shape as `OnPair` but
//! the dictionary entries are capped at 16 bytes so parsing can use a tight
//! SIMD-friendly inner loop.

use std::sync::Mutex;

use onpair_rs::OnPair16;

use super::{Backend, Pushdown};

pub struct OnPair16Backend {
    inner: Mutex<OnPair16>,
    lengths: Vec<usize>,
    decode_padding: usize,
    space_used: usize,
}

impl OnPair16Backend {
    pub fn train_and_compress(strings: &[Vec<u8>], threshold: u16) -> Self {
        let mut inner = OnPair16::new(threshold);
        let as_strs: Vec<&str> = strings
            .iter()
            .map(|s| std::str::from_utf8(s).expect("synthetic data is utf-8"))
            .collect();
        inner.compress_strings(&as_strs);
        inner.shrink_to_fit();

        let lengths: Vec<usize> = strings.iter().map(|s| s.len()).collect();
        let space_used = inner.space_used();
        Self { inner: Mutex::new(inner), lengths, decode_padding: 32, space_used }
    }

    fn scratch_capacity(&self) -> usize {
        self.lengths.iter().copied().max().unwrap_or(0) + self.decode_padding
    }
}

impl Backend for OnPair16Backend {
    fn name(&self) -> &'static str {
        "onpair16"
    }

    fn compressed_payload_bytes(&self) -> usize {
        self.space_used
    }

    fn total_compressed_bytes(&self) -> usize {
        self.space_used + self.lengths.len() * size_of::<u64>()
    }

    fn decompress_all(&self) -> Vec<Vec<u8>> {
        let mut out: Vec<Vec<u8>> = Vec::with_capacity(self.lengths.len());
        let mut scratch = vec![0u8; self.scratch_capacity()];
        let mut inner = self.inner.lock().expect("onpair16 mutex poisoned");
        for (i, &true_len) in self.lengths.iter().enumerate() {
            let written = inner.decompress_string(i, &mut scratch);
            let len = written.min(true_len.max(written));
            out.push(scratch[..len.min(true_len)].to_vec());
        }
        out
    }
}

impl Pushdown for OnPair16Backend {
    fn equals(&self, needle: &[u8]) -> Vec<usize> {
        let mut buf = vec![0u8; self.scratch_capacity()];
        let mut inner = self.inner.lock().expect("onpair16 mutex poisoned");
        self.lengths
            .iter()
            .enumerate()
            .filter_map(|(i, &true_len)| {
                let written = inner.decompress_string(i, &mut buf);
                let len = written.min(true_len);
                (len == needle.len() && &buf[..len] == needle).then_some(i)
            })
            .collect()
    }

    fn contains(&self, needle: &[u8]) -> Vec<usize> {
        let mut buf = vec![0u8; self.scratch_capacity()];
        let mut inner = self.inner.lock().expect("onpair16 mutex poisoned");
        self.lengths
            .iter()
            .enumerate()
            .filter_map(|(i, &true_len)| {
                let written = inner.decompress_string(i, &mut buf);
                let len = written.min(true_len);
                buf[..len].windows(needle.len()).any(|w| w == needle).then_some(i)
            })
            .collect()
    }

    fn starts_with(&self, prefix: &[u8]) -> Vec<usize> {
        let mut buf = vec![0u8; self.scratch_capacity()];
        let mut inner = self.inner.lock().expect("onpair16 mutex poisoned");
        self.lengths
            .iter()
            .enumerate()
            .filter_map(|(i, &true_len)| {
                let written = inner.decompress_string(i, &mut buf);
                let len = written.min(true_len);
                (len >= prefix.len() && &buf[..prefix.len()] == prefix).then_some(i)
            })
            .collect()
    }
}
