// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized FSST decompressor that replaces the default fsst-rs decompressor
//! with a version tuned for throughput.
//!
//! Key optimizations over the baseline fsst-rs implementation:
//! 1. Symbols stored as `u64` directly, avoiding `Symbol::to_u64()` conversion per lookup.
//! 2. Multi-level block processing: 32-code, 16-code, and 8-code fast paths that process
//!    compressed data in large chunks when no escape codes are present.
//! 3. Fully unrolled escape handling via match statement for optimal branch prediction.

use std::mem::MaybeUninit;

use fsst::ESCAPE_CODE;
use fsst::Symbol;

/// Optimized FSST decompressor using separate symbol/length tables.
///
/// The symbol table stores pre-converted `u64` values to avoid per-lookup
/// conversion overhead. Separate arrays keep the cache footprint small:
/// symbols (2KB) + lengths (256B) ≈ 2.3KB, fitting entirely in L1 cache.
pub struct OptimizedDecompressor {
    /// Symbol values indexed by code (0-255). Each value is the symbol's bytes
    /// packed into a little-endian u64.
    symbols: Box<[u64; 256]>,
    /// Symbol lengths indexed by code (0-255). Each value is 1-8.
    lengths: Box<[u8; 256]>,
}

impl OptimizedDecompressor {
    /// Build from symbol table slices (same inputs as `fsst::Decompressor::new`).
    pub fn new(symbols: &[Symbol], lengths: &[u8]) -> Self {
        assert!(
            symbols.len() <= 255,
            "symbol table cannot exceed 255 entries"
        );
        assert_eq!(symbols.len(), lengths.len());

        let mut sym_table = Box::new([0u64; 256]);
        let mut len_table = Box::new([1u8; 256]);
        for (i, (sym, &len)) in symbols.iter().zip(lengths.iter()).enumerate() {
            sym_table[i] = sym.to_u64();
            len_table[i] = len;
        }
        Self {
            symbols: sym_table,
            lengths: len_table,
        }
    }

    /// Decompress `compressed` codes into `decoded` buffer.
    ///
    /// Returns the number of bytes written to `decoded`.
    ///
    /// # Panics
    ///
    /// Panics if `decoded` is smaller than `compressed.len() / 2`.
    pub fn decompress_into(&self, compressed: &[u8], decoded: &mut [MaybeUninit<u8>]) -> usize {
        assert!(
            decoded.len() >= compressed.len() / 2,
            "decoded buffer too small"
        );

        // SAFETY: We carefully manage pointer bounds within the inner function.
        unsafe { self.decompress_inner(compressed, decoded) }
    }

    /// SWAR escape detection for a u64 block of 8 codes.
    /// Returns a mask with the high bit set in each byte that equals 0xFF.
    #[inline(always)]
    const fn escape_mask(block: u64) -> u64 {
        (block & 0x8080_8080_8080_8080)
            & (((!block & 0x7F7F_7F7F_7F7F_7F7F).wrapping_add(0x7F7F_7F7F_7F7F_7F7F))
                ^ 0x8080_8080_8080_8080)
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn, clippy::cast_possible_truncation)]
    unsafe fn decompress_inner(&self, compressed: &[u8], decoded: &mut [MaybeUninit<u8>]) -> usize {
        let mut in_ptr = compressed.as_ptr();
        let in_end = in_ptr.add(compressed.len());

        let mut out_ptr: *mut u8 = decoded.as_mut_ptr().cast();
        let out_begin = out_ptr.cast_const();
        let out_end = decoded.as_ptr().add(decoded.len()).cast::<u8>();

        let symbols = self.symbols.as_ptr();
        let lengths = self.lengths.as_ptr();

        macro_rules! emit_symbol {
            ($code:expr) => {{
                let c = $code as usize;
                out_ptr.cast::<u64>().write_unaligned(*symbols.add(c));
                out_ptr = out_ptr.add(*lengths.add(c) as usize);
            }};
        }

        macro_rules! emit_block {
            ($block:expr) => {{
                emit_symbol!(($block) & 0xFF);
                emit_symbol!(($block >> 8) & 0xFF);
                emit_symbol!(($block >> 16) & 0xFF);
                emit_symbol!(($block >> 24) & 0xFF);
                emit_symbol!(($block >> 32) & 0xFF);
                emit_symbol!(($block >> 40) & 0xFF);
                emit_symbol!(($block >> 48) & 0xFF);
                emit_symbol!(($block >> 56) & 0xFF);
            }};
        }

        macro_rules! handle_escape_block {
            ($block:expr, $first_esc:expr) => {
                match $first_esc {
                    7 => {
                        emit_symbol!(($block) & 0xFF);
                        emit_symbol!(($block >> 8) & 0xFF);
                        emit_symbol!(($block >> 16) & 0xFF);
                        emit_symbol!(($block >> 24) & 0xFF);
                        emit_symbol!(($block >> 32) & 0xFF);
                        emit_symbol!(($block >> 40) & 0xFF);
                        emit_symbol!(($block >> 48) & 0xFF);
                        in_ptr = in_ptr.add(7);
                    }
                    6 => {
                        emit_symbol!(($block) & 0xFF);
                        emit_symbol!(($block >> 8) & 0xFF);
                        emit_symbol!(($block >> 16) & 0xFF);
                        emit_symbol!(($block >> 24) & 0xFF);
                        emit_symbol!(($block >> 32) & 0xFF);
                        emit_symbol!(($block >> 40) & 0xFF);
                        out_ptr.write((($block >> 56) & 0xFF) as u8);
                        out_ptr = out_ptr.add(1);
                        in_ptr = in_ptr.add(8);
                    }
                    5 => {
                        emit_symbol!(($block) & 0xFF);
                        emit_symbol!(($block >> 8) & 0xFF);
                        emit_symbol!(($block >> 16) & 0xFF);
                        emit_symbol!(($block >> 24) & 0xFF);
                        emit_symbol!(($block >> 32) & 0xFF);
                        out_ptr.write((($block >> 48) & 0xFF) as u8);
                        out_ptr = out_ptr.add(1);
                        in_ptr = in_ptr.add(7);
                    }
                    4 => {
                        emit_symbol!(($block) & 0xFF);
                        emit_symbol!(($block >> 8) & 0xFF);
                        emit_symbol!(($block >> 16) & 0xFF);
                        emit_symbol!(($block >> 24) & 0xFF);
                        out_ptr.write((($block >> 40) & 0xFF) as u8);
                        out_ptr = out_ptr.add(1);
                        in_ptr = in_ptr.add(6);
                    }
                    3 => {
                        emit_symbol!(($block) & 0xFF);
                        emit_symbol!(($block >> 8) & 0xFF);
                        emit_symbol!(($block >> 16) & 0xFF);
                        out_ptr.write((($block >> 32) & 0xFF) as u8);
                        out_ptr = out_ptr.add(1);
                        in_ptr = in_ptr.add(5);
                    }
                    2 => {
                        emit_symbol!(($block) & 0xFF);
                        emit_symbol!(($block >> 8) & 0xFF);
                        out_ptr.write((($block >> 24) & 0xFF) as u8);
                        out_ptr = out_ptr.add(1);
                        in_ptr = in_ptr.add(4);
                    }
                    1 => {
                        emit_symbol!(($block) & 0xFF);
                        out_ptr.write((($block >> 16) & 0xFF) as u8);
                        out_ptr = out_ptr.add(1);
                        in_ptr = in_ptr.add(3);
                    }
                    0 => {
                        out_ptr.write((($block >> 8) & 0xFF) as u8);
                        out_ptr = out_ptr.add(1);
                        in_ptr = in_ptr.add(2);
                    }
                    _ => core::hint::unreachable_unchecked(),
                }
            };
        }

        // 32-code fast path: process four 8-byte blocks when all are escape-free.
        if decoded.len() >= 256 && compressed.len() >= 32 {
            let block_out_end = out_end.sub(256);
            let block_in_end = in_end.sub(32);

            while out_ptr.cast_const() <= block_out_end && in_ptr < block_in_end {
                let b0 = in_ptr.cast::<u64>().read_unaligned();
                let b1 = in_ptr.add(8).cast::<u64>().read_unaligned();
                let b2 = in_ptr.add(16).cast::<u64>().read_unaligned();
                let b3 = in_ptr.add(24).cast::<u64>().read_unaligned();

                let esc = Self::escape_mask(b0)
                    | Self::escape_mask(b1)
                    | Self::escape_mask(b2)
                    | Self::escape_mask(b3);

                if esc == 0 {
                    emit_block!(b0);
                    emit_block!(b1);
                    emit_block!(b2);
                    emit_block!(b3);
                    in_ptr = in_ptr.add(32);
                    continue;
                }
                // Fall through to 8-code path for escape handling.
                break;
            }
        }

        // 8-code fast path with escape handling.
        if decoded.len() >= 64 && compressed.len() >= 8 {
            let block_out_end = out_end.sub(64);
            let block_in_end = in_end.sub(8);

            while out_ptr.cast_const() <= block_out_end && in_ptr < block_in_end {
                let block = in_ptr.cast::<u64>().read_unaligned();
                let escape_mask = Self::escape_mask(block);

                if escape_mask == 0 {
                    emit_block!(block);
                    in_ptr = in_ptr.add(8);
                } else {
                    let first_esc = (escape_mask.trailing_zeros() >> 3) as usize;
                    handle_escape_block!(block, first_esc);
                }
            }
        }

        // Scalar fallback for remaining bytes.
        while out_end.offset_from(out_ptr) > 8 && in_ptr < in_end {
            let code = in_ptr.read();
            in_ptr = in_ptr.add(1);

            if code == ESCAPE_CODE {
                out_ptr.write(in_ptr.read());
                in_ptr = in_ptr.add(1);
                out_ptr = out_ptr.add(1);
            } else {
                emit_symbol!(code);
            }
        }

        debug_assert_eq!(
            in_ptr, in_end,
            "decompression should exhaust input before output"
        );

        out_ptr.offset_from(out_begin) as usize
    }
}

#[cfg(test)]
mod tests {
    use fsst::CompressorBuilder;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn test_basic_decompress() -> VortexResult<()> {
        let mut builder = CompressorBuilder::new();
        builder.insert(Symbol::from_slice(b"hello\0\0\0"), 5);
        let compressor = builder.build();

        let compressed = compressor.compress(b"hello");
        let decompressor =
            OptimizedDecompressor::new(compressor.symbol_table(), compressor.symbol_lengths());

        let mut output = Vec::with_capacity(64);
        let len = decompressor.decompress_into(&compressed, output.spare_capacity_mut());
        unsafe { output.set_len(len) };

        assert_eq!(&output, b"hello");
        Ok(())
    }

    #[test]
    fn test_escape_codes() -> VortexResult<()> {
        let compressor = CompressorBuilder::default().build();
        let input = b"abc";
        let compressed = compressor.compress(input);

        let decompressor =
            OptimizedDecompressor::new(compressor.symbol_table(), compressor.symbol_lengths());

        let mut output = Vec::with_capacity(64);
        let len = decompressor.decompress_into(&compressed, output.spare_capacity_mut());
        unsafe { output.set_len(len) };

        assert_eq!(&output, b"abc");
        Ok(())
    }

    #[test]
    fn test_matches_baseline() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut owned: Vec<Vec<u8>> = Vec::new();

        for _ in 0..100 {
            let len = rng.random_range(5..50);
            let s: Vec<u8> = (0..len).map(|_| rng.random_range(b'a'..=b'z')).collect();
            owned.push(s);
        }
        let lines: Vec<&[u8]> = owned.iter().map(|s| s.as_slice()).collect();

        let compressor = fsst::Compressor::train(&lines);
        let baseline = compressor.decompressor();
        let optimized =
            OptimizedDecompressor::new(compressor.symbol_table(), compressor.symbol_lengths());

        for line in &lines {
            let compressed = compressor.compress(line);
            let baseline_result = baseline.decompress(&compressed);

            let mut opt_result =
                Vec::with_capacity(baseline.max_decompression_capacity(&compressed) + 7);
            let len = optimized.decompress_into(&compressed, opt_result.spare_capacity_mut());
            unsafe { opt_result.set_len(len) };

            assert_eq!(
                baseline_result, opt_result,
                "Mismatch for input: {:?}",
                line
            );
        }
        Ok(())
    }

    #[test]
    fn test_matches_baseline_with_escapes() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(99);
        let mut owned: Vec<Vec<u8>> = Vec::new();

        for _ in 0..100 {
            let len = rng.random_range(5..100);
            let s: Vec<u8> = (0..len).map(|_| rng.random_range(0..=255u8)).collect();
            owned.push(s);
        }
        let lines: Vec<&[u8]> = owned.iter().map(|s| s.as_slice()).collect();

        let compressor = fsst::Compressor::train(&lines);
        let baseline = compressor.decompressor();
        let optimized =
            OptimizedDecompressor::new(compressor.symbol_table(), compressor.symbol_lengths());

        for line in &lines {
            let compressed = compressor.compress(line);
            let baseline_result = baseline.decompress(&compressed);

            let mut opt_result =
                Vec::with_capacity(baseline.max_decompression_capacity(&compressed) + 7);
            let len = optimized.decompress_into(&compressed, opt_result.spare_capacity_mut());
            unsafe { opt_result.set_len(len) };

            assert_eq!(baseline_result, opt_result);
        }
        Ok(())
    }

    #[test]
    fn test_empty_input() -> VortexResult<()> {
        let compressor = CompressorBuilder::default().build();
        let decompressor =
            OptimizedDecompressor::new(compressor.symbol_table(), compressor.symbol_lengths());

        let mut output = Vec::with_capacity(64);
        let len = decompressor.decompress_into(&[], output.spare_capacity_mut());
        assert_eq!(len, 0);
        Ok(())
    }

    #[test]
    fn test_large_corpus() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut owned: Vec<Vec<u8>> = Vec::new();

        for _ in 0..1000 {
            let len = rng.random_range(1..500);
            let s: Vec<u8> = (0..len).map(|_| rng.random_range(b'a'..=b'z')).collect();
            owned.push(s);
        }
        let lines: Vec<&[u8]> = owned.iter().map(|s| s.as_slice()).collect();

        let compressor = fsst::Compressor::train(&lines);
        let baseline = compressor.decompressor();
        let optimized =
            OptimizedDecompressor::new(compressor.symbol_table(), compressor.symbol_lengths());

        let mut all_compressed = Vec::new();
        let mut all_expected = Vec::new();
        for line in &lines {
            let compressed = compressor.compress(line);
            all_compressed.extend_from_slice(&compressed);
            all_expected.extend_from_slice(line);
        }

        let baseline_result = baseline.decompress(&all_compressed);

        let mut opt_result =
            Vec::with_capacity(baseline.max_decompression_capacity(&all_compressed) + 7);
        let len = optimized.decompress_into(&all_compressed, opt_result.spare_capacity_mut());
        unsafe { opt_result.set_len(len) };

        assert_eq!(baseline_result, opt_result);
        assert_eq!(all_expected, opt_result);
        Ok(())
    }
}
