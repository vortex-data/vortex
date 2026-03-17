// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized FSST decompressor that replaces the default fsst-rs decompressor
//! with a version tuned for throughput.
//!
//! Key optimizations over the baseline fsst-rs implementation:
//! 1. Symbols stored as `u64` directly, avoiding `Symbol::to_u64()` conversion per lookup.
//! 2. Multi-level block processing: 32-code and 8-code fast paths that process
//!    compressed data in large chunks when no escape codes are present.
//! 3. Unified loop that re-enters the 32-code fast path after handling each escape,
//!    instead of permanently dropping to the slower 8-code path.
//! 4. Fully unrolled escape handling via match statement for optimal branch prediction.
//! 5. Runtime CPU feature detection for BMI1/BMI2/POPCNT-optimized codegen on x86-64.

use std::mem::MaybeUninit;

use fsst::ESCAPE_CODE;
use fsst::Symbol;

/// Hint that the calling branch is cold (unlikely). Placing a `#[cold]`
/// `#[inline(never)]` call at the top of a branch causes LLVM to treat
/// the entire branch as unlikely, improving code layout for the hot path.
#[cold]
#[inline(never)]
fn cold() {}

/// Optimized FSST decompressor using separate symbol/length tables.
///
/// The symbol table stores pre-converted `u64` values to avoid per-lookup
/// conversion overhead. Separate arrays keep the cache footprint small:
/// symbols (2KB) + lengths (256B) ≈ 2.3KB, fitting entirely in L1 cache.
pub struct OptimizedDecompressor {
    symbols: Box<[u64; 256]>,
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

        // Use target-feature-optimized path on x86-64 for better tzcnt codegen.
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("bmi1") {
                return unsafe { self.decompress_inner_bmi(compressed, decoded) };
            }
        }
        unsafe { self.decompress_inner(compressed, decoded) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "bmi1,bmi2,popcnt")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn decompress_inner_bmi(
        &self,
        compressed: &[u8],
        decoded: &mut [MaybeUninit<u8>],
    ) -> usize {
        self.decompress_inner(compressed, decoded)
    }

    /// SWAR escape detection: returns a mask with the high bit set in each byte
    /// that equals 0xFF.
    #[inline(always)]
    const fn escape_mask(block: u64) -> u64 {
        let hi = block & 0x8080_8080_8080_8080;
        let lo_inv = !block & 0x7F7F_7F7F_7F7F_7F7F;
        hi & (lo_inv.wrapping_add(0x7F7F_7F7F_7F7F_7F7F) ^ 0x8080_8080_8080_8080)
    }

    /// Safe end-pointer for block processing. Returns null when the buffer is
    /// too small, which makes `ptr <= null` immediately false.
    #[inline(always)]
    fn block_end(end: *const u8, margin: usize, len: usize) -> *const u8 {
        if len >= margin {
            unsafe { end.sub(margin) }
        } else {
            core::ptr::null()
        }
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

        // Emit one symbol: write 8 bytes (may overshoot), advance by actual length.
        macro_rules! emit_symbol {
            ($code:expr) => {{
                let c = $code as usize;
                out_ptr.cast::<u64>().write_unaligned(*symbols.add(c));
                out_ptr = out_ptr.add(*lengths.add(c) as usize);
            }};
        }

        // Emit all 8 symbols from a u64 block (no escapes).
        macro_rules! emit_block {
            ($block:expr) => {{
                let b = $block;
                emit_symbol!((b) & 0xFF);
                emit_symbol!((b >> 8) & 0xFF);
                emit_symbol!((b >> 16) & 0xFF);
                emit_symbol!((b >> 24) & 0xFF);
                emit_symbol!((b >> 32) & 0xFF);
                emit_symbol!((b >> 40) & 0xFF);
                emit_symbol!((b >> 48) & 0xFF);
                emit_symbol!((b >> 56) & 0xFF);
            }};
        }

        // Emit symbols before the first escape, write the escaped literal,
        // and advance `in_ptr`. The loop body is small enough for LLVM to
        // unroll when `pos` is a known constant from `trailing_zeros`.
        macro_rules! emit_before_escape {
            ($b:expr, $esc_pos:expr) => {{
                let b = $b;
                let pos = $esc_pos;
                // Emit each non-escape symbol before the escape byte.
                let mut i = 0usize;
                while i < pos {
                    emit_symbol!((b >> (i as u32 * 8)) & 0xFF);
                    i += 1;
                }
                if pos < 7 {
                    // Literal byte follows the escape within this block.
                    let literal_shift = (pos as u32 + 1) * 8;
                    out_ptr.write(((b >> literal_shift) & 0xFF) as u8);
                    out_ptr = out_ptr.add(1);
                    in_ptr = in_ptr.add(pos + 2);
                } else {
                    // Escape is at byte 7 — literal is in the next block.
                    // Just consume the 7 symbols; the outer loop will
                    // re-read starting at the escape byte.
                    in_ptr = in_ptr.add(7);
                }
            }};
        }

        let out_end32 = Self::block_end(out_end, 256, decoded.len());
        let in_end32 = Self::block_end(in_end, 32, compressed.len());
        let out_end8 = Self::block_end(out_end, 64, decoded.len());
        let in_end8 = Self::block_end(in_end, 8, compressed.len());

        // Main loop: 32-code escape-free fast path, falling back to single
        // 8-code blocks for escapes, then immediately re-entering the fast path.
        'outer: while out_ptr.cast_const() <= out_end8 && in_ptr < in_end8 {
            // 32-code escape-free inner loop.
            while out_ptr.cast_const() <= out_end32 && in_ptr < in_end32 {
                let b0 = in_ptr.cast::<u64>().read_unaligned();
                let b1 = in_ptr.add(8).cast::<u64>().read_unaligned();
                let b2 = in_ptr.add(16).cast::<u64>().read_unaligned();
                let b3 = in_ptr.add(24).cast::<u64>().read_unaligned();

                let m0 = Self::escape_mask(b0);
                let m1 = Self::escape_mask(b1);
                let m2 = Self::escape_mask(b2);
                let m3 = Self::escape_mask(b3);

                if (m0 | m1 | m2 | m3) != 0 {
                    cold();
                    // Process escape-free blocks before the first escape,
                    // then handle the escape and break to re-check bounds.
                    if m0 != 0 {
                        let first_esc = (m0.trailing_zeros() >> 3) as usize;
                        emit_before_escape!(b0, first_esc);
                        break;
                    }
                    emit_block!(b0);
                    in_ptr = in_ptr.add(8);

                    if m1 != 0 {
                        let first_esc = (m1.trailing_zeros() >> 3) as usize;
                        emit_before_escape!(b1, first_esc);
                        break;
                    }
                    emit_block!(b1);
                    in_ptr = in_ptr.add(8);

                    if m2 != 0 {
                        let first_esc = (m2.trailing_zeros() >> 3) as usize;
                        emit_before_escape!(b2, first_esc);
                        break;
                    }
                    emit_block!(b2);
                    in_ptr = in_ptr.add(8);

                    let first_esc = (m3.trailing_zeros() >> 3) as usize;
                    emit_before_escape!(b3, first_esc);
                    break;
                }

                emit_block!(b0);
                emit_block!(b1);
                emit_block!(b2);
                emit_block!(b3);
                in_ptr = in_ptr.add(32);
            }

            // Single 8-code block with escape handling, then re-enter fast path.
            if out_ptr.cast_const() > out_end8 || in_ptr >= in_end8 {
                break 'outer;
            }
            let block = in_ptr.cast::<u64>().read_unaligned();
            let esc = Self::escape_mask(block);

            if esc == 0 {
                emit_block!(block);
                in_ptr = in_ptr.add(8);
            } else {
                cold();
                let first_esc = (esc.trailing_zeros() >> 3) as usize;
                emit_before_escape!(block, first_esc);
            }
        }

        // Scalar tail.
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

    #[test]
    fn test_large_corpus_with_escapes() -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut owned: Vec<Vec<u8>> = Vec::new();

        for _ in 0..1000 {
            let len = rng.random_range(1..500);
            let s: Vec<u8> = (0..len).map(|_| rng.random_range(0..=255u8)).collect();
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
