// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized FSST decompressor that replaces the default fsst-rs decompressor
//! with a version tuned for throughput.
//!
//! Key optimizations over the baseline fsst-rs implementation:
//! 1. Packed symbol+length table: symbol value and length in a single 16-byte struct,
//!    eliminating dual array lookups and improving cache locality.
//! 2. Simplified escape handling: uses a compact loop instead of an 8-arm match statement,
//!    reducing code size and improving instruction cache utilization.
//! 3. SWAR escape detection: same approach as fsst-rs but with tighter code generation.

use std::mem::MaybeUninit;

use fsst::ESCAPE_CODE;
use fsst::Symbol;

/// Packed entry combining symbol value and length for cache-friendly lookup.
///
/// By packing symbol and length together, we eliminate the dual array lookup
/// (one for symbols, one for lengths) that the baseline decompressor uses.
/// Each entry is 16 bytes to ensure natural alignment.
#[derive(Copy, Clone)]
#[repr(C, align(16))]
pub(crate) struct PackedSymbol {
    /// The symbol value (up to 8 bytes, little-endian packed into u64).
    value: u64,
    /// The number of valid bytes in `value` (1-8).
    len: u64,
}

/// Optimized FSST decompressor with a packed lookup table.
pub struct OptimizedDecompressor {
    /// Lookup table indexed by code (0-255). Index 255 is unused (escape).
    /// 256 entries x 16 bytes = 4KB, fits entirely in L1 cache.
    table: Box<[PackedSymbol; 256]>,
}

impl OptimizedDecompressor {
    /// Build from symbol table slices (same inputs as `fsst::Decompressor::new`).
    pub fn new(symbols: &[Symbol], lengths: &[u8]) -> Self {
        assert!(
            symbols.len() <= 255,
            "symbol table cannot exceed 255 entries"
        );
        assert_eq!(symbols.len(), lengths.len());

        let mut table = Box::new([PackedSymbol { value: 0, len: 1 }; 256]);
        for (i, (sym, &len)) in symbols.iter().zip(lengths.iter()).enumerate() {
            table[i] = PackedSymbol {
                value: sym.to_u64(),
                len: len as u64,
            };
        }
        Self { table }
    }

    /// Decompress `compressed` codes into `decoded` buffer.
    ///
    /// Returns the number of bytes written to `decoded`.
    ///
    /// The `decoded` buffer must have at least `compressed.len() / 2` capacity (lower bound).
    /// For best results, provide `8 * compressed.len() + 7` capacity (upper bound).
    ///
    /// # Panics
    ///
    /// Panics if `decoded` is too small.
    pub fn decompress_into(&self, compressed: &[u8], decoded: &mut [MaybeUninit<u8>]) -> usize {
        assert!(
            decoded.len() >= compressed.len() / 2,
            "decoded buffer too small"
        );

        // SAFETY: We carefully manage pointer bounds within the inner function.
        unsafe { self.decompress_inner(compressed, decoded) }
    }

    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn, clippy::cast_possible_truncation)]
    unsafe fn decompress_inner(&self, compressed: &[u8], decoded: &mut [MaybeUninit<u8>]) -> usize {
        let mut in_ptr = compressed.as_ptr();
        let in_end = in_ptr.add(compressed.len());

        let mut out_ptr: *mut u8 = decoded.as_mut_ptr().cast();
        let out_begin = out_ptr.cast_const();
        let out_end = decoded.as_ptr().add(decoded.len()).cast::<u8>();

        let table = self.table.as_ptr();

        // Fast path: process 8 codes at a time.
        // Need 64 bytes output headroom (8 symbols x 8 bytes max each).
        if decoded.len() >= 64 && compressed.len() >= 8 {
            let block_out_end = out_end.sub(64) as *mut u8;
            let block_in_end = in_end.sub(8);

            while out_ptr <= block_out_end && in_ptr < block_in_end {
                // Read 8 codes as a u64 (little-endian).
                let next_block = in_ptr.cast::<u64>().read_unaligned();

                // Detect escape codes (byte == 0xFF) using SWAR.
                // For byte b: b == 0xFF iff high bit set AND low 7 bits all set.
                let escape_mask = (next_block & 0x8080_8080_8080_8080)
                    & (((!next_block & 0x7F7F_7F7F_7F7F_7F7F).wrapping_add(0x7F7F_7F7F_7F7F_7F7F))
                        ^ 0x8080_8080_8080_8080);

                if escape_mask == 0 {
                    // No escapes: process all 8 codes in straight-line sequence.
                    // Each write: store u64 at out_ptr, advance by symbol length.
                    // Using a local variable for out_ptr to help the compiler
                    // avoid re-reading from memory.
                    let mut p = out_ptr;
                    let c0 = (next_block & 0xFF) as usize;
                    let e0 = &*table.add(c0);
                    p.cast::<u64>().write_unaligned(e0.value);
                    p = p.add(e0.len as usize);

                    let c1 = ((next_block >> 8) & 0xFF) as usize;
                    let e1 = &*table.add(c1);
                    p.cast::<u64>().write_unaligned(e1.value);
                    p = p.add(e1.len as usize);

                    let c2 = ((next_block >> 16) & 0xFF) as usize;
                    let e2 = &*table.add(c2);
                    p.cast::<u64>().write_unaligned(e2.value);
                    p = p.add(e2.len as usize);

                    let c3 = ((next_block >> 24) & 0xFF) as usize;
                    let e3 = &*table.add(c3);
                    p.cast::<u64>().write_unaligned(e3.value);
                    p = p.add(e3.len as usize);

                    let c4 = ((next_block >> 32) & 0xFF) as usize;
                    let e4 = &*table.add(c4);
                    p.cast::<u64>().write_unaligned(e4.value);
                    p = p.add(e4.len as usize);

                    let c5 = ((next_block >> 40) & 0xFF) as usize;
                    let e5 = &*table.add(c5);
                    p.cast::<u64>().write_unaligned(e5.value);
                    p = p.add(e5.len as usize);

                    let c6 = ((next_block >> 48) & 0xFF) as usize;
                    let e6 = &*table.add(c6);
                    p.cast::<u64>().write_unaligned(e6.value);
                    p = p.add(e6.len as usize);

                    let c7 = ((next_block >> 56) & 0xFF) as usize;
                    let e7 = &*table.add(c7);
                    p.cast::<u64>().write_unaligned(e7.value);
                    p = p.add(e7.len as usize);

                    out_ptr = p;
                    in_ptr = in_ptr.add(8);
                } else {
                    // Escape found: process codes before the first escape,
                    // then handle the escape pair.
                    let first_esc = (escape_mask.trailing_zeros() >> 3) as usize;

                    let mut p = out_ptr;
                    let mut shift = 0u32;
                    for _ in 0..first_esc {
                        let code = ((next_block >> shift) & 0xFF) as usize;
                        let entry = &*table.add(code);
                        p.cast::<u64>().write_unaligned(entry.value);
                        p = p.add(entry.len as usize);
                        shift += 8;
                    }

                    // Write the escaped literal byte.
                    let escaped = ((next_block >> (shift + 8)) & 0xFF) as u8;
                    p.write(escaped);
                    p = p.add(1);

                    out_ptr = p;
                    in_ptr = in_ptr.add(first_esc + 2);
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
                let entry = &*table.add(code as usize);
                out_ptr.cast::<u64>().write_unaligned(entry.value);
                out_ptr = out_ptr.add(entry.len as usize);
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

        // Generate a mix of short and long strings
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

        // Compress all lines into one big buffer (simulating bulk decompression)
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
