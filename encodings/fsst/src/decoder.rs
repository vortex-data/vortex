// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Portions of this file are derived from `fsst-rs` 0.5.10 (Apache-2.0,
// SpiralDB Developers). The control flow of `decompress_into` and the escape
// handling are mirrored from upstream so the output is byte-for-byte equivalent.

//! Local FSST decoder.
//!
//! This duplicates the logic of [`fsst::Decompressor::decompress_into`] so we
//! can tune the hot path independently of the upstream crate. Today the body
//! mirrors upstream byte-for-byte; the structure exists so future tuning
//! (e.g. AVX2 gathers, AVX-512 scatter, partitioning by symbol length) can
//! land here without forking `fsst-rs`.
//!
//! ## Microbenchmarks
//!
//! See `benches/fsst_decode.rs` for a head-to-head against
//! [`fsst::Decompressor::decompress_into`]. A previous attempt to break the
//! `out_ptr += len[code]` dependency chain by hoisting all 8 symbol/length
//! lookups and computing a prefix sum was **slower** by ~1.5x on Skylake-X,
//! likely because holding 8 symbols + 8 lengths + 8 offsets in flight forces
//! register spills. The current implementation accepts the serial
//! `out_ptr` chain that upstream uses.

use std::mem::MaybeUninit;
use std::mem::size_of;

use fsst::ESCAPE_CODE;
use fsst::Symbol;

/// Decoder over a borrowed FSST symbol table.
///
/// `symbols[i]` is the up-to-8 byte expansion of code `i`, packed little-endian
/// into a `u64`. `lengths[i]` is the number of valid bytes (1..=8). Code 255
/// (`ESCAPE_CODE`) is reserved: it indicates that the next byte in the input
/// stream is an unencoded literal.
pub struct Decoder<'a> {
    symbols: &'a [Symbol],
    lengths: &'a [u8],
}

impl<'a> Decoder<'a> {
    /// Build a decoder from a borrowed symbol table.
    ///
    /// `symbols` and `lengths` must have the same length, and that length must
    /// be at most 255 (matching upstream's `FSST_CODE_BASE` constraint).
    pub fn new(symbols: &'a [Symbol], lengths: &'a [u8]) -> Self {
        assert_eq!(
            symbols.len(),
            lengths.len(),
            "FSST symbols and lengths must have the same length"
        );
        assert!(
            symbols.len() < 256,
            "FSST symbol table cannot have size exceeding 255"
        );
        Self { symbols, lengths }
    }

    /// Upper bound on the size of the decompressed output for `compressed`.
    pub fn max_decompression_capacity(&self, compressed: &[u8]) -> usize {
        size_of::<Symbol>() * (compressed.len() + 1)
    }

    /// Decompress `compressed` into `decoded`, returning the number of bytes
    /// written.
    ///
    /// `decoded` must be at least `compressed.len() / 2` bytes (the lower bound
    /// reachable when every code is an escape) and ideally at least
    /// [`max_decompression_capacity`][Self::max_decompression_capacity].
    pub fn decompress_into(&self, compressed: &[u8], decoded: &mut [MaybeUninit<u8>]) -> usize {
        assert!(
            decoded.len() >= compressed.len() / 2,
            "decoded buffer is smaller than the lower-bound decompressed size"
        );

        // SAFETY: all the unsafe blocks below operate on raw pointers derived
        // from `compressed` and `decoded`. The outer loop checks ensure we never
        // read past `in_end` or write past `out_end`.
        unsafe {
            let mut in_ptr = compressed.as_ptr();
            let in_end = in_ptr.add(compressed.len());

            let mut out_ptr: *mut u8 = decoded.as_mut_ptr().cast();
            let out_begin = out_ptr.cast_const();
            let out_end = decoded.as_ptr().add(decoded.len()).cast::<u8>();

            // Hot path: 8 codes at a time, with a 64-byte tail-write reservation
            // on the output and an 8-byte tail-read reservation on the input.
            if decoded.len() >= 8 * size_of::<Symbol>() && compressed.len() >= 8 {
                let block_out_end = out_end.sub(8 * size_of::<Symbol>());
                let block_in_end = in_end.sub(8);

                while out_ptr.cast_const() <= block_out_end && in_ptr < block_in_end {
                    let next_block = in_ptr.cast::<u64>().read_unaligned();

                    // Bit trick from upstream: for each byte, set the high bit
                    // iff the byte is exactly 0xFF (the escape code).
                    let escape_mask = (next_block & 0x8080_8080_8080_8080)
                        & ((((!next_block) & 0x7F7F_7F7F_7F7F_7F7F) + 0x7F7F_7F7F_7F7F_7F7F)
                            ^ 0x8080_8080_8080_8080);

                    if escape_mask == 0 {
                        // No escapes: decode all 8 codes.
                        out_ptr = self.decode_block_8(next_block, out_ptr);
                        in_ptr = in_ptr.add(8);
                    } else if (next_block & 0x00FF_00FF_00FF_00FF) == 0x00FF_00FF_00FF_00FF {
                        // All 4 even-positioned bytes are ESCAPE_CODE: emit the
                        // 4 odd-positioned literals directly.
                        out_ptr.write(((next_block >> 8) & 0xFF) as u8);
                        out_ptr.add(1).write(((next_block >> 24) & 0xFF) as u8);
                        out_ptr.add(2).write(((next_block >> 40) & 0xFF) as u8);
                        out_ptr.add(3).write(((next_block >> 56) & 0xFF) as u8);
                        out_ptr = out_ptr.add(4);
                        in_ptr = in_ptr.add(8);
                    } else {
                        // Mixed: decode codes up to the first escape, then emit
                        // the literal that follows it.
                        let first_escape_pos = (escape_mask.trailing_zeros() >> 3) as usize;
                        debug_assert!(first_escape_pos < 8);
                        let (advance_in, advance_out) =
                            self.decode_until_escape(next_block, first_escape_pos, out_ptr);
                        in_ptr = in_ptr.add(advance_in);
                        out_ptr = out_ptr.add(advance_out);
                    }
                }
            }

            // Slower mid-path: still safe to use 8-byte stores, one code at a
            // time.
            while out_end.offset_from(out_ptr) >= size_of::<Symbol>() as isize && in_ptr < in_end {
                let code = in_ptr.read();
                in_ptr = in_ptr.add(1);

                if code == ESCAPE_CODE {
                    assert!(
                        in_ptr < in_end,
                        "truncated compressed string: escape code at end of input"
                    );
                    out_ptr.write(in_ptr.read());
                    in_ptr = in_ptr.add(1);
                    out_ptr = out_ptr.add(1);
                } else {
                    out_ptr
                        .cast::<u64>()
                        .write_unaligned(self.symbols.get_unchecked(code as usize).to_u64());
                    out_ptr = out_ptr.add(*self.lengths.get_unchecked(code as usize) as usize);
                }
            }

            // Final tail: too little output room to do an 8-byte unaligned
            // write; copy exactly `len` bytes.
            while in_ptr < in_end {
                let code = in_ptr.read();
                in_ptr = in_ptr.add(1);

                if code == ESCAPE_CODE {
                    assert!(
                        in_ptr < in_end,
                        "truncated compressed string: escape code at end of input"
                    );
                    assert!(
                        out_ptr.cast_const() < out_end,
                        "output buffer sized too small"
                    );
                    out_ptr.write(in_ptr.read());
                    in_ptr = in_ptr.add(1);
                    out_ptr = out_ptr.add(1);
                } else {
                    let len = *self.lengths.get_unchecked(code as usize) as usize;
                    assert!(
                        out_end.offset_from(out_ptr) >= len as isize,
                        "output buffer sized too small"
                    );
                    let sym = self.symbols.get_unchecked(code as usize).to_u64();
                    let sym_bytes = sym.to_le_bytes();
                    std::ptr::copy_nonoverlapping(sym_bytes.as_ptr(), out_ptr, len);
                    out_ptr = out_ptr.add(len);
                }
            }

            assert_eq!(
                in_ptr, in_end,
                "decompression should exhaust input before output"
            );

            out_ptr.offset_from(out_begin) as usize
        }
    }

    /// Decode a block of 8 non-escape codes packed into `block` (little-endian).
    ///
    /// Returns the new `out_ptr` after the 8 stores. The caller must guarantee
    /// at least 64 bytes of writable room at `out_ptr`, so each lane can do a
    /// full `u64` unaligned store regardless of its symbol length; subsequent
    /// stores overwrite the trailing zero padding from earlier ones.
    ///
    /// The body mirrors the upstream `fsst-rs` no-escape branch verbatim:
    /// `store; out_ptr += len; store; out_ptr += len; ...`. We tried hoisting
    /// all symbol/length lookups and computing a prefix sum so the 8 stores
    /// could issue independently, but on Skylake-X that costs ~1.5x more
    /// (microbench `benches/fsst_decode.rs`). The likely cause is register
    /// pressure: holding 8 symbols + 8 lengths + 8 offsets simultaneously
    /// forces spills, while the upstream pattern keeps a single
    /// `(out_ptr, code, len)` triple live at any moment.
    ///
    /// # Safety
    ///
    /// `out_ptr` must point to at least 64 bytes of writable memory.
    #[inline(always)]
    unsafe fn decode_block_8(&self, block: u64, mut out_ptr: *mut u8) -> *mut u8 {
        let symbols = self.symbols.as_ptr();
        let lengths = self.lengths.as_ptr();

        // SAFETY: each byte of `block` is a non-escape code and is in range
        // because the caller verified `escape_mask == 0`; the encoder only
        // emits in-range codes. Each store is an unaligned `u64` write into
        // the 64-byte reservation made by the outer loop.
        unsafe {
            let c = (block & 0xFF) as usize;
            out_ptr
                .cast::<u64>()
                .write_unaligned((*symbols.add(c)).to_u64());
            out_ptr = out_ptr.add(*lengths.add(c) as usize);

            let c = ((block >> 8) & 0xFF) as usize;
            out_ptr
                .cast::<u64>()
                .write_unaligned((*symbols.add(c)).to_u64());
            out_ptr = out_ptr.add(*lengths.add(c) as usize);

            let c = ((block >> 16) & 0xFF) as usize;
            out_ptr
                .cast::<u64>()
                .write_unaligned((*symbols.add(c)).to_u64());
            out_ptr = out_ptr.add(*lengths.add(c) as usize);

            let c = ((block >> 24) & 0xFF) as usize;
            out_ptr
                .cast::<u64>()
                .write_unaligned((*symbols.add(c)).to_u64());
            out_ptr = out_ptr.add(*lengths.add(c) as usize);

            let c = ((block >> 32) & 0xFF) as usize;
            out_ptr
                .cast::<u64>()
                .write_unaligned((*symbols.add(c)).to_u64());
            out_ptr = out_ptr.add(*lengths.add(c) as usize);

            let c = ((block >> 40) & 0xFF) as usize;
            out_ptr
                .cast::<u64>()
                .write_unaligned((*symbols.add(c)).to_u64());
            out_ptr = out_ptr.add(*lengths.add(c) as usize);

            let c = ((block >> 48) & 0xFF) as usize;
            out_ptr
                .cast::<u64>()
                .write_unaligned((*symbols.add(c)).to_u64());
            out_ptr = out_ptr.add(*lengths.add(c) as usize);

            let c = ((block >> 56) & 0xFF) as usize;
            out_ptr
                .cast::<u64>()
                .write_unaligned((*symbols.add(c)).to_u64());
            out_ptr = out_ptr.add(*lengths.add(c) as usize);

            out_ptr
        }
    }

    /// Mixed-block path. Decodes the codes in `block` up to (but not
    /// including) the byte at position `escape_pos`, then emits the escaped
    /// literal that follows it. Returns `(advance_in, advance_out)` to
    /// advance the input and output pointers respectively.
    ///
    /// # Safety
    ///
    /// `out_ptr` must point to at least 64 bytes of writable memory; the
    /// caller's reservation in the outer loop guarantees this.
    #[inline(always)]
    unsafe fn decode_until_escape(
        &self,
        block: u64,
        escape_pos: usize,
        out_ptr: *mut u8,
    ) -> (usize, usize) {
        let bytes = block.to_le_bytes();
        let mut local_out = out_ptr;

        // SAFETY: byte indices are bounded by `escape_pos < 8` and the symbol
        // table is sized to cover any valid non-escape code; the per-lane
        // u64 stores fit because the caller reserved 64 bytes.
        unsafe {
            for i in 0..escape_pos {
                let code = bytes[i];
                local_out
                    .cast::<u64>()
                    .write_unaligned(self.symbols.get_unchecked(code as usize).to_u64());
                local_out = local_out.add(*self.lengths.get_unchecked(code as usize) as usize);
            }

            // The byte at `escape_pos` is ESCAPE_CODE; the byte at
            // `escape_pos + 1` is the literal. Emit it.
            //
            // If the escape sits at position 7 of an 8-byte block we have not
            // yet seen the literal byte (it's the first byte of the *next*
            // input block), so we only consume 7 input bytes in that case.
            if escape_pos == 7 {
                let advance_out = local_out.offset_from(out_ptr) as usize;
                return (7, advance_out);
            }

            let literal = bytes[escape_pos + 1];
            local_out.write(literal);
            local_out = local_out.add(1);
            let advance_in = escape_pos + 2;
            let advance_out = local_out.offset_from(out_ptr) as usize;
            (advance_in, advance_out)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use fsst::Compressor;
    use fsst::CompressorBuilder;
    use fsst::Symbol;
    use rstest::rstest;

    use super::Decoder;

    fn build_compressor(samples: &[&[u8]]) -> Compressor {
        let mut builder = CompressorBuilder::new();
        for s in samples {
            builder.insert(Symbol::from_slice(&pad8(s)), s.len().min(8));
        }
        builder.build()
    }

    fn pad8(s: &[u8]) -> [u8; 8] {
        let mut out = [0u8; 8];
        let n = s.len().min(8);
        out[..n].copy_from_slice(&s[..n]);
        out
    }

    fn train(samples: &[&[u8]]) -> Compressor {
        let owned: Vec<&[u8]> = samples.to_vec();
        Compressor::train(&owned)
    }

    fn roundtrip_via_compressor(plain: &[u8], compressor: &Compressor) {
        let compressed = compressor.compress(plain);
        let decomp = compressor.decompressor();

        // Reference output via upstream decompressor.
        let mut upstream_buf = Vec::with_capacity(decomp.max_decompression_capacity(&compressed));
        let n_upstream = decomp.decompress_into(&compressed, upstream_buf.spare_capacity_mut());
        // SAFETY: upstream initialized n_upstream bytes.
        unsafe { upstream_buf.set_len(n_upstream) };
        assert_eq!(upstream_buf, plain, "upstream sanity check failed");

        // Local decoder.
        let local = Decoder::new(compressor.symbol_table(), compressor.symbol_lengths());
        let mut local_buf: Vec<MaybeUninit<u8>> =
            Vec::with_capacity(local.max_decompression_capacity(&compressed));
        local_buf.resize(local_buf.capacity(), MaybeUninit::uninit());
        let n_local = local.decompress_into(&compressed, &mut local_buf[..]);
        assert_eq!(n_local, n_upstream);

        // SAFETY: local decoder initialized n_local bytes.
        let local_bytes: Vec<u8> = local_buf[..n_local]
            .iter()
            .map(|b| unsafe { b.assume_init() })
            .collect();
        assert_eq!(local_bytes, plain);
    }

    #[rstest]
    #[case::empty(&b""[..])]
    #[case::short(&b"hello"[..])]
    #[case::repeating(b"abcdefghabcdefgh".as_slice())]
    #[case::all_ascii(b"the quick brown fox jumps over the lazy dog".as_slice())]
    fn roundtrip_basic(#[case] input: &[u8]) {
        let samples: &[&[u8]] = &[input];
        let compressor = train(samples);
        roundtrip_via_compressor(input, &compressor);
    }

    #[test]
    fn roundtrip_with_escapes() {
        // Force escapes by training on a tiny corpus then compressing very
        // different bytes.
        let compressor = build_compressor(&[b"aa"]);
        let plain = b"bxbxbxbxbxbxbxbxbxbxbxbx";
        roundtrip_via_compressor(plain, &compressor);
    }

    #[test]
    fn roundtrip_long_random() {
        // Pseudo-random repeatable input.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut buf = Vec::with_capacity(64 * 1024);
        for _ in 0..buf.capacity() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            #[expect(clippy::cast_possible_truncation)]
            let byte = (state >> 32) as u8;
            buf.push(byte);
        }
        let compressor = train(&[&buf]);
        roundtrip_via_compressor(&buf, &compressor);
    }

    #[test]
    fn roundtrip_realistic_text() {
        let plain = b"the quick brown fox jumps over the lazy dog. \
                      the quick brown fox jumps over the lazy dog. \
                      the quick brown fox jumps over the lazy dog. \
                      the quick brown fox jumps over the lazy dog.";
        let compressor = train(&[plain]);
        roundtrip_via_compressor(plain, &compressor);
    }
}
