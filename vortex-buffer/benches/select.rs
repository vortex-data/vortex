// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks comparing select implementations for aligned vs unaligned BitBuffers.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::UnalignedBitChunk;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[100_000];
const PERCENTILES: &[usize] = &[2, 10, 25, 50, 75, 90, 98];

// =============================================================================
// Select-in-word helper
// =============================================================================

static SELECT_IN_BYTE: [[u8; 8]; 256] = {
    let mut table = [[8u8; 8]; 256];
    let mut byte = 0usize;
    while byte < 256 {
        let mut bit_pos = 0usize;
        let mut rank = 0usize;
        while bit_pos < 8 {
            if (byte >> bit_pos) & 1 == 1 {
                table[byte][rank] = bit_pos as u8;
                rank += 1;
            }
            bit_pos += 1;
        }
        byte += 1;
    }
    table
};

#[inline]
fn select_in_word(word: u64, mut n: usize) -> usize {
    if n <= 2 {
        let mut word = word;
        loop {
            let tz = word.trailing_zeros() as usize;
            if n == 0 {
                return tz;
            }
            word &= word - 1;
            n -= 1;
        }
    } else {
        let mut word = word;
        let mut pos = 0usize;

        let lower = (word as u32).count_ones() as usize;
        if n >= lower {
            n -= lower;
            word >>= 32;
            pos += 32;
        }

        let lower = (word as u16).count_ones() as usize;
        if n >= lower {
            n -= lower;
            word >>= 16;
            pos += 16;
        }

        let lower = (word as u8).count_ones() as usize;
        if n >= lower {
            n -= lower;
            word >>= 8;
            pos += 8;
        }

        pos + SELECT_IN_BYTE[(word as u8) as usize][n] as usize
    }
}

/// Select from end of word (for reverse iteration)
#[inline]
fn select_in_word_reverse(word: u64, n: usize) -> usize {
    // Find the (popcount - 1 - n)th bit from the start = nth bit from end
    let popcount = word.count_ones() as usize;
    63 - select_in_word(word.reverse_bits(), popcount - 1 - n)
}

// =============================================================================
// Select implementations
// =============================================================================

/// Baseline: simple BitChunkIterator
#[inline]
fn select_simple(buf: &BitBuffer, n: usize) -> usize {
    let mut remaining = n;
    let chunks = buf.chunks();

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let popcount = chunk.count_ones() as usize;
        if remaining < popcount {
            return chunk_idx * 64 + select_in_word(chunk, remaining);
        }
        remaining -= popcount;
    }

    let rem = chunks.remainder_bits();
    if rem != 0 && remaining < rem.count_ones() as usize {
        return (buf.len() / 64) * 64 + select_in_word(rem, remaining);
    }
    panic!("out of bounds");
}

/// Unrolled: process 2 chunks at a time, reducing loop overhead
#[inline]
fn select_unrolled2(buf: &BitBuffer, n: usize) -> usize {
    let mut remaining = n;
    let chunks = buf.chunks();
    let mut iter = chunks.iter().enumerate();

    // Process pairs of chunks
    loop {
        let Some((idx0, chunk0)) = iter.next() else {
            break;
        };

        let pop0 = chunk0.count_ones() as usize;

        if let Some((idx1, chunk1)) = iter.next() {
            let pop1 = chunk1.count_ones() as usize;
            let combined_pop = pop0 + pop1;

            if remaining < combined_pop {
                // Target is in one of these two chunks
                if remaining < pop0 {
                    return idx0 * 64 + select_in_word(chunk0, remaining);
                } else {
                    return idx1 * 64 + select_in_word(chunk1, remaining - pop0);
                }
            }
            remaining -= combined_pop;
        } else {
            // Odd chunk at end
            if remaining < pop0 {
                return idx0 * 64 + select_in_word(chunk0, remaining);
            }
            remaining -= pop0;
            break;
        }
    }

    let rem = chunks.remainder_bits();
    if rem != 0 && remaining < rem.count_ones() as usize {
        return (buf.len() / 64) * 64 + select_in_word(rem, remaining);
    }
    panic!("out of bounds");
}

/// Block-based: compute 4 popcounts independently, single dependency per block
/// This allows SIMD popcount without loop-carried dependency per chunk
#[inline]
fn select_block4(buf: &BitBuffer, n: usize) -> usize {
    let mut remaining = n;
    let bytes = buf.inner().as_slice();
    let len = buf.len();
    let offset = buf.offset();

    // Only handle aligned case for simplicity
    if offset != 0 {
        return select_simple(buf, n);
    }

    let ptr = bytes.as_ptr() as *const u64;
    let num_chunks = len / 64;
    let mut i = 0;

    // Process 4 chunks at a time - popcounts computed independently
    while i + 4 <= num_chunks {
        // Load 4 chunks (could be SIMD loaded)
        let c0 = unsafe { ptr.add(i).read_unaligned() };
        let c1 = unsafe { ptr.add(i + 1).read_unaligned() };
        let c2 = unsafe { ptr.add(i + 2).read_unaligned() };
        let c3 = unsafe { ptr.add(i + 3).read_unaligned() };

        // Compute 4 popcounts independently (no dependency between them!)
        let pop0 = c0.count_ones() as usize;
        let pop1 = c1.count_ones() as usize;
        let pop2 = c2.count_ones() as usize;
        let pop3 = c3.count_ones() as usize;

        // Single dependency point: sum the block
        let block_pop = pop0 + pop1 + pop2 + pop3;

        if remaining < block_pop {
            // Narrow down within block (still need sequential check here)
            if remaining < pop0 {
                return i * 64 + select_in_word(c0, remaining);
            }
            remaining -= pop0;
            if remaining < pop1 {
                return (i + 1) * 64 + select_in_word(c1, remaining);
            }
            remaining -= pop1;
            if remaining < pop2 {
                return (i + 2) * 64 + select_in_word(c2, remaining);
            }
            remaining -= pop2;
            return (i + 3) * 64 + select_in_word(c3, remaining);
        }
        remaining -= block_pop;
        i += 4;
    }

    // Handle remaining chunks
    while i < num_chunks {
        let chunk = unsafe { ptr.add(i).read_unaligned() };
        let pop = chunk.count_ones() as usize;
        if remaining < pop {
            return i * 64 + select_in_word(chunk, remaining);
        }
        remaining -= pop;
        i += 1;
    }

    // Handle remainder bits
    let rem_bits = len % 64;
    if rem_bits > 0 {
        let start = num_chunks * 8;
        let mut buf8 = [0u8; 8];
        let avail = (bytes.len() - start).min(8);
        buf8[..avail].copy_from_slice(&bytes[start..start + avail]);
        let rem = u64::from_le_bytes(buf8) & ((1u64 << rem_bits) - 1);

        if remaining < rem.count_ones() as usize {
            return num_chunks * 64 + select_in_word(rem, remaining);
        }
    }
    panic!("out of bounds");
}

/// Bidirectional: search from end if n > 50% of true_count
/// Note: true_count must be passed in (don't compute it here!)
#[inline]
fn select_bidirectional(buf: &BitBuffer, n: usize, true_count: usize) -> usize {
    if n < true_count / 2 {
        select_simple(buf, n)
    } else {
        select_reverse(buf, true_count - 1 - n)
    }
}

/// Reverse select: find nth set bit from the end (no allocation)
#[inline]
fn select_reverse(buf: &BitBuffer, n: usize) -> usize {
    let mut remaining = n;
    let len = buf.len();
    let bytes = buf.inner().as_slice();
    let offset = buf.offset();

    // For simplicity, only handle aligned case for now
    if offset != 0 {
        // Fall back to forward search for unaligned
        let true_count = buf.true_count();
        return select_simple(buf, true_count - 1 - n);
    }

    let num_full_chunks = len / 64;
    let rem_bits = len % 64;

    // Handle remainder bits first (they're at the end)
    if rem_bits > 0 {
        let start = num_full_chunks * 8;
        let mut buf8 = [0u8; 8];
        let avail = (bytes.len() - start).min(8);
        buf8[..avail].copy_from_slice(&bytes[start..start + avail]);
        let rem = u64::from_le_bytes(buf8) & ((1u64 << rem_bits) - 1);

        let pop = rem.count_ones() as usize;
        if remaining < pop {
            return num_full_chunks * 64 + select_in_word_reverse(rem, remaining);
        }
        remaining -= pop;
    }

    // Iterate chunks in reverse (no allocation - direct pointer access)
    let ptr = bytes.as_ptr() as *const u64;
    for i in (0..num_full_chunks).rev() {
        let chunk = unsafe { ptr.add(i).read_unaligned() };
        let pop = chunk.count_ones() as usize;
        if remaining < pop {
            return i * 64 + select_in_word_reverse(chunk, remaining);
        }
        remaining -= pop;
    }

    panic!("out of bounds");
}

/// UnalignedBitChunk approach (for comparison)
#[inline]
fn select_bitchunk(buf: &BitBuffer, n: usize) -> usize {
    let mut remaining = n;
    let unaligned = UnalignedBitChunk::new(buf.inner().as_slice(), buf.offset(), buf.len());

    let lead_padding = unaligned.lead_padding();
    let mut bit_idx = 0usize;

    if let Some(prefix) = unaligned.prefix() {
        let popcount = prefix.count_ones() as usize;
        if remaining < popcount {
            let pos_in_prefix = select_in_word(prefix, remaining);
            return pos_in_prefix - lead_padding;
        }
        remaining -= popcount;
        bit_idx += 64 - lead_padding;
    }

    for &chunk in unaligned.chunks() {
        let popcount = chunk.count_ones() as usize;
        if remaining < popcount {
            return bit_idx + select_in_word(chunk, remaining);
        }
        remaining -= popcount;
        bit_idx += 64;
    }

    if let Some(suffix) = unaligned.suffix() {
        let popcount = suffix.count_ones() as usize;
        if remaining < popcount {
            return bit_idx + select_in_word(suffix, remaining);
        }
    }

    panic!("out of bounds");
}

// =============================================================================
// OPTIMAL: Combined implementation with all optimizations
// =============================================================================

/// Optimal select: combines block4 + bidirectional + unaligned handling
/// - Block4: 4 independent popcounts for ILP (1.75x speedup)
/// - Bidirectional: search from end if n > 50% (up to 39x speedup)
/// - UnalignedBitChunk: aligned loads for unaligned data (1.27x speedup)
#[inline]
fn select_optimal(buf: &BitBuffer, n: usize, true_count: usize) -> usize {
    // Decide search direction based on which end is closer
    if n < true_count / 2 {
        // Search forward
        if buf.offset() == 0 {
            select_aligned_forward_block4(buf, n)
        } else {
            select_unaligned_forward_block4(buf, n)
        }
    } else {
        // Search backward (from end)
        let n_from_end = true_count - 1 - n;
        if buf.offset() == 0 {
            select_aligned_reverse_block4(buf, n_from_end)
        } else {
            select_unaligned_reverse_block4(buf, n_from_end)
        }
    }
}

/// Forward search with block4 for aligned data
#[inline]
fn select_aligned_forward_block4(buf: &BitBuffer, n: usize) -> usize {
    let mut remaining = n;
    let bytes = buf.inner().as_slice();
    let len = buf.len();
    let ptr = bytes.as_ptr() as *const u64;
    let num_chunks = len / 64;
    let mut i = 0;

    // Process 4 chunks at a time
    while i + 4 <= num_chunks {
        let c0 = unsafe { ptr.add(i).read_unaligned() };
        let c1 = unsafe { ptr.add(i + 1).read_unaligned() };
        let c2 = unsafe { ptr.add(i + 2).read_unaligned() };
        let c3 = unsafe { ptr.add(i + 3).read_unaligned() };

        let pop0 = c0.count_ones() as usize;
        let pop1 = c1.count_ones() as usize;
        let pop2 = c2.count_ones() as usize;
        let pop3 = c3.count_ones() as usize;
        let block_pop = pop0 + pop1 + pop2 + pop3;

        if remaining < block_pop {
            if remaining < pop0 {
                return i * 64 + select_in_word(c0, remaining);
            }
            remaining -= pop0;
            if remaining < pop1 {
                return (i + 1) * 64 + select_in_word(c1, remaining);
            }
            remaining -= pop1;
            if remaining < pop2 {
                return (i + 2) * 64 + select_in_word(c2, remaining);
            }
            remaining -= pop2;
            return (i + 3) * 64 + select_in_word(c3, remaining);
        }
        remaining -= block_pop;
        i += 4;
    }

    // Handle remaining chunks
    while i < num_chunks {
        let chunk = unsafe { ptr.add(i).read_unaligned() };
        let pop = chunk.count_ones() as usize;
        if remaining < pop {
            return i * 64 + select_in_word(chunk, remaining);
        }
        remaining -= pop;
        i += 1;
    }

    // Handle remainder bits
    let rem_bits = len % 64;
    if rem_bits > 0 {
        let start = num_chunks * 8;
        let mut buf8 = [0u8; 8];
        let avail = (bytes.len() - start).min(8);
        buf8[..avail].copy_from_slice(&bytes[start..start + avail]);
        let rem = u64::from_le_bytes(buf8) & ((1u64 << rem_bits) - 1);
        if remaining < rem.count_ones() as usize {
            return num_chunks * 64 + select_in_word(rem, remaining);
        }
    }
    panic!("out of bounds");
}

/// Reverse search with block4 for aligned data
#[inline]
fn select_aligned_reverse_block4(buf: &BitBuffer, n_from_end: usize) -> usize {
    let mut remaining = n_from_end;
    let bytes = buf.inner().as_slice();
    let len = buf.len();
    let ptr = bytes.as_ptr() as *const u64;
    let num_chunks = len / 64;
    let rem_bits = len % 64;

    // Handle remainder bits first (they're at the end)
    if rem_bits > 0 {
        let start = num_chunks * 8;
        let mut buf8 = [0u8; 8];
        let avail = (bytes.len() - start).min(8);
        buf8[..avail].copy_from_slice(&bytes[start..start + avail]);
        let rem = u64::from_le_bytes(buf8) & ((1u64 << rem_bits) - 1);
        let pop = rem.count_ones() as usize;
        if remaining < pop {
            return num_chunks * 64 + select_in_word_reverse(rem, remaining);
        }
        remaining -= pop;
    }

    // Process chunks in reverse, 4 at a time
    let mut i = num_chunks;
    while i >= 4 {
        let c0 = unsafe { ptr.add(i - 1).read_unaligned() };
        let c1 = unsafe { ptr.add(i - 2).read_unaligned() };
        let c2 = unsafe { ptr.add(i - 3).read_unaligned() };
        let c3 = unsafe { ptr.add(i - 4).read_unaligned() };

        let pop0 = c0.count_ones() as usize;
        let pop1 = c1.count_ones() as usize;
        let pop2 = c2.count_ones() as usize;
        let pop3 = c3.count_ones() as usize;
        let block_pop = pop0 + pop1 + pop2 + pop3;

        if remaining < block_pop {
            if remaining < pop0 {
                return (i - 1) * 64 + select_in_word_reverse(c0, remaining);
            }
            remaining -= pop0;
            if remaining < pop1 {
                return (i - 2) * 64 + select_in_word_reverse(c1, remaining);
            }
            remaining -= pop1;
            if remaining < pop2 {
                return (i - 3) * 64 + select_in_word_reverse(c2, remaining);
            }
            remaining -= pop2;
            return (i - 4) * 64 + select_in_word_reverse(c3, remaining);
        }
        remaining -= block_pop;
        i -= 4;
    }

    // Handle remaining chunks
    while i > 0 {
        i -= 1;
        let chunk = unsafe { ptr.add(i).read_unaligned() };
        let pop = chunk.count_ones() as usize;
        if remaining < pop {
            return i * 64 + select_in_word_reverse(chunk, remaining);
        }
        remaining -= pop;
    }

    panic!("out of bounds");
}

/// Forward search with block4 for unaligned data (uses UnalignedBitChunk)
#[inline]
fn select_unaligned_forward_block4(buf: &BitBuffer, n: usize) -> usize {
    let mut remaining = n;
    let unaligned = UnalignedBitChunk::new(buf.inner().as_slice(), buf.offset(), buf.len());
    let lead_padding = unaligned.lead_padding();
    let mut bit_idx = 0usize;

    // Handle prefix
    if let Some(prefix) = unaligned.prefix() {
        let pop = prefix.count_ones() as usize;
        if remaining < pop {
            return select_in_word(prefix, remaining) - lead_padding;
        }
        remaining -= pop;
        bit_idx += 64 - lead_padding;
    }

    // Process aligned middle chunks with block4
    let chunks = unaligned.chunks();
    let num_chunks = chunks.len();
    let mut i = 0;

    while i + 4 <= num_chunks {
        let c0 = chunks[i];
        let c1 = chunks[i + 1];
        let c2 = chunks[i + 2];
        let c3 = chunks[i + 3];

        let pop0 = c0.count_ones() as usize;
        let pop1 = c1.count_ones() as usize;
        let pop2 = c2.count_ones() as usize;
        let pop3 = c3.count_ones() as usize;
        let block_pop = pop0 + pop1 + pop2 + pop3;

        if remaining < block_pop {
            if remaining < pop0 {
                return bit_idx + select_in_word(c0, remaining);
            }
            remaining -= pop0;
            if remaining < pop1 {
                return bit_idx + 64 + select_in_word(c1, remaining);
            }
            remaining -= pop1;
            if remaining < pop2 {
                return bit_idx + 128 + select_in_word(c2, remaining);
            }
            remaining -= pop2;
            return bit_idx + 192 + select_in_word(c3, remaining);
        }
        remaining -= block_pop;
        bit_idx += 256;
        i += 4;
    }

    // Handle remaining chunks
    while i < num_chunks {
        let chunk = chunks[i];
        let pop = chunk.count_ones() as usize;
        if remaining < pop {
            return bit_idx + select_in_word(chunk, remaining);
        }
        remaining -= pop;
        bit_idx += 64;
        i += 1;
    }

    // Handle suffix
    if let Some(suffix) = unaligned.suffix() {
        let pop = suffix.count_ones() as usize;
        if remaining < pop {
            return bit_idx + select_in_word(suffix, remaining);
        }
    }

    panic!("out of bounds");
}

/// Reverse search with block4 for unaligned data
#[inline]
fn select_unaligned_reverse_block4(buf: &BitBuffer, n_from_end: usize) -> usize {
    let mut remaining = n_from_end;
    let unaligned = UnalignedBitChunk::new(buf.inner().as_slice(), buf.offset(), buf.len());
    let lead_padding = unaligned.lead_padding();
    let chunks = unaligned.chunks();
    let num_chunks = chunks.len();

    // Calculate bit positions for suffix and chunks
    let prefix_bits = if unaligned.prefix().is_some() { 64 - lead_padding } else { 0 };
    let middle_bits = num_chunks * 64;
    let suffix_start = prefix_bits + middle_bits;

    // Handle suffix first (it's at the end)
    if let Some(suffix) = unaligned.suffix() {
        let pop = suffix.count_ones() as usize;
        if remaining < pop {
            return suffix_start + select_in_word_reverse(suffix, remaining);
        }
        remaining -= pop;
    }

    // Process aligned middle chunks in reverse with block4
    let mut i = num_chunks;
    while i >= 4 {
        let c0 = chunks[i - 1];
        let c1 = chunks[i - 2];
        let c2 = chunks[i - 3];
        let c3 = chunks[i - 4];

        let pop0 = c0.count_ones() as usize;
        let pop1 = c1.count_ones() as usize;
        let pop2 = c2.count_ones() as usize;
        let pop3 = c3.count_ones() as usize;
        let block_pop = pop0 + pop1 + pop2 + pop3;

        if remaining < block_pop {
            if remaining < pop0 {
                return prefix_bits + (i - 1) * 64 + select_in_word_reverse(c0, remaining);
            }
            remaining -= pop0;
            if remaining < pop1 {
                return prefix_bits + (i - 2) * 64 + select_in_word_reverse(c1, remaining);
            }
            remaining -= pop1;
            if remaining < pop2 {
                return prefix_bits + (i - 3) * 64 + select_in_word_reverse(c2, remaining);
            }
            remaining -= pop2;
            return prefix_bits + (i - 4) * 64 + select_in_word_reverse(c3, remaining);
        }
        remaining -= block_pop;
        i -= 4;
    }

    // Handle remaining middle chunks
    while i > 0 {
        i -= 1;
        let chunk = chunks[i];
        let pop = chunk.count_ones() as usize;
        if remaining < pop {
            return prefix_bits + i * 64 + select_in_word_reverse(chunk, remaining);
        }
        remaining -= pop;
    }

    // Handle prefix last
    if let Some(prefix) = unaligned.prefix() {
        let pop = prefix.count_ones() as usize;
        if remaining < pop {
            return select_in_word_reverse(prefix, remaining) - lead_padding;
        }
    }

    panic!("out of bounds");
}

// =============================================================================
// Test data generators
// =============================================================================

fn make_aligned_buf(len: usize) -> BitBuffer {
    BitBuffer::from_iter((0..len).map(|i| i % 10 == 0)) // 10% density
}

fn make_unaligned_buf(len: usize) -> BitBuffer {
    let buf = BitBuffer::from_iter((0..len + 1).map(|i| i % 10 == 0));
    buf.slice(1..len + 1)
}

// =============================================================================
// Benchmarks: Different percentiles (aligned)
// =============================================================================

#[divan::bench(args = PERCENTILES)]
fn aligned_simple(bencher: Bencher, pct: usize) {
    let buf = make_aligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(b, t)| select_simple(b, *t));
}

#[divan::bench(args = PERCENTILES)]
fn aligned_unrolled2(bencher: Bencher, pct: usize) {
    let buf = make_aligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(b, t)| select_unrolled2(b, *t));
}

#[divan::bench(args = PERCENTILES)]
fn aligned_block4(bencher: Bencher, pct: usize) {
    let buf = make_aligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(b, t)| select_block4(b, *t));
}

#[divan::bench(args = PERCENTILES)]
fn aligned_bidirectional(bencher: Bencher, pct: usize) {
    let buf = make_aligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target, true_count))
        .bench_refs(|(b, t, tc)| select_bidirectional(b, *t, *tc));
}

#[divan::bench(args = PERCENTILES)]
fn aligned_bitchunk(bencher: Bencher, pct: usize) {
    let buf = make_aligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(b, t)| select_bitchunk(b, *t));
}

// =============================================================================
// Benchmarks: Different percentiles (unaligned)
// =============================================================================

#[divan::bench(args = PERCENTILES)]
fn unaligned_simple(bencher: Bencher, pct: usize) {
    let buf = make_unaligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(b, t)| select_simple(b, *t));
}

#[divan::bench(args = PERCENTILES)]
fn unaligned_unrolled2(bencher: Bencher, pct: usize) {
    let buf = make_unaligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(b, t)| select_unrolled2(b, *t));
}

#[divan::bench(args = PERCENTILES)]
fn unaligned_bidirectional(bencher: Bencher, pct: usize) {
    let buf = make_unaligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target, true_count))
        .bench_refs(|(b, t, tc)| select_bidirectional(b, *t, *tc));
}

#[divan::bench(args = PERCENTILES)]
fn unaligned_bitchunk(bencher: Bencher, pct: usize) {
    let buf = make_unaligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target))
        .bench_refs(|(b, t)| select_bitchunk(b, *t));
}

// =============================================================================
// Benchmarks: Optimal (combined block4 + bidirectional + unaligned)
// =============================================================================

#[divan::bench(args = PERCENTILES)]
fn aligned_optimal(bencher: Bencher, pct: usize) {
    let buf = make_aligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target, true_count))
        .bench_refs(|(b, t, tc)| select_optimal(b, *t, *tc));
}

#[divan::bench(args = PERCENTILES)]
fn unaligned_optimal(bencher: Bencher, pct: usize) {
    let buf = make_unaligned_buf(SIZES[0]);
    let true_count = buf.true_count();
    let target = true_count * pct / 100;
    bencher
        .with_inputs(|| (&buf, target, true_count))
        .bench_refs(|(b, t, tc)| select_optimal(b, *t, *tc));
}
