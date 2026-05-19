// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Token-space block front-coding over OnPair codes.
//!
//! Per block of `K` rows in sort order:
//! - The first row in the block is stored as an **anchor**: full token sequence.
//! - Each subsequent row stores `(shared_with_prev: u16, suffix: &[u16])`.
//!
//! Reconstruction of row `i` walks the chain inside the block from the anchor
//! forward, so random access cost is O(K). With K=256 that's ≤256 token-prefix
//! copies, which is cheap.

/// Longest common token prefix.
#[inline]
fn lcp(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Total *payload* bytes for the front-coded representation, excluding the
/// dict itself (the dict is stored once per column, see `encoders.rs`).
///
/// Token IDs are bit-packed at `bits_per_token` bits each (matching the
/// underlying OnPair encoding's bit width). Per non-anchor row stores a
/// 2-byte `shared_with_prev` count.
pub fn front_coded_size(rows: &[&[u16]], block: usize, bits_per_token: u32) -> usize {
    assert!(block > 0);
    let n = rows.len();
    // Per-row offsets into the payload (random access).
    let offset_bytes = n * 4;
    let mut total_token_bits: u64 = 0;
    let mut shared_count_bytes: usize = 0;
    let mut i = 0;
    while i < n {
        let end = (i + block).min(n);
        // anchor row: full token sequence
        total_token_bits += rows[i].len() as u64 * bits_per_token as u64;
        for j in (i + 1)..end {
            let shared = lcp(rows[j - 1], rows[j]);
            let suffix = rows[j].len() - shared;
            total_token_bits += suffix as u64 * bits_per_token as u64;
            shared_count_bytes += 2;
        }
        i = end;
    }
    offset_bytes + shared_count_bytes + ((total_token_bits + 7) / 8) as usize
}

/// Byte-level front-coding for comparison. Each row stores `(shared_u32, suffix)`
/// against the previous row in the block. Anchor every K rows.
pub fn front_coded_bytes_size(rows: &[&[u8]], block: usize) -> usize {
    assert!(block > 0);
    let n = rows.len();
    let mut total = 0;
    total += n * 4; // per-row offsets
    let mut i = 0;
    while i < n {
        let end = (i + block).min(n);
        total += rows[i].len();
        for j in (i + 1)..end {
            let shared = rows[j - 1]
                .iter()
                .zip(rows[j].iter())
                .take_while(|(x, y)| x == y)
                .count();
            let suffix = rows[j].len() - shared;
            total += 4 + suffix; // 4 bytes shared count (u32) + suffix bytes
        }
        i = end;
    }
    total
}
