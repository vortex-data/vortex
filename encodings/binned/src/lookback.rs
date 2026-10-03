// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-local lookback deltas: each latent is stored as its difference from an earlier latent
//! in the same block, plus the distance to that latent.
//!
//! The candidate search is adapted from Pco's `delta/lookback.rs`. Pco looks back across a
//! window of up to 2^15 values; here references never leave the block, so every block stays
//! independently decodable. A distance of 0 means "no reference" and is used for each block's
//! first value.

use crate::latent::Latent;
use crate::stream::BLOCK_SIZE;

const PROPOSED: usize = 16;
const BRUTE: usize = 6;
const REPEATING: usize = 4;
const COARSENESSES: [u32; 2] = [0, 8];
const HASH_LOG: u32 = 11;

fn hash(mut x: u64) -> usize {
    x = (x ^ (x >> 32)).wrapping_mul(11_400_714_819_323_197_441);
    x ^= x >> 32;
    #[allow(clippy::cast_possible_truncation)]
    let h = (x as usize) & ((1 << HASH_LOG) - 1);
    h
}

fn leading_zeros<L: Latent>(x: L) -> u32 {
    x.to_u64().leading_zeros() - (u64::BITS - L::BITS)
}

/// Chooses a lookback distance for every latent.
pub(crate) fn choose<L: Latent>(latents: &[L]) -> Vec<u16> {
    let table_n = 1usize << HASH_LOG;
    // Holds `index + 1` of the last latent seen per hash bucket; 0 is empty.
    let mut table = vec![0usize; COARSENESSES.len() * table_n];
    let mut counts = vec![1u32; BLOCK_SIZE];
    let mut goodness = vec![1u32; BLOCK_SIZE];
    let mut proposals = [0usize; PROPOSED];
    for (k, p) in proposals.iter_mut().take(BRUTE).enumerate() {
        *p = k + 1;
    }
    let mut lookbacks = Vec::with_capacity(latents.len());
    let mut best = 1usize;
    let mut repeating_idx = 0usize;

    for (i, &l) in latents.iter().enumerate() {
        let local = i % BLOCK_SIZE;
        // Hash proposals: the last index with a nearby value, at two coarsenesses.
        let mut slot = BRUTE + REPEATING;
        for (c, coarseness) in COARSENESSES.iter().enumerate() {
            let bucket = l.to_u64() >> coarseness;
            let offset = c * table_n;
            for b in [bucket.wrapping_sub(1), bucket, bucket.wrapping_add(1)] {
                let last = table[offset + hash(b)];
                proposals[slot] = if last == 0 { 0 } else { i + 1 - last };
                slot += 1;
            }
            table[offset + hash(bucket)] = i + 1;
        }
        if local == 0 {
            lookbacks.push(0);
            continue;
        }

        let mut best_goodness = 0;
        let mut new_best = 0;
        for &lb in &proposals {
            if lb == 0 || lb > local {
                continue;
            }
            let other = latents[i - lb];
            let delta = l.wrapping_sub(other).min(other.wrapping_sub(l));
            let g = goodness[lb - 1] + leading_zeros(delta);
            if g > best_goodness {
                best_goodness = g;
                new_best = lb;
            }
        }
        let new_best = new_best.max(1);
        if new_best != best {
            repeating_idx += 1;
        }
        proposals[BRUTE + repeating_idx % REPEATING] = new_best;
        best = new_best;
        let count = &mut counts[best - 1];
        *count += 1;
        if count.is_power_of_two() {
            goodness[best - 1] += 1;
        }
        #[allow(clippy::cast_possible_truncation)]
        lookbacks.push(best as u16);
    }
    lookbacks
}

/// Replaces each latent with its centered difference from the latent it looks back to.
pub(crate) fn encode<L: Latent>(latents: &[L], lookbacks: &[u16]) -> Vec<L> {
    latents
        .iter()
        .zip(lookbacks)
        .enumerate()
        .map(|(i, (&l, &lb))| {
            let reference = if lb == 0 {
                L::ZERO
            } else {
                latents[i - usize::from(lb)]
            };
            l.wrapping_sub(reference).wrapping_add(L::MID)
        })
        .collect()
}

/// Inverts [`encode`] in place over one block's prefix. Distances that reach before the block
/// (only possible in corrupt data) are treated as 0.
#[inline]
pub(crate) fn decode_in_place<L: Latent>(deltas: &mut [L], lookbacks: &[u16]) {
    for i in 0..deltas.len() {
        let lb = usize::from(lookbacks[i]);
        let reference = if lb == 0 || lb > i {
            L::ZERO
        } else {
            deltas[i - lb]
        };
        deltas[i] = reference.wrapping_add(deltas[i]).wrapping_sub(L::MID);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_finds_repeats() {
        let pattern = [17u32, 900, 4, 4_000_000, 33];
        let latents: Vec<u32> = (0..3000).map(|i| pattern[i % 5] + (i as u32 % 3)).collect();
        let lookbacks = choose(&latents);
        assert!(lookbacks.iter().all(|&lb| usize::from(lb) < BLOCK_SIZE));
        let mut deltas = encode(&latents, &lookbacks);
        for block in deltas.chunks_mut(BLOCK_SIZE).zip(lookbacks.chunks(BLOCK_SIZE)) {
            decode_in_place(block.0, block.1);
        }
        assert_eq!(deltas, latents);
        let fives = lookbacks.iter().filter(|&&lb| lb == 5 || lb == 15).count();
        assert!(fives > 2500, "{fives}");
    }
}
