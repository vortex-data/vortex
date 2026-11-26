// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A byte-level chunker matching the Hugging Face Xet chunking specification.
//!
//! This is *not* used when writing files. It exists so tests, benchmarks, and examples can
//! measure how well two Vortex files deduplicate against each other at the granularity the Xet
//! storage layer actually uses: GEAR-hashed chunks with a 64 KiB target size, as defined by the
//! [Xet chunking spec]. Byte ranges whose chunk hashes match between two file versions are
//! stored (and uploaded) only once by Xet-backed storage such as the Hugging Face Hub.
//!
//! The boundary scan runs on the [`gearhash`] crate's `next_match`, which dispatches to AVX2 or
//! SSE4.2 kernels at runtime where available. The GEAR table is [`gearhash::DEFAULT_TABLE`],
//! which the Xet chunking spec references normatively; a test pins its contents so an upstream
//! change cannot silently move chunk boundaries.
//!
//! [Xet chunking spec]: https://huggingface.co/docs/xet/chunking

use std::ops::Range;

use gearhash::Hasher;

/// The target chunk size of the Xet chunker.
pub const XET_TARGET_CHUNK_SIZE: usize = 64 * 1024;

/// The minimum chunk size: boundary tests are skipped before this many bytes.
pub const XET_MIN_CHUNK_SIZE: usize = XET_TARGET_CHUNK_SIZE / 8;

/// The maximum chunk size: a boundary is forced at this many bytes.
pub const XET_MAX_CHUNK_SIZE: usize = XET_TARGET_CHUNK_SIZE * 2;

/// A chunk boundary is placed where the top 16 bits of the rolling hash are all zero.
pub const XET_BOUNDARY_MASK: u64 = 0xFFFF_0000_0000_0000;

// The warm-up skip in `xet_chunks` requires the minimum chunk size to cover the 64-byte window
// the GEAR hash depends on.
const _: () = assert!(XET_MIN_CHUNK_SIZE >= 64);

/// Split `data` into content-defined chunks exactly as the Xet storage layer would.
///
/// Returns the byte ranges of every chunk, in order. Ranges are contiguous and cover all of
/// `data`. The boundaries follow the Xet chunking specification: the rolling hash
/// `h = (h << 1) + table[byte]` is updated for every byte, a boundary is taken at the
/// first position at least [`XET_MIN_CHUNK_SIZE`] bytes into the chunk where
/// `h & XET_BOUNDARY_MASK == 0`, a boundary is forced at [`XET_MAX_CHUNK_SIZE`] bytes, and the
/// hash is reset to zero after every boundary.
///
/// Each GEAR update shifts the previous state left by one bit, so a byte's contribution is gone
/// after 64 steps and the hash tested at any position is a function of the 64 bytes ending
/// there alone. The implementation leans on that twice: it skips hashing all but the last 63
/// bytes of each minimum-size window, and it scans with [`gearhash`]'s SIMD `next_match`, whose
/// parallel stripes are warmed up the same way. The resulting boundaries are identical to the
/// byte-at-a-time definition above, which the tests assert against a scalar reference.
pub fn xet_chunks(data: &[u8]) -> Vec<Range<usize>> {
    let mut chunks = Vec::with_capacity(data.len() / XET_TARGET_CHUNK_SIZE + 1);
    let mut hasher = Hasher::default();
    let mut start = 0usize;

    while data.len() - start >= XET_MIN_CHUNK_SIZE {
        // The earliest byte a boundary test may fire on is the one completing the minimum
        // chunk size. The hash there depends only on the 64 bytes ending at it, so warm the
        // hasher up over the 63 preceding bytes instead of the whole minimum-size window.
        let first_test = start + XET_MIN_CHUNK_SIZE - 1;
        let scan_end = data.len().min(start + XET_MAX_CHUNK_SIZE);
        hasher.set_hash(0);
        hasher.update(&data[first_test - 63..first_test]);

        if let Some(scanned) = hasher.next_match(&data[first_test..scan_end], XET_BOUNDARY_MASK) {
            chunks.push(start..first_test + scanned);
            start = first_test + scanned;
        } else if scan_end == start + XET_MAX_CHUNK_SIZE {
            // No boundary within the maximum chunk size: force one.
            chunks.push(start..scan_end);
            start = scan_end;
        } else {
            // The data ran out before the maximum chunk size: the tail is the final chunk.
            break;
        }
    }
    if start < data.len() {
        chunks.push(start..data.len());
    }
    chunks
}
