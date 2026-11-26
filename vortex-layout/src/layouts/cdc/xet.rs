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
//! [Xet chunking spec]: https://huggingface.co/docs/xet/chunking

use std::ops::Range;

use super::GEAR_TABLE;

/// The target chunk size of the Xet chunker.
pub const XET_TARGET_CHUNK_SIZE: usize = 64 * 1024;

/// The minimum chunk size: boundary tests are skipped before this many bytes.
pub const XET_MIN_CHUNK_SIZE: usize = XET_TARGET_CHUNK_SIZE / 8;

/// The maximum chunk size: a boundary is forced at this many bytes.
pub const XET_MAX_CHUNK_SIZE: usize = XET_TARGET_CHUNK_SIZE * 2;

/// A chunk boundary is placed where the top 16 bits of the rolling hash are all zero.
pub const XET_BOUNDARY_MASK: u64 = 0xFFFF_0000_0000_0000;

/// Split `data` into content-defined chunks exactly as the Xet storage layer would.
///
/// Returns the byte ranges of every chunk, in order. Ranges are contiguous and cover all of
/// `data`. The boundaries follow the Xet chunking specification: the rolling hash
/// `h = (h << 1) + GEAR_TABLE[byte]` is updated for every byte, a boundary is taken at the
/// first position at least [`XET_MIN_CHUNK_SIZE`] bytes into the chunk where
/// `h & XET_BOUNDARY_MASK == 0`, a boundary is forced at [`XET_MAX_CHUNK_SIZE`] bytes, and the
/// hash is reset to zero after every boundary.
pub fn xet_chunks(data: &[u8]) -> Vec<Range<usize>> {
    let mut chunks = Vec::with_capacity(data.len() / XET_TARGET_CHUNK_SIZE + 1);
    let mut hash = 0u64;
    let mut start = 0usize;

    for (i, &byte) in data.iter().enumerate() {
        hash = (hash << 1).wrapping_add(GEAR_TABLE[byte as usize]);
        let size = i + 1 - start;
        if size < XET_MIN_CHUNK_SIZE {
            continue;
        }
        if size >= XET_MAX_CHUNK_SIZE || hash & XET_BOUNDARY_MASK == 0 {
            chunks.push(start..i + 1);
            start = i + 1;
            hash = 0;
        }
    }
    if start < data.len() {
        chunks.push(start..data.len());
    }
    chunks
}
