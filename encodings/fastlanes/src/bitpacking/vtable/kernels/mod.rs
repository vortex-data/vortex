// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod filter;
mod slice;

/// Assuming the buffer is already allocated (which will happen at most once), then unpacking all
/// 1024 elements takes ~8.8x as long as unpacking a single element on an M2 Macbook Air.
///
/// See https://github.com/vortex-data/vortex/pull/190#issue-2223752833
const UNPACK_CHUNK_THRESHOLD: usize = 8;

fn chunked_indices<F: FnMut(usize, &[usize])>(indices: &[usize], offset: usize, mut chunk_fn: F) {
    if indices.is_empty() {
        return;
    }

    let mut indices_within_chunk: Vec<usize> = Vec::with_capacity(1024);

    let first_idx = indices[0];
    let mut current_chunk_idx = (first_idx + offset) / 1024;
    indices_within_chunk.push((first_idx + offset) % 1024);

    for idx in &indices[1..] {
        let new_chunk_idx = (idx + offset) / 1024;

        if new_chunk_idx != current_chunk_idx {
            chunk_fn(current_chunk_idx, &indices_within_chunk);
            indices_within_chunk.clear();
        }

        current_chunk_idx = new_chunk_idx;
        indices_within_chunk.push((idx + offset) % 1024);
    }

    if !indices_within_chunk.is_empty() {
        chunk_fn(current_chunk_idx, &indices_within_chunk);
    }
}
