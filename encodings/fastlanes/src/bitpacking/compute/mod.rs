use vortex_array::vtable::ComputeVTable;

use crate::BitPackedEncoding;

mod between;
mod filter;
mod is_constant;
mod take;

impl ComputeVTable for BitPackedEncoding {}

fn chunked_indices<F: FnMut(usize, &[usize])>(
    mut indices: impl Iterator<Item = usize>,
    offset: usize,
    mut chunk_fn: F,
) {
    let mut indices_within_chunk: Vec<usize> = Vec::with_capacity(1024);

    let Some(first_idx) = indices.next() else {
        return;
    };

    let mut current_chunk_idx = (first_idx + offset) / 1024;
    indices_within_chunk.push((first_idx + offset) % 1024);
    for idx in indices {
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

#[cfg(test)]
mod tests {
    use crate::bitpacking::compute::chunked_indices;

    #[test]
    fn chunk_indices_repeated() {
        let mut called = false;
        chunked_indices([0; 1025].into_iter(), 0, |chunk_idx, idxs| {
            assert_eq!(chunk_idx, 0);
            assert_eq!(idxs, [0; 1025]);
            called = true;
        });
        assert!(called);
    }
}
