use vortex_array::Array;
use vortex_array::compute::{IsConstantFn, ScalarAtFn, SearchSortedFn, SliceFn, TakeFn};
use vortex_array::vtable::ComputeVTable;

use crate::BitPackedEncoding;

mod between;
mod filter;
mod is_constant;
mod scalar_at;
mod search_sorted;
mod slice;
mod take;

impl ComputeVTable for BitPackedEncoding {
    fn is_constant_fn(&self) -> Option<&dyn IsConstantFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}

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
