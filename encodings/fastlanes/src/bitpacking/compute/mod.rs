use std::mem;
use std::mem::MaybeUninit;

use vortex_array::compute::{
    BetweenFn, BetweenOptions, FilterKernelAdapter, KernelRef, ScalarAtFn, SearchSortedFn, SliceFn,
    TakeFn, between,
};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayComputeImpl, ArrayRef, IntoArray};
use vortex_error::VortexResult;

use crate::{BitPackedArray, BitPackedEncoding};

mod filter;
mod scalar_at;
mod search_sorted;
mod slice;
mod take;

impl ArrayComputeImpl for BitPackedArray {
    const FILTER: Option<KernelRef> = FilterKernelAdapter(BitPackedEncoding).some();
}

impl ComputeVTable for BitPackedEncoding {
    fn between_fn(&self) -> Option<&dyn BetweenFn<&dyn Array>> {
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
    let mut indices_within_chunk = [const { MaybeUninit::<usize>::uninit() }; 1024];
    let mut indices_len = 0;

    let Some(first_idx) = indices.next() else {
        return;
    };

    let mut current_chunk_idx = (first_idx + offset) / 1024;
    indices_within_chunk[indices_len] = MaybeUninit::new((first_idx + offset) % 1024);
    indices_len += 1;
    for idx in indices {
        let new_chunk_idx = (idx + offset) / 1024;

        if new_chunk_idx != current_chunk_idx {
            chunk_fn(current_chunk_idx, unsafe {
                mem::transmute::<&[MaybeUninit<usize>], &[usize]>(
                    &indices_within_chunk[..indices_len],
                )
            });
            indices_len = 0;
        }

        current_chunk_idx = new_chunk_idx;
        indices_within_chunk[indices_len] = MaybeUninit::new((idx + offset) % 1024);
        indices_len += 1;
    }

    if indices_len > 0 {
        chunk_fn(current_chunk_idx, unsafe {
            mem::transmute::<&[MaybeUninit<usize>], &[usize]>(&indices_within_chunk[..indices_len])
        });
    }
}

impl BetweenFn<&BitPackedArray> for BitPackedEncoding {
    fn between(
        &self,
        array: &BitPackedArray,
        lower: &dyn Array,
        upper: &dyn Array,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        if !lower.is_constant() || !upper.is_constant() {
            return Ok(None);
        };

        between(
            &array.clone().to_canonical()?.into_array(),
            lower,
            upper,
            options,
        )
        .map(Some)
    }
}
