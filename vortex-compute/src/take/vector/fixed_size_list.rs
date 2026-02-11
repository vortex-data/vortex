// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Either;
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_dtype::UnsignedPType;
use vortex_mask::Mask;
use vortex_vector::Vector;
use vortex_vector::VectorOps;
use vortex_vector::fixed_size_list::FixedSizeListVector;
use vortex_vector::match_each_pvector;
use vortex_vector::primitive::PVector;

use crate::take::Take;

impl<I: UnsignedPType> Take<PVector<I>> for &FixedSizeListVector {
    type Output = FixedSizeListVector;

    fn take(self, indices: &PVector<I>) -> FixedSizeListVector {
        if indices.validity().all_true() {
            self.take(indices.elements().as_slice())
        } else {
            take_nullable(self, indices)
        }
    }
}

impl<I: UnsignedPType> Take<[I]> for &FixedSizeListVector {
    type Output = FixedSizeListVector;

    fn take(self, indices: &[I]) -> FixedSizeListVector {
        let list_size = self.list_size() as usize;
        let taken_validity = self.validity().take(indices);

        // If we are in the degenerate case (`list_size == 0`), then we just need to `take` on the
        // validity (since there is no backing data).
        if list_size == 0 {
            debug_assert!(
                self.elements().is_empty(),
                "The elements of a degenerate `FixedSizeListVector` is somehow non-empty"
            );

            // SAFETY: `list_size` is 0 and elements array is empty, so the invariant is maintained.
            return unsafe {
                FixedSizeListVector::new_unchecked(
                    self.elements().clone(),
                    self.list_size(),
                    taken_validity,
                )
            };
        }

        let taken_elements = take_fsl_elements(self.elements(), indices, list_size);

        debug_assert_eq!(taken_elements.len(), indices.len() * list_size);
        debug_assert_eq!(taken_validity.len(), indices.len());

        // SAFETY: We took elements at expanded indices and validity at list indices, maintaining
        // the invariant that `elements.len() == validity.len() * list_size`.
        unsafe {
            FixedSizeListVector::new_unchecked(
                Arc::new(taken_elements),
                self.list_size(),
                taken_validity,
            )
        }
    }
}

/// Takes elements from an FSL's backing vector using contiguous chunk copies when possible.
///
/// For primitive elements, this copies contiguous memory chunks (one `memcpy` per list index)
/// instead of expanding to individual element indices and doing per-element random access.
/// This is significantly faster for large list sizes (e.g., 1024-element feature vectors).
fn take_fsl_elements<I: UnsignedPType>(
    elements: &Arc<Vector>,
    list_indices: &[I],
    list_size: usize,
) -> Vector {
    if let Vector::Primitive(pv) = elements.as_ref() {
        match_each_pvector!(pv, |typed_pv| {
            Vector::Primitive(take_fsl_primitive_elements(typed_pv, list_indices, list_size).into())
        })
    } else {
        // Non-primitive elements: fall back to expand_indices.
        let element_indices = expand_indices(list_indices, list_size);
        elements.as_ref().take(element_indices.as_slice())
    }
}

/// Copies contiguous chunks of `list_size` elements from a primitive vector's buffer.
///
/// For each list index, copies a contiguous range of `list_size` elements from the source
/// buffer using `copy_nonoverlapping` instead of per-element gather.
fn take_fsl_primitive_elements<T: NativePType, I: UnsignedPType>(
    elements: &PVector<T>,
    list_indices: &[I],
    list_size: usize,
) -> PVector<T> {
    let total = list_indices.len() * list_size;
    let src = elements.elements().as_slice();

    let mut result = BufferMut::with_capacity(total);
    let dst_ptr = result.spare_capacity_mut().as_mut_ptr().cast::<T>();

    for (i, idx) in list_indices.iter().enumerate() {
        let src_start = (*idx).as_() * list_size;
        // SAFETY:
        // - `src` has length `n * list_size` (FSL invariant) and `src_start + list_size` is
        //   within bounds because `idx` is a valid list index.
        // - `dst` has capacity for `total` elements and we write at non-overlapping offsets.
        // - Source and destination don't overlap (destination is a fresh allocation).
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr().add(src_start),
                dst_ptr.add(i * list_size),
                list_size,
            );
        }
    }

    // SAFETY: We wrote exactly `total` elements.
    unsafe { result.set_len(total) };
    let taken_buf = result.freeze();

    let taken_validity = if elements.validity().all_true() {
        Mask::new_true(total)
    } else {
        // For nullable elements, expand indices and take validity from the mask.
        let element_indices = expand_indices(list_indices, list_size);
        elements.validity().take(element_indices.as_slice())
    };

    // SAFETY: Both buffer and validity have length `total`.
    unsafe { PVector::new_unchecked(taken_buf, taken_validity) }
}

fn take_nullable<I: UnsignedPType>(
    fsl: &FixedSizeListVector,
    indices: &PVector<I>,
) -> FixedSizeListVector {
    let list_size = fsl.list_size() as usize;
    let taken_validity = fsl.validity().take(indices);

    // If we are in the degenerate case (`list_size == 0`), then we just need to `take` on the
    // validity (since there is no backing data).
    if list_size == 0 {
        debug_assert!(
            fsl.elements().is_empty(),
            "The elements of a degenerate `FixedSizeListVector` is somehow non-empty"
        );

        // SAFETY: `list_size` is 0 and elements array is empty, so the invariant is maintained.
        return unsafe {
            FixedSizeListVector::new_unchecked(
                fsl.elements().clone(),
                fsl.list_size(),
                taken_validity,
            )
        };
    }

    let taken_elements = take_fsl_elements_nullable(fsl.elements(), indices, list_size);

    debug_assert_eq!(taken_elements.len(), indices.len() * list_size);
    debug_assert_eq!(taken_validity.len(), indices.len());

    // SAFETY: We took elements at expanded indices and validity at list indices, maintaining the
    // invariant that `elements.len() == validity.len() * list_size`.
    unsafe {
        FixedSizeListVector::new_unchecked(
            Arc::new(taken_elements),
            fsl.list_size(),
            taken_validity,
        )
    }
}

/// Takes elements from an FSL's backing vector when indices may be null, using contiguous
/// chunk copies when possible.
fn take_fsl_elements_nullable<I: UnsignedPType>(
    elements: &Arc<Vector>,
    list_indices: &PVector<I>,
    list_size: usize,
) -> Vector {
    if let Vector::Primitive(pv) = elements.as_ref() {
        match_each_pvector!(pv, |typed_pv| {
            Vector::Primitive(
                take_fsl_primitive_elements_nullable(typed_pv, list_indices, list_size).into(),
            )
        })
    } else {
        // Non-primitive elements: fall back to expand_nullable_indices.
        let expanded_nullable_indices = expand_nullable_indices(list_indices, list_size);
        elements.as_ref().take(expanded_nullable_indices.as_slice())
    }
}

/// Copies contiguous chunks from a primitive vector's buffer, handling null indices by
/// substituting reads from position 0 (the data is irrelevant since list-level validity
/// will mask it out).
fn take_fsl_primitive_elements_nullable<T: NativePType, I: UnsignedPType>(
    elements: &PVector<T>,
    list_indices: &PVector<I>,
    list_size: usize,
) -> PVector<T> {
    let total = list_indices.len() * list_size;
    let src = elements.elements().as_slice();

    let mut result = BufferMut::with_capacity(total);
    let dst_ptr = result.spare_capacity_mut().as_mut_ptr().cast::<T>();

    list_indices.validity().iter_bools(|validity_iter| {
        for (i, (idx, is_valid)) in list_indices
            .elements()
            .iter()
            .zip(validity_iter)
            .enumerate()
        {
            // For null indices, copy from position 0 as a placeholder (list validity masks it).
            let src_start = if is_valid {
                (*idx).as_() * list_size
            } else {
                0
            };
            // SAFETY: Same as `take_fsl_primitive_elements`. For null indices, position 0 is
            // always valid because `list_size > 0` implies the elements buffer is non-empty.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_start),
                    dst_ptr.add(i * list_size),
                    list_size,
                );
            }
        }
    });

    // SAFETY: We wrote exactly `total` elements.
    unsafe { result.set_len(total) };
    let taken_buf = result.freeze();

    let taken_validity = if elements.validity().all_true() {
        Mask::new_true(total)
    } else {
        let expanded = expand_nullable_indices(list_indices, list_size);
        elements.validity().take(expanded.as_slice())
    };

    // SAFETY: Both buffer and validity have length `total`.
    unsafe { PVector::new_unchecked(taken_buf, taken_validity) }
}

// TODO(connor): Ideally we match on the pointe width and return either a `Vec<u64>` or `Vec<u32>`
// (feature gated by `#[cfg(target_pointer_width = "64")]`), but that is probably overkill.
/// "Expands" the given indices by constructing new indices where for each list index `i`, we fill
/// add indices `[i*list_size..(i+1)*list_size]`.
///
/// The resulting indices vector will have length `indices.len() * list_size`.
fn expand_indices<I: UnsignedPType>(indices: &[I], list_size: usize) -> Vec<u64> {
    let list_size_u64 = list_size as u64;

    // Expand list indices to element indices.
    let expanded_indices: Vec<u64> = indices
        .iter()
        .flat_map(|idx| {
            let list_idx = (*idx).as_() as u64;
            let start = list_idx * list_size_u64;
            let end = (list_idx + 1) * list_size_u64;
            start..end
        })
        .collect();

    debug_assert_eq!(indices.len() * list_size, expanded_indices.len());
    expanded_indices
}

// TODO(connor): Ideally we match on the pointe width and return either a `Vec<u64>` or `Vec<u32>`
// (feature gated by `#[cfg(target_pointer_width = "64")]`), but that is probably overkill.
/// "Expands" the given indices by constructing new indices where for each list index `i`, we fill
/// add indices `[i*list_size..(i+1)*list_size]`.
///
/// The resulting indices vector will have length `indices.len() * list_size`.
///
/// For null indices that we need to expand, we use index 0 as a placeholder for all of them since
/// the validity will mask them out anyway.
fn expand_nullable_indices<I: UnsignedPType>(indices: &PVector<I>, list_size: usize) -> Vec<u64> {
    let indices_validity = indices.validity();
    let list_size_u64 = list_size as u64;

    indices_validity.iter_bools(|validity_iter| {
        let expanded_indices: Vec<u64> = indices
            .elements()
            .iter()
            .zip(validity_iter)
            .flat_map(|(idx, is_valid)| {
                if is_valid {
                    let list_idx = (*idx).as_() as u64;
                    let start = list_idx * list_size_u64;
                    let end = (list_idx + 1) * list_size_u64;
                    Either::Left(start..end)
                } else {
                    Either::Right(std::iter::repeat_n(0u64, list_size))
                }
            })
            .collect();

        debug_assert_eq!(indices.len() * list_size, expanded_indices.len());
        expanded_indices
    })
}
