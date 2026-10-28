// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared gather kernel for fixed-width [`Interleave`](super::super::Interleave) values.
//!
//! Each source holds a decoded buffer or one constant payload. Selector bounds are checked against
//! the logical source length before the kernel maps a row to its physical payload.

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

/// Physical storage for one fixed-width source array.
pub(super) struct Source<T> {
    /// One constant payload or a decoded payload for each logical row.
    pub(super) data: Buffer<T>,
    /// The logical row count, which can exceed `data.len()` for a constant.
    pub(super) len: usize,
    /// Maps rows to `data`: zero for constants and `usize::MAX` for decoded buffers.
    pub(super) row_mask: usize,
}

pub(super) fn gather<T, A, R>(
    values: &[Source<T>],
    branches: &[A],
    rows: &[R],
    allocator: &BufferAllocatorRef,
) -> VortexResult<Buffer<T>>
where
    T: Copy,
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    // `zip` truncates to the shorter input.
    vortex_ensure_eq!(
        rows.len(),
        branches.len(),
        "interleave row_indices length does not match array_indices length",
    );

    // Write rows through a `&mut` slice. Yielding `Result<T, _>` per row splits each `[u8; W]` copy
    // into narrow moves (the `Ok` payload is misaligned), and pushing through `BufferMut` reloads
    // its length on every row.
    let len = branches.len();
    let mut output = BufferMut::with_capacity_in(len, allocator.clone());
    let slots = &mut output.spare_capacity_mut()[..len];
    for ((slot, branch), row) in slots.iter_mut().zip(branches).zip(rows) {
        let Some(source) = values.get((*branch).as_()) else {
            vortex_bail!("interleave array index out of bounds");
        };
        let row = (*row).as_();
        vortex_ensure!(row < source.len, "interleave row index out of bounds");
        slot.write(source.data[row & source.row_mask]);
    }
    // SAFETY: the loop returns on any error, so reaching here means it wrote all `len` slots:
    // `slots`, `branches`, and `rows` all have length `len`.
    unsafe { output.set_len(len) };
    Ok(output.freeze())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::Source;
    use super::gather;
    use crate::memory::BufferAllocatorRef;

    #[rstest]
    #[case::array_index(1, 0, "array index out of bounds")]
    #[case::row_index(0, 1, "row index out of bounds")]
    fn rejects_out_of_bounds_selectors(
        #[case] branch: u8,
        #[case] row: u8,
        #[case] message: &str,
    ) -> VortexResult<()> {
        let values = [Source {
            data: Buffer::full_in(1u32, 1, BufferAllocatorRef::statically_allocated()),
            len: 1,
            row_mask: 0,
        }];

        let allocator = BufferAllocatorRef::statically_allocated();
        let error = gather(&values, &[branch], &[row], &allocator).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");

        Ok(())
    }
}
