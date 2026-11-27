// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::ArrayView;
use crate::arrays::Patched;
use crate::arrays::Primitive;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::block::BLOCK_LEN;
use crate::block::Block;
use crate::block::BlockOp;
use crate::block::Values;
use crate::block::Written;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;

pub(super) fn patch_op(array: ArrayView<'_, Patched>) -> Option<Box<dyn BlockOp>> {
    let values = array.patch_values().as_opt::<Primitive>()?;
    match_each_native_ptype!(values.ptype(), |T| {
        Some(Box::new(PatchKernel::<T>::try_new(array)?))
    })
}

/// Applies chunk-local patch indices to decoded primitive values.
#[derive(Clone)]
pub struct PatchKernel<T> {
    offsets: Buffer<u32>,
    indices: Buffer<u16>,
    values: Buffer<T>,
    n_lanes: usize,
    offset: usize,
}

impl<T: NativePType> PatchKernel<T> {
    /// Builds a kernel when the patch children are primitive and the values have type `T`.
    pub fn try_new(array: ArrayView<'_, Patched>) -> Option<Self> {
        let values = array.patch_values().as_opt::<Primitive>()?;
        if values.ptype() != T::PTYPE {
            return None;
        }
        Some(Self {
            offsets: array
                .lane_offsets()
                .as_opt::<Primitive>()?
                .to_buffer::<u32>(),
            indices: array
                .patch_indices()
                .as_opt::<Primitive>()?
                .to_buffer::<u16>(),
            values: values.to_buffer::<T>(),
            n_lanes: array.n_lanes(),
            offset: array.offset(),
        })
    }

    /// Applies the patches for the given decoded rows. `start` is relative to the logical array.
    // Keep full-block writes in the calling decode loop; partial windows share the fallback.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn apply(&self, start: usize, out: &mut [T]) {
        let start = self.offset + start;
        if start.is_multiple_of(BLOCK_LEN) && out.len() == BLOCK_LEN {
            let lane = (start / BLOCK_LEN) * self.n_lanes;
            let patches = self.offsets[lane] as usize..self.offsets[lane + self.n_lanes] as usize;
            for (&index, &value) in self.indices[patches.clone()]
                .iter()
                .zip(&self.values[patches])
            {
                out[usize::from(index)] = value;
            }
            return;
        }
        self.apply_partial(start, out);
    }

    #[inline(never)]
    fn apply_partial(&self, start: usize, out: &mut [T]) {
        let end = start + out.len();
        for chunk in start / BLOCK_LEN..end.div_ceil(BLOCK_LEN) {
            let lane = chunk * self.n_lanes;
            let patches = self.offsets[lane] as usize..self.offsets[lane + self.n_lanes] as usize;
            for (&index, &value) in self.indices[patches.clone()]
                .iter()
                .zip(&self.values[patches])
            {
                let row = chunk * BLOCK_LEN + usize::from(index);
                if let Some(dst) = out.get_mut(row.wrapping_sub(start)) {
                    *dst = value;
                }
            }
        }
    }
}

impl<T: NativePType> BlockOp for PatchKernel<T> {
    fn prefers_output(&self) -> bool {
        true
    }

    fn fuse_standalone(&self, rows: usize) -> bool {
        // Small standalone decodes already patch an owned, cache-resident output. The
        // million-row benchmark benefits from patching while each decoded block is hot.
        rows.div_ceil(BLOCK_LEN) >= 1024
    }

    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        let start = block.start();
        let mut written = match block.values {
            Values::InPlace(values) => values.written(),
            copy => copy.map(|value: T| value),
        };
        self.apply(start, written.as_mut_slice::<T>());
        Ok(written)
    }
}
