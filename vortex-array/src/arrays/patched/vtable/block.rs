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

pub(super) fn patch_op(array: ArrayView<'_, Patched>) -> Box<dyn BlockOp> {
    let offsets = array.lane_offsets().as_::<Primitive>().to_buffer::<u32>();
    let indices = array.patch_indices().as_::<Primitive>().to_buffer::<u16>();
    let values = array.patch_values().as_::<Primitive>();
    match_each_native_ptype!(values.ptype(), |T| {
        Box::new(ApplyPatches {
            next_patch: offsets[0] as usize,
            offsets,
            indices,
            values: values.to_buffer::<T>(),
            n_lanes: array.n_lanes(),
            offset: array.offset(),
        })
    })
}

struct ApplyPatches<T> {
    offsets: Buffer<u32>,
    indices: Buffer<u16>,
    values: Buffer<T>,
    n_lanes: usize,
    offset: usize,
    next_patch: usize,
}

impl<T: NativePType> BlockOp for ApplyPatches<T> {
    fn prefers_output(&self) -> bool {
        true
    }

    fn fuse_standalone(&self, rows: usize) -> bool {
        // Small standalone decodes already patch an owned, cache-resident output. The
        // million-row benchmark benefits from patching while each decoded block is hot.
        rows.div_ceil(BLOCK_LEN) >= 1024
    }

    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        let start = self.offset + block.start();
        let mut written = match block.values {
            Values::InPlace(values) => values.written(),
            copy => copy.map(|value: T| value),
        };
        let out = written.as_mut_slice::<T>();
        let end = start + out.len();
        for chunk in start / 1024..end.div_ceil(1024) {
            let stop = self.offsets[(chunk + 1) * self.n_lanes] as usize;
            let patches = self.next_patch..stop;
            for (&index, &value) in self.indices[patches.clone()]
                .iter()
                .zip(&self.values[patches])
            {
                let row = chunk * 1024 + usize::from(index);
                if let Some(dst) = out.get_mut(row.wrapping_sub(start)) {
                    *dst = value;
                }
            }
            // Patches are lane-ordered within each physical chunk. Keep its start if a
            // misaligned source splits it across two blocks.
            if end / 1024 > chunk {
                self.next_patch = stop;
            }
        }
        Ok(written)
    }
}
