// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cranelift::prelude::InstBuilder;
use vortex_error::VortexResult;

use crate::emit::{ArgKey, EmitCtx, Lanes, LaneSlice, SigBuilder};
use crate::form::{Form, Layout, PType};
use crate::stage::JitStage;

/// Leaf stage: load one block of primitive values from the input buffer.
///
/// v1 emits SIMD chunk loads (`i32x4`/`i64x2`/etc.). The `block_size` in
/// `Pipeline` is the logical lanes per block; this stage produces
/// `block_size / simd_lanes` chunks per block.
#[derive(Debug, Clone, Copy)]
pub struct LoadIn {
    pub ptype: PType,
}

impl JitStage for LoadIn {
    fn tag(&self) -> &'static str {
        "LoadIn"
    }

    fn fingerprint(&self) -> Vec<u8> {
        vec![self.ptype as u8]
    }

    fn output(&self) -> Form {
        Form::Lane(self.ptype, Layout::Linear)
    }

    fn declare(&self, sig: &mut SigBuilder) {
        sig.request_arg(ArgKey::InPtr);
    }

    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()> {
        let in_ptr = cx.runtime_arg(&ArgKey::InPtr)?;
        let block_idx = cx.block_idx();
        let n_lanes = cx.chunk_count();
        let lanes_per_chunk = self.ptype.simd_lanes() as usize;
        assert!(
            n_lanes.is_multiple_of(lanes_per_chunk),
            "block_size {n_lanes} must be a multiple of simd_lanes {lanes_per_chunk}",
        );
        let n_chunks = n_lanes / lanes_per_chunk;

        // base = in_ptr + block_idx * n_lanes * sizeof(T)
        let elems_per_block = {
            let n_v = cx.const_int(PType::I64, n_lanes as i64);
            cx.fb().ins().imul(block_idx, n_v)
        };
        let base = cx.offset_ptr(in_ptr, elems_per_block, self.ptype);

        let mut chunks = Vec::with_capacity(n_chunks);
        for c in 0..n_chunks {
            chunks.push(cx.load_chunk(base, c, self.ptype));
        }
        cx.put_output(Lanes::Of(LaneSlice::new_simd(chunks, self.ptype, Layout::Linear)));
        Ok(())
    }
}
