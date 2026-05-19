// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cranelift::prelude::InstBuilder;
use smallvec::{SmallVec, smallvec};
use vortex_error::VortexResult;

use crate::emit::{ArgKey, EmitCtx, SigBuilder};
use crate::form::{Form, Layout, PType};
use crate::stage::JitStage;

/// Terminal stage: store one block of lanes to the output buffer.
#[derive(Debug, Clone, Copy)]
pub struct StoreOut {
    pub ptype: PType,
}

impl JitStage for StoreOut {
    fn tag(&self) -> &'static str {
        "StoreOut"
    }

    fn fingerprint(&self) -> Vec<u8> {
        vec![self.ptype as u8]
    }

    fn input(&self) -> SmallVec<[Form; 1]> {
        smallvec![Form::Lane(self.ptype, Layout::Linear)]
    }

    fn output(&self) -> Form {
        Form::None
    }

    fn declare(&self, sig: &mut SigBuilder) {
        sig.request_arg(ArgKey::OutPtr);
    }

    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()> {
        let lanes = cx.take_input().into_lane(self.ptype)?;
        let out_ptr = cx.runtime_arg(&ArgKey::OutPtr)?;
        let block_idx = cx.block_idx();
        let n_lanes = cx.chunk_count();

        // base = out_ptr + block_idx * n_lanes * sizeof(T)
        let elems_per_block = {
            let n_v = cx.const_int(PType::I64, n_lanes as i64);
            cx.fb().ins().imul(block_idx, n_v)
        };
        let base = cx.offset_ptr(out_ptr, elems_per_block, self.ptype);

        let chunks: Vec<_> = lanes.chunks().to_vec();
        match lanes.lanes_per_chunk() {
            1 => {
                // Scalar path — emit one store per lane.
                for (i, v) in chunks.iter().enumerate() {
                    cx.store_lane(*v, base, i, self.ptype);
                }
            }
            _ => {
                // SIMD path — emit one vector store per chunk.
                for (c, v) in chunks.iter().enumerate() {
                    cx.store_chunk(*v, base, c, self.ptype);
                }
            }
        }
        Ok(())
    }
}
