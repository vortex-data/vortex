// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cranelift::prelude::InstBuilder;
use smallvec::{SmallVec, smallvec};
use vortex_error::VortexResult;

use crate::emit::{ArgKey, EmitCtx, Lanes, LaneSlice};
use crate::form::{Form, Layout, PType};
use crate::stage::JitStage;

/// In-block prefix sum with a per-block carry-in.
///
/// Serial dependency through the block, so v1 doesn't exploit SIMD within the
/// prefix sum itself. When the input is in SIMD form, this stage extracts
/// lanes to scalars, runs the serial prefix sum, then re-packs lanes back into
/// SIMD chunks for the next stage. The extract/pack overhead is real; a
/// production version would either:
///   - Convert to a per-lane SIMD prefix sum (Hillis-Steele within chunks +
///     scalar cross-chunk carry), or
///   - Force its consumers to accept scalar lanes when it produces them.
///
/// Carry-in: one scalar per block, loaded from the `delta_bases` runtime arg.
#[derive(Debug, Clone, Copy)]
pub struct DeltaPrefixSum {
    pub ptype: PType,
}

impl JitStage for DeltaPrefixSum {
    fn tag(&self) -> &'static str {
        "DeltaPrefixSum"
    }

    fn fingerprint(&self) -> Vec<u8> {
        vec![self.ptype as u8]
    }

    fn input(&self) -> SmallVec<[Form; 1]> {
        smallvec![Form::Lane(self.ptype, Layout::Either)]
    }

    fn output(&self) -> Form {
        Form::Lane(self.ptype, Layout::Linear)
    }

    fn declare(&self, sig: &mut crate::emit::SigBuilder) {
        sig.request_arg(ArgKey::Named("delta_bases"));
    }

    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()> {
        let lanes = cx.take_input().into_lane(self.ptype)?;
        let bases_ptr = cx.runtime_arg(&ArgKey::Named("delta_bases"))?;
        let block_idx = cx.block_idx();

        // base_for_this_block = *(bases_ptr + block_idx * sizeof(T))
        let base_ptr = cx.offset_ptr(bases_ptr, block_idx, self.ptype);
        let base = cx.load_lane(base_ptr, 0, self.ptype);

        let lpc = lanes.lanes_per_chunk();

        let mut running = base;
        if lpc == 1 {
            // Pure scalar fast path.
            let mut out = Vec::with_capacity(lanes.len());
            for x in lanes.chunks() {
                running = cx.fb().ins().iadd(running, *x);
                out.push(running);
            }
            cx.put_output(Lanes::Of(LaneSlice::new_scalar(out, self.ptype, Layout::Linear)));
        } else {
            // SIMD input: extract -> serial prefix -> insert.
            let chunks: Vec<_> = lanes.chunks().to_vec();
            let mut out_chunks = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                let mut new_chunk = chunk;
                for lane_idx in 0..lpc {
                    let scalar = cx.fb().ins().extractlane(chunk, lane_idx as u8);
                    running = cx.fb().ins().iadd(running, scalar);
                    new_chunk = cx.fb().ins().insertlane(new_chunk, running, lane_idx as u8);
                }
                out_chunks.push(new_chunk);
            }
            cx.put_output(Lanes::Of(LaneSlice::new_simd(
                out_chunks,
                self.ptype,
                Layout::Linear,
            )));
        }
        Ok(())
    }
}
