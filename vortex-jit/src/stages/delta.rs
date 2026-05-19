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
/// v0: scalar serial prefix sum. The framework supports vector layouts via
/// `Form::Lane(_, Layout::FastLanesTransposed(N))` — a real Delta would use
/// that to do an N-way prefix sum with N independent carry chains.
///
/// Carry-in: one scalar per block, loaded from the `bases` runtime arg at
/// offset `block_idx * sizeof(T)`.
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

        // Serial prefix sum: out[i] = base + (sum of inputs[0..=i]).
        // Equivalent: running = base; for x in inputs { running += x; emit running. }
        let mut running = base;
        let mut out = Vec::with_capacity(lanes.len());
        for x in lanes.chunks() {
            running = cx.fb().ins().iadd(running, *x);
            out.push(running);
        }
        cx.put_output(Lanes::Of(LaneSlice::new(out, self.ptype, Layout::Linear)));
        Ok(())
    }
}
