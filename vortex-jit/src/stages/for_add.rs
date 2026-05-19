// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cranelift::prelude::InstBuilder;
use smallvec::{SmallVec, smallvec};
use vortex_error::VortexResult;

use crate::emit::{EmitCtx, Lanes};
use crate::form::{Form, Layout, PType};
use crate::stage::JitStage;

/// Frame-of-Reference add: every lane gets `reference` added in place.
///
/// v1: operates on SIMD chunks. The reference is broadcast once per block via
/// `splat`, then a single vector `iadd` per chunk. Cranelift lowers this to
/// one `vpaddd` (or equivalent) per chunk on the target ISA.
#[derive(Debug, Clone, Copy)]
pub struct ForAdd {
    pub ptype: PType,
    pub reference: i64,
}

impl JitStage for ForAdd {
    fn tag(&self) -> &'static str {
        "ForAdd"
    }

    fn fingerprint(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(9);
        v.push(self.ptype as u8);
        v.extend_from_slice(&self.reference.to_le_bytes());
        v
    }

    fn input(&self) -> SmallVec<[Form; 1]> {
        smallvec![Form::Lane(self.ptype, Layout::Either)]
    }

    fn output(&self) -> Form {
        Form::Lane(self.ptype, Layout::Linear)
    }

    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()> {
        let lanes = cx.take_input().into_lane(self.ptype)?;
        // Broadcast scalar reference into a SIMD chunk. Pre-block-loop hoisting
        // would be nicer, but the loop driver doesn't expose pre-loop hooks in
        // v1 — Cranelift's LICM should still hoist this splat out of the loop
        // body since it has no loop-variant inputs.
        let scalar_ref = cx.const_int(self.ptype, self.reference);
        let ref_chunk = if lanes.lanes_per_chunk() == 1 {
            scalar_ref
        } else {
            cx.splat(self.ptype, scalar_ref)
        };
        let out = lanes.map_chunks(cx.fb(), |fb, x| fb.ins().iadd(x, ref_chunk));
        cx.put_output(Lanes::Of(out));
        Ok(())
    }
}
