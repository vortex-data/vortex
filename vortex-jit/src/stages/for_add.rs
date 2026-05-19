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
/// This is the simplest fusion candidate. Composed after `LoadIn` or any
/// Lane-producing stage, it adds one IR `iadd` per lane — and because the
/// previous stage's outputs are SSA Values handed in via `take_input`, the
/// emitted IR is the lane + reference add with zero memory traffic between
/// the two stages.
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
        let ref_v = cx.const_int(self.ptype, self.reference);
        let out = lanes.map_chunks(cx.fb(), |fb, x| fb.ins().iadd(x, ref_v));
        cx.put_output(Lanes::Of(out));
        Ok(())
    }
}
