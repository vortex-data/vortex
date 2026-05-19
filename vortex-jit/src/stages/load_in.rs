// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cranelift::prelude::InstBuilder;
use vortex_error::VortexResult;

use crate::emit::{ArgKey, EmitCtx, Lanes, LaneSlice, SigBuilder};
use crate::form::{Form, Layout, PType};
use crate::stage::JitStage;

/// Leaf stage: load one block of primitive values from the input buffer.
///
/// v0: input is a plain primitive buffer of `block_size` elements per block.
/// A real BitPacked leaf would unpack bits here instead.
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
        let n = cx.chunk_count();

        // base + block_idx * block_size elements
        let elems_per_block = {
            let n_v = cx.const_int(PType::I64, n as i64);
            cx.fb().ins().imul(block_idx, n_v)
        };
        let base = cx.offset_ptr(in_ptr, elems_per_block, self.ptype);

        let mut chunks = Vec::with_capacity(n);
        for i in 0..n {
            chunks.push(cx.load_lane(base, i, self.ptype));
        }
        cx.put_output(Lanes::Of(LaneSlice::new(chunks, self.ptype, Layout::Linear)));
        Ok(())
    }
}
