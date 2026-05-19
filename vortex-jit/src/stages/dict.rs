// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cranelift::prelude::{InstBuilder, MemFlags};
use smallvec::{SmallVec, smallvec};
use vortex_error::{VortexResult, vortex_bail};

use crate::emit::{ArgKey, EmitCtx, Lanes, LaneSlice, SigBuilder};
use crate::form::{Form, Layout, PType};
use crate::stage::JitStage;

/// Dictionary lookup: `output[i] = values[codes[i]]`.
///
/// Consumes a stream of integer codes and produces a stream of values via
/// random indexing into a runtime-provided table. The lookup is a software
/// gather — each lane's code is extracted to a scalar, used to load from the
/// values table, then `insertlane`'d back into the output chunk. AVX2 has a
/// hardware gather (`vpgatherdd`) which Cranelift doesn't yet emit; this
/// scalar-extract approach is the portable fallback.
///
/// Code type currently must be `I32`. Value type can be any `PType`.
#[derive(Debug, Clone, Copy)]
pub struct DictLookup {
    pub code_ptype: PType,
    pub value_ptype: PType,
}

impl JitStage for DictLookup {
    fn tag(&self) -> &'static str {
        "DictLookup"
    }

    fn fingerprint(&self) -> Vec<u8> {
        vec![self.code_ptype as u8, self.value_ptype as u8]
    }

    fn input(&self) -> SmallVec<[Form; 1]> {
        smallvec![Form::Lane(self.code_ptype, Layout::Either)]
    }

    fn output(&self) -> Form {
        Form::Lane(self.value_ptype, Layout::Linear)
    }

    fn declare(&self, sig: &mut SigBuilder) {
        sig.request_arg(ArgKey::Named("dict_values"));
    }

    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()> {
        if !matches!(self.code_ptype, PType::I32) {
            vortex_bail!("DictLookup v0 supports only I32 codes");
        }
        let codes = cx.take_input().into_lane(self.code_ptype)?;
        let values_ptr = cx.runtime_arg(&ArgKey::Named("dict_values"))?;
        let value_byte_w = self.value_ptype.byte_width();
        let lpc = codes.lanes_per_chunk();

        // Build the output: for each input chunk, extract lanes as scalar
        // codes, load values[code], insert into a fresh output chunk.
        let zero_scalar = cx
            .fb()
            .ins()
            .iconst(self.value_ptype.cl_type(), 0);
        let zero_vec = if lpc > 1 {
            Some(cx.fb().ins().splat(self.value_ptype.simd_type(), zero_scalar))
        } else {
            None
        };

        let mut out_chunks = Vec::with_capacity(codes.len());
        for code_chunk in codes.chunks() {
            if lpc == 1 {
                // Scalar path: one code → one value
                let code = *code_chunk;
                let value = load_from_table(cx, values_ptr, code, self.value_ptype, value_byte_w);
                out_chunks.push(value);
            } else {
                let mut vec = zero_vec.unwrap();
                for lane in 0..lpc {
                    let code = cx.fb().ins().extractlane(*code_chunk, lane as u8);
                    let value = load_from_table(cx, values_ptr, code, self.value_ptype, value_byte_w);
                    vec = cx.fb().ins().insertlane(vec, value, lane as u8);
                }
                out_chunks.push(vec);
            }
        }

        let out_lanes = if lpc == 1 {
            LaneSlice::new_scalar(out_chunks, self.value_ptype, Layout::Linear)
        } else {
            LaneSlice::new_simd(out_chunks, self.value_ptype, Layout::Linear)
        };
        cx.put_output(Lanes::Of(out_lanes));
        Ok(())
    }
}

fn load_from_table(
    cx: &mut EmitCtx<'_, '_>,
    table_ptr: cranelift::prelude::Value,
    code: cranelift::prelude::Value,
    value_t: PType,
    value_byte_w: u32,
) -> cranelift::prelude::Value {
    // offset_bytes = code * sizeof(value)
    let stride = cx.fb().ins().iconst(
        cranelift::prelude::types::I64,
        i64::from(value_byte_w),
    );
    // codes are i32; extend to module pointer width.
    let code_pt = cx.module_pointer_type();
    let code_ext = if cx.fb().func.dfg.value_type(code) == code_pt {
        code
    } else {
        cx.fb().ins().uextend(code_pt, code)
    };
    let off = cx.fb().ins().imul(code_ext, stride);
    let addr = cx.fb().ins().iadd(table_ptr, off);
    cx.fb()
        .ins()
        .load(value_t.cl_type(), MemFlags::trusted(), addr, 0)
}
