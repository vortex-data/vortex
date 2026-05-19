// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cranelift::prelude::InstBuilder;
use smallvec::{SmallVec, smallvec};
use vortex_error::{VortexResult, vortex_bail};

use crate::emit::{EmitCtx, Lanes, LaneSlice};
use crate::form::{Form, Layout, PType};
use crate::stage::JitStage;

/// ALP decompression stage: int-to-float convert, then multiply by a
/// precomputed scale.
///
/// ALP encodes f32/f64 as i32/i64 with a per-array `(e, f)` pair of exponent
/// indices. Decode per element is `(encoded as Float) * F10[f] * IF10[e]`.
/// In a JIT we precompute the product `F10[f] * IF10[e]` as a single IR
/// constant and emit:
///   `fcvt_from_sint` (vcvtdq2ps on x86) + `fmul` (vmulps).
///
/// The current Vortex implementation at
/// `encodings/alp/src/alp/mod.rs:253-261` is `iter_mut().for_each(|v|
/// decode_single(...))` — a closure over a function call with table
/// lookups. LLVM's autovectorizer doesn't reliably penetrate that shape,
/// which is the gap this stage exists to close.
#[derive(Debug, Clone, Copy)]
pub struct AlpDecode {
    pub in_ptype: PType,
    pub out_ptype: PType,
    /// Precomputed `F10[f] * IF10[e]`.
    pub scale: f64,
}

impl JitStage for AlpDecode {
    fn tag(&self) -> &'static str {
        "AlpDecode"
    }

    fn fingerprint(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(10);
        v.push(self.in_ptype as u8);
        v.push(self.out_ptype as u8);
        v.extend_from_slice(&self.scale.to_le_bytes());
        v
    }

    fn input(&self) -> SmallVec<[Form; 1]> {
        smallvec![Form::Lane(self.in_ptype, Layout::Either)]
    }

    fn output(&self) -> Form {
        Form::Lane(self.out_ptype, Layout::Linear)
    }

    fn emit(&self, cx: &mut EmitCtx<'_, '_>) -> VortexResult<()> {
        // Validate the type pair.
        match (self.in_ptype, self.out_ptype) {
            (PType::I32, PType::F32) | (PType::I64, PType::F64) => {}
            (a, b) => vortex_bail!(
                "AlpDecode requires (I32->F32) or (I64->F64), got ({:?}->{:?})",
                a,
                b
            ),
        }

        let lanes = cx.take_input().into_lane(self.in_ptype)?;
        let lpc = lanes.lanes_per_chunk();

        let scale = match self.out_ptype {
            PType::F32 => cx.const_f32(self.scale as f32),
            PType::F64 => cx.const_f64(self.scale),
            _ => unreachable!(),
        };
        let scale_chunk = if lpc == 1 {
            scale
        } else {
            cx.splat(self.out_ptype, scale)
        };

        let out_cl = if lpc == 1 {
            self.out_ptype.cl_type()
        } else {
            self.out_ptype.simd_type()
        };

        let mut new_chunks = Vec::with_capacity(lanes.len());
        for chunk in lanes.chunks() {
            // i32x4 -> f32x4 on SSE2: vcvtdq2ps, single instruction.
            // Scalar i32 -> f32: cvtsi2ss.
            let as_float = cx.fb().ins().fcvt_from_sint(out_cl, *chunk);
            // Multiply by precomputed scale; LLVM-equivalent is vmulps.
            let scaled = cx.fb().ins().fmul(as_float, scale_chunk);
            new_chunks.push(scaled);
        }

        let out_lanes = if lpc == 1 {
            LaneSlice::new_scalar(new_chunks, self.out_ptype, Layout::Linear)
        } else {
            LaneSlice::new_simd(new_chunks, self.out_ptype, Layout::Linear)
        };
        cx.put_output(Lanes::Of(out_lanes));
        Ok(())
    }
}
