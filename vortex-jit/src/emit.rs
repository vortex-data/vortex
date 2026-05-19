// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The stage authoring contract.
//!
//! Stages don't touch `FunctionBuilder` directly for the common case — they
//! read inputs via `EmitCtx::take_input`, transform them through
//! `LaneSlice::map_chunks`, and hand outputs to `EmitCtx::put_output`. The
//! escape hatch (`EmitCtx::fb`) is there for ops the typed wrappers don't
//! cover.

use std::collections::HashMap;

use cranelift::codegen::ir::FuncRef;
use cranelift::prelude::{
    FunctionBuilder, InstBuilder, MemFlags, Type as ClType, Value as ClValue,
};
use vortex_error::{VortexResult, vortex_bail, vortex_err};

use crate::form::{Layout, PType};

/// Runtime arguments a stage requests via `EmitCtx::runtime_arg`.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ArgKey {
    /// Pointer to the leaf input buffer.
    InPtr,
    /// Number of *blocks* the kernel will iterate.
    NBlocks,
    /// Pointer to the canonical output buffer.
    OutPtr,
    /// Per-stage named pointer (e.g. delta bases, FoR reference scalar,
    /// patches index/value arrays).
    Named(&'static str),
}

/// Identifier for an extern symbol the stage wants to call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExternId(pub(crate) u32);

/// Signature builder — collected once at compile time across all stages.
#[derive(Debug, Default)]
pub struct SigBuilder {
    pub(crate) args: Vec<ArgKey>,
    pub(crate) externs: Vec<&'static str>,
}

impl SigBuilder {
    pub fn request_arg(&mut self, key: ArgKey) {
        if !self.args.contains(&key) {
            self.args.push(key);
        }
    }

    pub fn request_extern(&mut self, name: &'static str) -> ExternId {
        if let Some(idx) = self.externs.iter().position(|n| *n == name) {
            ExternId(idx as u32)
        } else {
            let idx = self.externs.len();
            self.externs.push(name);
            ExternId(idx as u32)
        }
    }
}

/// A single SSA value with its primitive type.
#[derive(Debug, Clone, Copy)]
pub struct Scalar {
    pub(crate) v: ClValue,
    pub(crate) t: PType,
}

impl Scalar {
    pub fn value(self) -> ClValue {
        self.v
    }
    pub fn ptype(self) -> PType {
        self.t
    }
}

/// A typed lane container.
///
/// v0 implementation: one SSA `Value` per logical lane (chunk_width = 1).
/// A SIMD-aware version would hold one `Value` per `i32x8` chunk; the
/// map/store APIs are written so the SIMD lift is local — only `chunks` and
/// `len()` semantics change.
#[derive(Debug, Clone)]
pub struct LaneSlice {
    pub(crate) chunks: Vec<ClValue>,
    pub(crate) t: PType,
    pub(crate) layout: Layout,
}

impl LaneSlice {
    pub fn new(chunks: Vec<ClValue>, t: PType, layout: Layout) -> Self {
        Self { chunks, t, layout }
    }

    pub fn ptype(&self) -> PType {
        self.t
    }
    pub fn layout(&self) -> Layout {
        self.layout
    }
    pub fn len(&self) -> usize {
        self.chunks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
    pub fn chunks(&self) -> &[ClValue] {
        &self.chunks
    }

    /// Map every chunk through `f`, returning a new `LaneSlice` of the same
    /// physical width.
    pub fn map_chunks<F>(self, fb: &mut FunctionBuilder<'_>, mut f: F) -> Self
    where
        F: FnMut(&mut FunctionBuilder<'_>, ClValue) -> ClValue,
    {
        let out: Vec<ClValue> = self.chunks.into_iter().map(|c| f(fb, c)).collect();
        Self {
            chunks: out,
            t: self.t,
            layout: self.layout,
        }
    }

    /// Map with paired access to a second slice of the same length.
    pub fn map_chunks_paired<F>(
        self,
        other: &Self,
        fb: &mut FunctionBuilder<'_>,
        mut f: F,
    ) -> VortexResult<Self>
    where
        F: FnMut(&mut FunctionBuilder<'_>, ClValue, ClValue) -> ClValue,
    {
        if self.chunks.len() != other.chunks.len() {
            vortex_bail!(
                "map_chunks_paired length mismatch: {} vs {}",
                self.chunks.len(),
                other.chunks.len()
            );
        }
        let out: Vec<ClValue> = self
            .chunks
            .into_iter()
            .zip(other.chunks.iter().copied())
            .map(|(a, b)| f(fb, a, b))
            .collect();
        Ok(Self {
            chunks: out,
            t: self.t,
            layout: self.layout,
        })
    }
}

/// The lane container exchanged between stages.
#[derive(Debug, Clone)]
pub enum Lanes {
    Of(LaneSlice),
    None,
}

impl Lanes {
    pub fn into_lane(self, expected: PType) -> VortexResult<LaneSlice> {
        match self {
            Self::Of(l) if l.t == expected => Ok(l),
            Self::Of(l) => vortex_bail!(
                "lane type mismatch: expected {:?}, got {:?}",
                expected,
                l.t
            ),
            Self::None => vortex_bail!("expected lane input, got None"),
        }
    }
}

/// The context handed to `JitStage::emit`.
pub struct EmitCtx<'a, 'b> {
    pub(crate) fb: &'a mut FunctionBuilder<'b>,
    pub(crate) args: &'a HashMap<ArgKey, ClValue>,
    pub(crate) externs_by_name: &'a HashMap<&'static str, FuncRef>,
    pub(crate) input: Lanes,
    pub(crate) output: Option<Lanes>,
    pub(crate) block_idx: ClValue,
    pub(crate) chunk_count: usize,
    pub(crate) module_pt: ClType,
}

impl<'a, 'b> EmitCtx<'a, 'b> {
    pub fn fb(&mut self) -> &mut FunctionBuilder<'b> {
        self.fb
    }

    pub fn block_idx(&self) -> ClValue {
        self.block_idx
    }

    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    pub fn module_pointer_type(&self) -> ClType {
        self.module_pt
    }

    pub fn take_input(&mut self) -> Lanes {
        std::mem::replace(&mut self.input, Lanes::None)
    }

    pub fn put_output(&mut self, lanes: Lanes) {
        self.output = Some(lanes);
    }

    pub fn runtime_arg(&self, key: &ArgKey) -> VortexResult<ClValue> {
        self.args
            .get(key)
            .copied()
            .ok_or_else(|| vortex_err!("runtime arg not declared: {:?}", key))
    }

    pub fn const_int(&mut self, t: PType, value: i64) -> ClValue {
        self.fb.ins().iconst(t.cl_type(), value)
    }

    pub fn const_f64(&mut self, value: f64) -> ClValue {
        self.fb.ins().f64const(value)
    }

    pub fn const_f32(&mut self, value: f32) -> ClValue {
        self.fb.ins().f32const(value)
    }

    /// Load one scalar lane from a buffer at `base + lane_idx * elem_width`.
    pub fn load_lane(&mut self, base_ptr: ClValue, lane_idx: usize, t: PType) -> ClValue {
        let off = i32::try_from(lane_idx * t.byte_width() as usize)
            .expect("lane offset fits in i32");
        self.fb
            .ins()
            .load(t.cl_type(), MemFlags::trusted(), base_ptr, off)
    }

    /// Store one scalar lane.
    pub fn store_lane(&mut self, value: ClValue, base_ptr: ClValue, lane_idx: usize, t: PType) {
        let off = i32::try_from(lane_idx * t.byte_width() as usize)
            .expect("lane offset fits in i32");
        self.fb.ins().store(MemFlags::trusted(), value, base_ptr, off);
    }

    /// Pointer arithmetic: `base + (offset_elems * elem_width)`.
    pub fn offset_ptr(&mut self, base: ClValue, offset_elems: ClValue, t: PType) -> ClValue {
        let w = self
            .fb
            .ins()
            .iconst(self.module_pt, i64::from(t.byte_width()));
        let off_extended = self.maybe_extend(offset_elems);
        let off_bytes = self.fb.ins().imul(off_extended, w);
        self.fb.ins().iadd(base, off_bytes)
    }

    fn maybe_extend(&mut self, v: ClValue) -> ClValue {
        let v_t = self.fb.func.dfg.value_type(v);
        if v_t == self.module_pt {
            v
        } else if v_t.bytes() < self.module_pt.bytes() {
            self.fb.ins().uextend(self.module_pt, v)
        } else {
            self.fb.ins().ireduce(self.module_pt, v)
        }
    }

    /// Call an extern Rust helper by registered name, returning its results.
    pub fn extern_call_by_name(&mut self, name: &'static str, args: &[ClValue]) -> Vec<ClValue> {
        let func_ref = *self
            .externs_by_name
            .get(name)
            .unwrap_or_else(|| panic!("extern not registered via SigBuilder: {name}"));
        let inst = self.fb.ins().call(func_ref, args);
        self.fb.inst_results(inst).to_vec()
    }
}
