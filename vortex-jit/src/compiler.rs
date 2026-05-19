// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cranelift JIT compiler driver.

use std::collections::HashMap;

use cranelift::prelude::{
    AbiParam, Configurable, FunctionBuilder, FunctionBuilderContext, InstBuilder, IntCC,
    Signature, Type as ClType, isa::CallConv, settings,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use vortex_error::{VortexResult, vortex_err};

use crate::emit::{ArgKey, EmitCtx, ExternId, Lanes, SigBuilder};
use crate::pipeline::Pipeline;

/// Operation the compiled kernel performs.
///
/// v0 only implements `Decompress`. `Filter` / `Take` would change the
/// terminal stage and possibly the driver; framework hooks left for §9.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelOp {
    Decompress,
}

/// One Rust extern callable from JITted code. Address must remain valid for
/// the lifetime of the `Compiler`.
#[derive(Debug, Clone, Copy)]
pub struct ExternFn {
    pub name: &'static str,
    pub addr: *const u8,
    pub params: &'static [ClType],
    pub returns: &'static [ClType],
}

// SAFETY: `addr` points to a `'static`-lifetime function in the host.
unsafe impl Send for ExternFn {}
unsafe impl Sync for ExternFn {}

/// A compiled kernel function pointer + its signature shape.
pub struct Compiled {
    pub raw_fn: *const u8,
    pub args: Vec<ArgKey>,
    pub ir_dump: String,
    _module: JITModule,
}

// SAFETY: the JITModule owns the executable code; Compiled is the only owner.
unsafe impl Send for Compiled {}

impl Compiled {
    /// v0 Decompress signature: `(in_ptr, out_ptr, n_blocks, ...named args...)`.
    /// All args are pointer-typed except `n_blocks` (i64).
    ///
    /// # Safety
    /// Caller must guarantee:
    ///   - `in_ptr` points to at least `n_blocks * block_size * in_ptype.byte_width()` bytes
    ///   - `out_ptr` points to at least `n_blocks * block_size * out_ptype.byte_width()` bytes
    ///   - Every named arg pointer is valid for the access pattern its stage emits
    pub unsafe fn call_decompress_only(
        &self,
        in_ptr: *const u8,
        out_ptr: *mut u8,
        n_blocks: u64,
    ) {
        debug_assert_eq!(self.args.len(), 3);
        debug_assert_eq!(self.args[0], ArgKey::InPtr);
        debug_assert_eq!(self.args[1], ArgKey::OutPtr);
        debug_assert_eq!(self.args[2], ArgKey::NBlocks);
        let f: unsafe extern "C" fn(*const u8, *mut u8, u64) =
            unsafe { std::mem::transmute(self.raw_fn) };
        unsafe { f(in_ptr, out_ptr, n_blocks) };
    }

    /// Call a 4-arg kernel: in, out, n_blocks, + one named pointer.
    ///
    /// # Safety
    /// Same as `call_decompress_only`, plus `named` must satisfy the named
    /// stage's access requirements.
    pub unsafe fn call_with_named(
        &self,
        in_ptr: *const u8,
        out_ptr: *mut u8,
        n_blocks: u64,
        named: *const u8,
    ) {
        debug_assert_eq!(self.args.len(), 4);
        let f: unsafe extern "C" fn(*const u8, *mut u8, u64, *const u8) =
            unsafe { std::mem::transmute(self.raw_fn) };
        unsafe { f(in_ptr, out_ptr, n_blocks, named) };
    }

    /// Call a kernel that wants in/out/n_blocks plus three named pointers
    /// (e.g. patches: indices, values, count).
    ///
    /// # Safety
    /// See `call_decompress_only`.
    pub unsafe fn call_with_three_named(
        &self,
        in_ptr: *const u8,
        out_ptr: *mut u8,
        n_blocks: u64,
        a: *const u8,
        b: *const u8,
        c: *const u8,
    ) {
        debug_assert_eq!(self.args.len(), 6);
        let f: unsafe extern "C" fn(*const u8, *mut u8, u64, *const u8, *const u8, *const u8) =
            unsafe { std::mem::transmute(self.raw_fn) };
        unsafe { f(in_ptr, out_ptr, n_blocks, a, b, c) };
    }
}

/// Driver for compiling pipelines into native code.
pub struct Compiler {
    module: Option<JITModule>,
    externs: Vec<ExternFn>,
}

impl Compiler {
    pub fn new(externs: Vec<ExternFn>) -> VortexResult<Self> {
        let mut flag_builder = settings::builder();
        // Crank Cranelift's optimizer; the framework's IR shape benefits from
        // LICM (hoisting the broadcast splat) and alias-aware scheduling.
        for (k, v) in [
            ("use_colocated_libcalls", "false"),
            ("is_pic", "false"),
            ("opt_level", "speed"),
            ("enable_alias_analysis", "true"),
            ("enable_verifier", "false"),
            ("preserve_frame_pointers", "false"),
        ] {
            flag_builder
                .set(k, v)
                .map_err(|e| vortex_err!("cranelift flag {k}={v}: {e}"))?;
        }
        let isa_builder = cranelift_native::builder()
            .map_err(|e| vortex_err!("cranelift native target: {e}"))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| vortex_err!("cranelift isa: {e}"))?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        for ext in &externs {
            builder.symbol(ext.name, ext.addr.cast::<u8>().cast_mut());
        }
        let module = JITModule::new(builder);
        Ok(Self {
            module: Some(module),
            externs,
        })
    }

    pub fn compile(mut self, pipeline: &Pipeline) -> VortexResult<Compiled> {
        let mut module = self.module.take().expect("module taken");
        let pt = module.target_config().pointer_type();

        // -- Pass 1: walk all stages once to collect runtime args and externs.
        let mut sig_builder = SigBuilder::default();
        // Driver always needs NBlocks; framework declares it implicitly.
        sig_builder.request_arg(ArgKey::NBlocks);
        for s in pipeline.in_block_stages() {
            s.declare(&mut sig_builder);
        }
        for s in pipeline.post_loop_stages() {
            s.declare(&mut sig_builder);
        }

        // Stable arg order: InPtr, OutPtr, NBlocks, then any Named in
        // declaration order. (We move InPtr/OutPtr/NBlocks to the front
        // because that's the call-site convention.)
        let mut ordered_args: Vec<ArgKey> = Vec::new();
        for fixed in [ArgKey::InPtr, ArgKey::OutPtr, ArgKey::NBlocks] {
            if sig_builder.args.contains(&fixed) {
                ordered_args.push(fixed);
            }
        }
        for a in &sig_builder.args {
            if !matches!(a, ArgKey::InPtr | ArgKey::OutPtr | ArgKey::NBlocks)
                && !ordered_args.contains(a)
            {
                ordered_args.push(a.clone());
            }
        }

        // -- Pass 2: declare extern Rust helpers in the module.
        let mut extern_fn_ids: HashMap<ExternId, cranelift_module::FuncId> = HashMap::new();
        for (idx, name) in sig_builder.externs.iter().enumerate() {
            let ext = self
                .externs
                .iter()
                .find(|e| e.name == *name)
                .ok_or_else(|| vortex_err!("extern {} not registered with Compiler", name))?;
            let mut ext_sig = module.make_signature();
            for p in ext.params {
                ext_sig.params.push(AbiParam::new(*p));
            }
            for r in ext.returns {
                ext_sig.returns.push(AbiParam::new(*r));
            }
            let func_id = module
                .declare_function(ext.name, Linkage::Import, &ext_sig)
                .map_err(|e| vortex_err!("declare extern {}: {e}", ext.name))?;
            extern_fn_ids.insert(ExternId(idx as u32), func_id);
        }

        // -- Pass 3: build the main kernel function.
        let mut ctx = module.make_context();
        let mut sig = Signature::new(CallConv::SystemV);
        for a in &ordered_args {
            let ty = match a {
                ArgKey::NBlocks => cranelift::prelude::types::I64,
                _ => pt,
            };
            sig.params.push(AbiParam::new(ty));
        }
        ctx.func.signature = sig;

        let func_id = module
            .declare_function("kernel", Linkage::Local, &ctx.func.signature)
            .map_err(|e| vortex_err!("declare kernel: {e}"))?;

        let mut fbc = FunctionBuilderContext::new();
        {
            let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbc);
            let entry = fb.create_block();
            fb.append_block_params_for_function_params(entry);
            fb.switch_to_block(entry);
            fb.seal_block(entry);

            // Materialize a HashMap<ArgKey, Value>.
            let entry_params: Vec<_> = fb.block_params(entry).to_vec();
            let arg_values: HashMap<ArgKey, _> = ordered_args
                .iter()
                .cloned()
                .zip(entry_params)
                .collect();

            // Import externs into this function so EmitCtx can call them.
            // We key by extern *name* — the SigBuilder hands ExternIds back
            // to internal book-keeping, but stages reference externs by
            // their stable Rust symbol name.
            let extern_refs_by_name: HashMap<&'static str, _> = sig_builder
                .externs
                .iter()
                .enumerate()
                .map(|(idx, name)| {
                    let fid = extern_fn_ids[&ExternId(idx as u32)];
                    let r = module.declare_func_in_func(fid, fb.func);
                    (*name, r)
                })
                .collect();

            // -- Block loop header --
            let n_blocks = arg_values
                .get(&ArgKey::NBlocks)
                .copied()
                .ok_or_else(|| vortex_err!("kernel needs NBlocks"))?;

            let loop_hdr = fb.create_block();
            let loop_body = fb.create_block();
            let loop_exit = fb.create_block();
            fb.append_block_param(loop_hdr, cranelift::prelude::types::I64);

            let zero = fb.ins().iconst(cranelift::prelude::types::I64, 0);
            fb.ins().jump(loop_hdr, &[zero.into()]);

            fb.switch_to_block(loop_hdr);
            let i = fb.block_params(loop_hdr)[0];
            let cond = fb.ins().icmp(IntCC::UnsignedLessThan, i, n_blocks);
            fb.ins().brif(cond, loop_body, &[], loop_exit, &[]);

            fb.switch_to_block(loop_body);
            // -- Emit in-block stages, threading Lanes through.
            let mut current: Lanes = Lanes::None;
            for stage in pipeline.in_block_stages() {
                let mut ecx = EmitCtx {
                    fb: &mut fb,
                    args: &arg_values,
                    externs_by_name: &extern_refs_by_name,
                    input: current,
                    output: None,
                    block_idx: i,
                    chunk_count: pipeline.block_size(),
                    module_pt: pt,
                };
                stage.emit(&mut ecx)?;
                current = ecx.output.unwrap_or(Lanes::None);
            }

            let one = fb.ins().iconst(cranelift::prelude::types::I64, 1);
            let next = fb.ins().iadd(i, one);
            fb.ins().jump(loop_hdr, &[next.into()]);
            fb.seal_block(loop_body);
            fb.seal_block(loop_hdr);

            // -- Post-loop stages --
            fb.switch_to_block(loop_exit);
            for stage in pipeline.post_loop_stages() {
                let zero_idx = fb.ins().iconst(cranelift::prelude::types::I64, 0);
                let mut ecx = EmitCtx {
                    fb: &mut fb,
                    args: &arg_values,
                    externs_by_name: &extern_refs_by_name,
                    input: Lanes::None,
                    output: None,
                    block_idx: zero_idx,
                    chunk_count: pipeline.block_size(),
                    module_pt: pt,
                };
                stage.emit(&mut ecx)?;
            }
            fb.ins().return_(&[]);
            fb.seal_block(loop_exit);

            fb.finalize();
        }

        // Stash the textual IR for debugging / verification before consuming.
        let ir_dump = ctx.func.display().to_string();

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| vortex_err!("define_function: {e}"))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| vortex_err!("finalize_definitions: {e}"))?;

        let raw_fn = module.get_finalized_function(func_id);

        Ok(Compiled {
            raw_fn,
            args: ordered_args,
            ir_dump,
            _module: module,
        })
    }
}
