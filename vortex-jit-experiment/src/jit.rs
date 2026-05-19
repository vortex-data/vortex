// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cranelift JIT that synthesises the fused unpack+compare kernel.
//!
//! The point of this module is to show, in real code, what a JIT crossing
//! the kernel boundary actually has to do:
//!
//! 1. Take a *kernel spec* (here: just `bit_width`, plus the implicit
//!    "unpack-then-compare-greater-than-k" template) and produce a single
//!    function that does both jobs.
//! 2. Specialise on the runtime parameter so that `bit_width`, the
//!    word-straddling load pattern, and the mask constant become
//!    `iconst` operands that Cranelift folds into addressing math.
//! 3. Hand the caller a function pointer with a stable C ABI so it can
//!    be called from anywhere (including from inside a hot vx executor
//!    loop, or from FFI).
//!
//! The generated function has signature, in Rust terms:
//!
//! ```ignore
//! extern "C" fn(packed: *const u32, k: u32, mask: *mut u64)
//! ```
//!
//! ## What's specialised, what isn't
//!
//! - **`bit_width`**: hard-coded as `iconst.i64`. This is the JIT-time
//!   parameter. Cranelift constant-folds `i * bit_width`, the mask
//!   `(1 << bit_width) - 1`, etc.
//! - **`CHUNK_SIZE` (1024)**: also a constant. Could be runtime, but
//!   matching the rest of vx makes the demo realistic.
//! - **`k` (threshold)**: kept as a runtime argument. In practice you'd
//!   *also* JIT-specialise on `k` if you knew it stayed constant across
//!   many chunks — that lets the comparison be eliminated entirely when
//!   `k >= (1 << bit_width) - 1` (always false) or `k == 0` (always
//!   true unless value is zero).
//!
//! ## Memory safety contract on the produced fn
//!
//! - `packed` must point to **at least `n_value_words + 2`** valid `u32`
//!   words, where `n_value_words = ceil(1024 * bit_width / 32)`. The
//!   two-word padding lets the inner loop do an unconditional unaligned
//!   8-byte load that straddles the value's word boundary without
//!   running off the buffer on the very last element.
//! - `mask` must point to 16 valid, writable `u64` words.

use cranelift::codegen::ir::AbiParam;
use cranelift::codegen::ir::Function;
use cranelift::codegen::ir::InstBuilder;
use cranelift::codegen::ir::MemFlags;
use cranelift::codegen::ir::UserFuncName;
use cranelift::codegen::ir::types::I8;
use cranelift::codegen::ir::types::I32;
use cranelift::codegen::ir::types::I64;
use cranelift::codegen::isa::CallConv;
use cranelift::codegen::settings;
use cranelift::codegen::settings::Configurable;
use cranelift::frontend::FunctionBuilder;
use cranelift::frontend::FunctionBuilderContext;
use cranelift::prelude::IntCC;
use cranelift_jit::JITBuilder;
use cranelift_jit::JITModule;
use cranelift_module::Linkage;
use cranelift_module::Module;

use crate::CHUNK_SIZE;
use crate::MASK_WORDS;

/// The C-ABI function pointer that the JIT hands back.
pub type FusedFn = unsafe extern "C" fn(packed: *const u32, k: u32, mask: *mut u64);

/// Owns the JIT module so the executable memory stays alive while
/// callers hold function pointers into it.
pub struct CompiledKernel {
    _module: JITModule,
    func: FusedFn,
    ir: String,
}

impl CompiledKernel {
    /// The Cranelift IR for the generated function, as text. Captured
    /// before lowering so we can show what the JIT actually emitted at
    /// the IR level.
    pub fn ir(&self) -> &str {
        &self.ir
    }

    /// Run the JIT'd kernel.
    ///
    /// # Safety
    /// - `packed.len() * 4 >= n_value_words + 2` words. See module docs.
    /// - `mask.len() == 16`.
    pub unsafe fn run(&self, packed: &[u32], k: u32, mask: &mut [u64; MASK_WORDS]) {
        // SAFETY: the contract on the FusedFn is documented above; caller
        // upholds it via the typed wrapper signature.
        unsafe { (self.func)(packed.as_ptr(), k, mask.as_mut_ptr()) }
    }
}

/// Build and compile a fused unpack+compare-greater-than kernel
/// specialised to `bit_width`.
pub fn compile(bit_width: u32) -> Result<CompiledKernel, String> {
    assert!((1..=31).contains(&bit_width), "bit_width must be in 1..=31",);

    // ---------------------------------------------------------------
    // 1. ISA + JIT module setup.
    // ---------------------------------------------------------------
    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| e.to_string())?;
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| e.to_string())?;
    // is_pic = false: we generate code into JIT memory, no relocations.
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| e.to_string())?;
    let isa_builder =
        cranelift_native::builder().map_err(|e| format!("host machine not supported: {e}"))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| e.to_string())?;

    let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(jit_builder);
    let pointer_type = module.target_config().pointer_type();

    // ---------------------------------------------------------------
    // 2. Function signature: (ptr, u32, ptr) -> ()
    // ---------------------------------------------------------------
    let mut sig = module.make_signature();
    sig.call_conv = CallConv::SystemV; // matches `extern "C"` on Linux/macOS-x86_64 + Linux-aarch64
    sig.params.push(AbiParam::new(pointer_type));
    sig.params.push(AbiParam::new(I32));
    sig.params.push(AbiParam::new(pointer_type));

    let func_id = module
        .declare_function("unpack_compare_fused", Linkage::Local, &sig)
        .map_err(|e| e.to_string())?;

    // ---------------------------------------------------------------
    // 3. Build the function body.
    // ---------------------------------------------------------------
    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
    let mut builder_ctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut func, &mut builder_ctx);

    let entry = b.create_block();
    let outer_hdr = b.create_block();
    let inner_hdr = b.create_block();
    let inner_body = b.create_block();
    let inner_exit = b.create_block();
    let exit = b.create_block();

    // Loop SSA: outer carries `word`, inner carries `word, bit, bits`.
    b.append_block_params_for_function_params(entry);
    b.append_block_param(outer_hdr, I64); // word
    b.append_block_param(inner_hdr, I64); // word
    b.append_block_param(inner_hdr, I64); // bit
    b.append_block_param(inner_hdr, I64); // bits-accumulator
    b.append_block_param(inner_exit, I64); // word
    b.append_block_param(inner_exit, I64); // final bits

    // -- entry: pull out function args, jump to outer header with word=0
    b.switch_to_block(entry);
    let packed_ptr = b.block_params(entry)[0];
    let k_arg = b.block_params(entry)[1];
    let mask_ptr = b.block_params(entry)[2];
    let zero_i64 = b.ins().iconst(I64, 0);
    b.ins().jump(outer_hdr, &[zero_i64.into()]);
    b.seal_block(entry);

    // -- outer header: if word >= MASK_WORDS, exit; else enter inner loop.
    b.switch_to_block(outer_hdr);
    let word = b.block_params(outer_hdr)[0];
    let mask_words_const = b.ins().iconst(I64, MASK_WORDS as i64);
    let outer_done = b
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, word, mask_words_const);
    let zero_bits = b.ins().iconst(I64, 0);
    let zero_bit_idx = b.ins().iconst(I64, 0);
    b.ins().brif(
        outer_done,
        exit,
        &[],
        inner_hdr,
        &[word.into(), zero_bit_idx.into(), zero_bits.into()],
    );

    // -- inner header: if bit >= 64, store and increment word; else inner_body.
    b.switch_to_block(inner_hdr);
    let inner_word = b.block_params(inner_hdr)[0];
    let inner_bit = b.block_params(inner_hdr)[1];
    let inner_bits = b.block_params(inner_hdr)[2];
    let bits_per_word = b.ins().iconst(I64, 64);
    let inner_done = b
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, inner_bit, bits_per_word);
    b.ins().brif(
        inner_done,
        inner_exit,
        &[inner_word.into(), inner_bits.into()],
        inner_body,
        &[],
    );

    // -- inner body: unpack one value, compare, OR bit into accumulator.
    b.switch_to_block(inner_body);
    // i = word * 64 + bit
    let i_val = {
        let shifted = b.ins().ishl_imm(inner_word, 6); // word * 64
        b.ins().iadd(shifted, inner_bit)
    };
    // bit_off = i * bit_width   (folded: bit_width is constant)
    let bit_off = b.ins().imul_imm(i_val, bit_width as i64);
    // word_off = bit_off >> 5
    let word_off = b.ins().ushr_imm(bit_off, 5);
    // shift = bit_off & 31
    let shift = b.ins().band_imm(bit_off, 31);
    // addr = packed_ptr + word_off * 4
    let byte_off = b.ins().ishl_imm(word_off, 2);
    let load_addr = b.ins().iadd(packed_ptr, byte_off);
    // Unaligned 8-byte load straddling the value's word boundary.
    let load_flags = MemFlags::new(); // potentially-unaligned, no aliasing assumptions
    let data = b.ins().load(I64, load_flags, load_addr, 0);
    // v64 = (data >> shift) & mask
    let shifted = b.ins().ushr(data, shift);
    let mask_const = if bit_width == 32 {
        u32::MAX as i64
    } else {
        (1i64 << bit_width) - 1
    };
    let v64 = b.ins().band_imm(shifted, mask_const);
    // Compare against k (zero-extended).
    let k64 = b.ins().uextend(I64, k_arg);
    let gt = b.ins().icmp(IntCC::UnsignedGreaterThan, v64, k64);
    // Widen the i8 boolean and OR into the accumulator at position `bit`.
    let gt64 = b.ins().uextend(I64, gt);
    // Bool comes back as I8 from icmp; uextend to I64 lands either 0 or 1.
    let _ = I8; // silence unused-import warning until we wire something I8-typed.
    let gt_shifted = b.ins().ishl(gt64, inner_bit);
    let new_bits = b.ins().bor(inner_bits, gt_shifted);
    let next_bit = b.ins().iadd_imm(inner_bit, 1);
    b.ins().jump(
        inner_hdr,
        &[inner_word.into(), next_bit.into(), new_bits.into()],
    );

    // -- inner exit: store mask[word] = bits, increment word, back to outer.
    b.switch_to_block(inner_exit);
    let store_word = b.block_params(inner_exit)[0];
    let final_bits = b.block_params(inner_exit)[1];
    let mask_byte_off = b.ins().ishl_imm(store_word, 3);
    let store_addr = b.ins().iadd(mask_ptr, mask_byte_off);
    let store_flags = MemFlags::new().with_aligned(); // [u64; 16] is 8-byte aligned
    b.ins().store(store_flags, final_bits, store_addr, 0);
    let next_word = b.ins().iadd_imm(store_word, 1);
    b.ins().jump(outer_hdr, &[next_word.into()]);

    // -- exit: return.
    b.switch_to_block(exit);
    b.ins().return_(&[]);

    b.seal_block(outer_hdr);
    b.seal_block(inner_hdr);
    b.seal_block(inner_body);
    b.seal_block(inner_exit);
    b.seal_block(exit);

    b.finalize();

    // Capture IR before lowering to native code.
    let ir = func.display().to_string();

    // ---------------------------------------------------------------
    // 4. Compile and link.
    // ---------------------------------------------------------------
    let mut ctx = module.make_context();
    ctx.func = func;
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define_function: {e}"))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| format!("finalize_definitions: {e}"))?;
    let code_ptr = module.get_finalized_function(func_id);
    // SAFETY: the JIT hands us a fn ptr matching the declared signature,
    // and CompiledKernel keeps `_module` alive so the code memory stays mapped.
    let func: FusedFn = unsafe { std::mem::transmute(code_ptr) };

    Ok(CompiledKernel {
        _module: module,
        func,
        ir,
    })
}

/// Suggested caller-side padding: callers must ensure `packed` has at
/// least this many trailing pad words beyond the values' real footprint.
pub const REQUIRED_PAD_WORDS: usize = 2;

/// Number of value words for `CHUNK_SIZE` elements at `bit_width`.
pub fn n_value_words(bit_width: u32) -> usize {
    (CHUNK_SIZE * bit_width as usize).div_ceil(32)
}
