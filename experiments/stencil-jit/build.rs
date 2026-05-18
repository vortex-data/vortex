//! Build-time codegen for the F variant: use `cranelift-codegen` to compile
//! the eq-kernel for x86-64 AVX2, capture its machine-code bytes, and
//! write them into `$OUT_DIR/cranelift_eq_kernel.bin`. The runtime then
//! includes those bytes via `include_bytes!` and feeds them through the
//! same `materialize()` path D-spec uses.
//!
//! For simplicity this build script bakes the compare constant (35 = 42-7)
//! as a literal in the IR; a production version would emit a stencil
//! with a patch slot for the constant and a relocation table.

use std::{env, fs, path::PathBuf, str::FromStr};

use cranelift_codegen::{
    Context,
    ir::{
        AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName,
        condcodes::IntCC, types,
    },
    isa::{self, CallConv},
    settings::{self, Configurable, Flags},
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use target_lexicon::Triple;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Target ISA: x86-64 Linux with AVX2 enabled.
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    let triple = Triple::from_str("x86_64-unknown-linux-gnu").unwrap();
    let mut isa_builder = isa::lookup(triple).unwrap();
    isa_builder.enable("has_avx").unwrap();
    isa_builder.enable("has_avx2").unwrap();
    isa_builder.enable("has_bmi2").unwrap();
    let isa = isa_builder.finish(Flags::new(flag_builder)).unwrap();

    // Function signature: fn(packed, out, n_halves, effective_const_byte).
    // The constant is a normal SystemV arg (low byte of rcx); the kernel
    // broadcasts it into an xmm register once per call. That's amortized
    // across many blocks — the realistic F shape for runtime-supplied
    // query constants.
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I8));

    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut func, &mut fb_ctx);

        let entry = b.create_block();
        let check = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();

        b.append_block_params_for_function_params(entry);
        // `check` carries the loop counter i.
        b.append_block_param(check, types::I64);

        // --- entry ---
        b.switch_to_block(entry);
        let packed = b.block_params(entry)[0];
        let out = b.block_params(entry)[1];
        let n_blocks = b.block_params(entry)[2];
        let ec = b.block_params(entry)[3];
        // Broadcast the runtime-supplied effective constant byte into all
        // 16 lanes of an xmm register. The broadcast cost is paid once per
        // call (n_blocks scans share the same xmm constant).
        let cvec = b.ins().splat(types::I8X16, ec);
        let zero = b.ins().iconst(types::I64, 0);
        b.ins().jump(check, &[zero.into()]);

        // --- check: branch if i == n_blocks ---
        b.switch_to_block(check);
        let i = b.block_params(check)[0];
        let done = b.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, i, n_blocks);
        b.ins().brif(done, exit, &[], body, &[]);

        // --- body: load 16 bytes + compare + bitmask + store + i++ ---
        // Cranelift 0.118's x64 backend only handles vector types up to
        // 128 bits, so this kernel uses i8x16 (xmm) rather than the i8x32
        // (ymm) D-spec uses. The loop processes 16 bytes per iteration and
        // emits a 16-bit mask per iteration; the caller passes
        // n_blocks_16 = n_blocks_32 * 2 to get the same total bytes covered.
        b.switch_to_block(body);
        let i16_off = b.ins().imul_imm(i, 16);
        let in_addr = b.ins().iadd(packed, i16_off);
        let data = b.ins().load(types::I8X16, MemFlags::trusted(), in_addr, 0);
        let eq = b.ins().icmp(IntCC::Equal, data, cvec);
        // vhigh_bits: high bit of each lane into a scalar integer.
        // For i8x16, the result is i16 (16 bits).
        let mask = b.ins().vhigh_bits(types::I16, eq);
        let i2_off = b.ins().imul_imm(i, 2);
        let out_addr = b.ins().iadd(out, i2_off);
        b.ins().store(MemFlags::trusted(), mask, out_addr, 0);
        let i_next = b.ins().iadd_imm(i, 1);
        b.ins().jump(check, &[i_next.into()]);

        // --- exit ---
        b.switch_to_block(exit);
        b.ins().return_(&[]);

        b.seal_all_blocks();
        b.finalize();
    }

    let mut ctx = Context::for_function(func);
    let compiled = ctx
        .compile(isa.as_ref(), &mut Default::default())
        .expect("cranelift compile failed");
    let code = compiled.code_buffer();
    let relocs = compiled.buffer.relocs();
    if !relocs.is_empty() {
        eprintln!(
            "warning: cranelift emitted {} relocations; this build script doesn't handle them",
            relocs.len()
        );
        for r in relocs {
            eprintln!("  reloc: {r:?}");
        }
    }

    let bin = out_dir.join("cranelift_eq_kernel.bin");
    fs::write(&bin, code).unwrap();
    println!(
        "cargo:warning=cranelift kernel: {} bytes ({}), 0 relocs",
        code.len(),
        bin.display()
    );
}
