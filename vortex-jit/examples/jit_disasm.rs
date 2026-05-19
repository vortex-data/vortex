// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extract the JIT-compiled ALP kernel bytes and dump them to a file, so we
//! can disassemble with system objdump and prove what Cranelift emitted.
//!
//! Run:
//!   cargo run --release -p vortex-jit --example jit_disasm
//!
//! Then:
//!   objdump -D -b binary -m i386:x86-64 -M intel \
//!     --no-show-raw-insn /tmp/jit_alp.bin | head -80

use std::sync::Arc;

use vortex_jit::stages::{AlpDecode, LoadIn, StoreOut};
use vortex_jit::{Compiler, PType, Pipeline};

const BLOCK: usize = 16;

fn main() {
    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(AlpDecode {
        in_ptype: PType::I32,
        out_ptype: PType::F32,
        scale: 0.01,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::F32 })).unwrap();

    let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();

    // Read the first 4 KiB of the function. Cranelift emits a `ret` at the
    // end; the rest may be padding or constant pool. Disasm will show us
    // where the real function ends.
    let n = 4096;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(compiled.raw_fn, n) };

    std::fs::write("/tmp/jit_alp.bin", bytes).expect("write");

    println!("wrote {} bytes to /tmp/jit_alp.bin", n);
    println!("kernel fn address: {:p}", compiled.raw_fn);
    println!();
    println!("Run:");
    println!(
        "  objdump -D -b binary -m i386:x86-64 -M intel --no-show-raw-insn /tmp/jit_alp.bin | head -80"
    );
}
