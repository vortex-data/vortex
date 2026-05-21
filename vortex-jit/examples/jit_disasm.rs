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

use vortex_jit::stages::{AlpDecode, BitPackedLoad, ForAdd, LoadIn, StoreOut};
use vortex_jit::{Compiler, PType, Pipeline};

const BLOCK: usize = 64;

fn dump_alp_only() {
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
    let n = 4096;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(compiled.raw_fn, n) };
    std::fs::write("/tmp/jit_alp.bin", bytes).expect("write");
    println!("ALP only:  /tmp/jit_alp.bin  fn @ {:p}", compiled.raw_fn);
}

fn dump_chain() {
    let mut p = Pipeline::new(PType::I32, 128);
    p.push(Arc::new(BitPackedLoad {
        ptype: PType::I32,
        bit_width: 11,
    }))
    .unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: 100,
    }))
    .unwrap();
    p.push(Arc::new(AlpDecode {
        in_ptype: PType::I32,
        out_ptype: PType::F32,
        scale: 0.01,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::F32 })).unwrap();
    let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
    let n = 16 * 1024;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(compiled.raw_fn, n) };
    std::fs::write("/tmp/jit_chain.bin", bytes).expect("write");
    println!("4-stage chain: /tmp/jit_chain.bin  fn @ {:p}", compiled.raw_fn);
}

fn dump_foralp(ptype: PType, ftype: PType, path: &str, label: &str) {
    let mut p = Pipeline::new(ptype, BLOCK);
    p.push(Arc::new(LoadIn { ptype })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype,
        reference: 100,
    }))
    .unwrap();
    p.push(Arc::new(AlpDecode {
        in_ptype: ptype,
        out_ptype: ftype,
        scale: 0.01,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: ftype })).unwrap();
    let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
    let n = 4096;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(compiled.raw_fn, n) };
    std::fs::write(path, bytes).expect("write");
    println!("{label}: {path}  fn @ {:p}", compiled.raw_fn);
}

fn main() {
    dump_alp_only();
    dump_chain();
    dump_foralp(PType::I32, PType::F32, "/tmp/jit_foralp_u32.bin", "FoR+ALP u32->f32");
    dump_foralp(PType::I64, PType::F64, "/tmp/jit_foralp_u64.bin", "FoR+ALP u64->f64");
}
