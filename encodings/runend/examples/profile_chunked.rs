// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Standalone profile binary for the chunked execution engine.
//!
//! Builds with `cargo build --release --example profile_chunked -p vortex-runend`.
//! Designed to be sampled by `samply record` or `perf record` — runs each path in a
//! tight loop with no benchmark-framework overhead, so the profile shows the
//! decompress kernels directly.
//!
//! Usage:
//!   samply record ./target/release/examples/profile_chunked dict_bp 4194304 256 8 chunked 50
//!   samply record ./target/release/examples/profile_chunked dict_bp 4194304 256 8 canonical 50

use std::env;
use std::sync::Arc;
use std::time::Instant;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::_chunked_exec::primitive::decode_to_buffer;
use vortex_array::_chunked_exec::primitive::default_dispatcher;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::BitPackedData;
use vortex_runend::_chunked_exec::register_chunk_kernels as register_runend_chunk_kernels;
use vortex_session::VortexSession;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "usage: {} <shape> <len> <dict_size> <bit_width> <path> <iters>",
            args[0]
        );
        eprintln!("  shape: dict_bp | dict_p");
        eprintln!("  path: chunked | canonical");
        std::process::exit(2);
    }
    let shape = &args[1];
    let len: usize = args[2].parse().unwrap();
    let dict_size: usize = args[3].parse().unwrap();
    let bit_width: u8 = args[4].parse().unwrap();
    let path = &args[5];
    let iters: usize = args.get(6).map(|s| s.parse().unwrap()).unwrap_or(20);

    let session = {
        let s = VortexSession::empty().with::<ArraySession>();
        vortex_runend::initialize(&s);
        vortex_fastlanes::initialize(&s);
        s
    };
    let dispatcher = {
        let mut d = default_dispatcher();
        register_runend_chunk_kernels(&mut d);
        vortex_fastlanes::_chunked_exec::register_chunk_kernels(&mut d);
        Arc::new(d)
    };

    let array = match shape.as_str() {
        "dict_bp" => build_dict_bp(&session, len, dict_size, bit_width),
        "dict_p" => build_dict_p(len, dict_size),
        _ => panic!("unknown shape {shape}"),
    };

    // Warm caches.
    for _ in 0..3 {
        match path.as_str() {
            "chunked" => {
                let mut ctx = session.create_execution_ctx();
                let _w = decode_to_buffer::<i32>(array.clone(), &dispatcher, &mut ctx).unwrap();
                std::hint::black_box(_w);
            }
            "canonical" => {
                let mut ctx = session.create_execution_ctx();
                let _w = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
                std::hint::black_box(_w);
            }
            _ => panic!("unknown path {path}"),
        }
    }

    // Time the loop body that the profiler will sample inside.
    let t0 = Instant::now();
    for _ in 0..iters {
        match path.as_str() {
            "chunked" => {
                let mut ctx = session.create_execution_ctx();
                let b = decode_to_buffer::<i32>(array.clone(), &dispatcher, &mut ctx).unwrap();
                std::hint::black_box(b);
            }
            "canonical" => {
                let mut ctx = session.create_execution_ctx();
                let b = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
                std::hint::black_box(b);
            }
            _ => unreachable!(),
        }
    }
    let elapsed = t0.elapsed();
    let per_iter_us = (elapsed.as_micros() as f64) / (iters as f64);
    let elems_per_us = (len as f64) / per_iter_us;
    println!(
        "{} {} N={} dict={} bw={} iters={}  per_iter={:.1}µs  {:.1}M elems/sec",
        shape,
        path,
        len,
        dict_size,
        bit_width,
        iters,
        per_iter_us,
        elems_per_us
    );
}

fn build_dict_bp(
    session: &VortexSession,
    len: usize,
    dict_size: usize,
    bit_width: u8,
) -> vortex_array::ArrayRef {
    let dict_values: Vec<i32> = (0..dict_size as i32).map(|i| i * 17 + 11).collect();
    let codes: Vec<u16> = (0..len).map(|i| (i % dict_size) as u16).collect();
    let dict = PrimitiveArray::new(
        Buffer::<i32>::from_iter(dict_values),
        Validity::NonNullable,
    );
    let codes_prim =
        PrimitiveArray::new(Buffer::<u16>::from_iter(codes), Validity::NonNullable);
    let mut ctx = session.create_execution_ctx();
    let bp = BitPackedData::encode(&codes_prim.into_array(), bit_width, &mut ctx).unwrap();
    DictArray::try_new(bp.into_array(), dict.into_array())
        .unwrap()
        .into_array()
}

fn build_dict_p(len: usize, dict_size: usize) -> vortex_array::ArrayRef {
    let dict_values: Vec<i32> = (0..dict_size as i32).map(|i| i * 17 + 11).collect();
    let codes: Vec<u32> = (0..len).map(|i| (i % dict_size) as u32).collect();
    let dict = PrimitiveArray::new(
        Buffer::<i32>::from_iter(dict_values),
        Validity::NonNullable,
    );
    let codes_prim =
        PrimitiveArray::new(Buffer::<u32>::from_iter(codes), Validity::NonNullable);
    DictArray::try_new(codes_prim.into_array(), dict.into_array())
        .unwrap()
        .into_array()
}
