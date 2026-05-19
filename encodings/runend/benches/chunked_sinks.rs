// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sink-fusion benchmark for the chunked execution engine.
//!
//! Compares the fused-pipeline path (decode + operator in one chunked pass, no
//! intermediate `Buffer<T>`) against the canonical two-pass equivalent
//! (`array.execute::<PrimitiveArray>(ctx)?` followed by a scalar loop over the materialised
//! buffer) for three common scalar operator shapes:
//!
//! - `filter(x > c)` on `Dict<BitPacked<u16>>`
//! - `cast(i32 → i64)` on `Dict<i32>`
//! - `scalar add` (`x + c`) on `Dict<i32>`
//!
//! Run with `cargo bench -p vortex-runend --bench chunked_sinks`.

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::_chunked_exec::primitive::PrimitiveChunkKernelDispatcher;
use vortex_array::_chunked_exec::primitive::default_dispatcher;
use vortex_array::_chunked_exec::sink::FilterSink;
use vortex_array::_chunked_exec::sink::MapSink;
use vortex_array::_chunked_exec::sink::drive_into_sink;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPackedData;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let s = VortexSession::empty().with::<ArraySession>();
    vortex_runend::initialize(&s);
    vortex_fastlanes::initialize(&s);
    s
});

static DISPATCHER: LazyLock<PrimitiveChunkKernelDispatcher> = LazyLock::new(|| {
    let mut d = default_dispatcher();
    vortex_runend::_chunked_exec::register_chunk_kernels(&mut d);
    vortex_fastlanes::_chunked_exec::register_chunk_kernels(&mut d);
    d
});

// ============================================================================
// Filter — Dict<BitPacked<u16>> with selectivity ~50%
// ============================================================================

#[derive(Copy, Clone)]
struct FilterArgs {
    len: usize,
    dict_size: usize,
    bit_width: u8,
    threshold: i32,
}

impl fmt::Display for FilterArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "len={} dict={} bw={} pred=x>{}",
            self.len, self.dict_size, self.bit_width, self.threshold
        )
    }
}

const FILTER_ARGS: &[FilterArgs] = &[
    FilterArgs {
        len: 1_048_576,
        dict_size: 256,
        bit_width: 8,
        threshold: 2000,
    },
    FilterArgs {
        len: 4_194_304,
        dict_size: 256,
        bit_width: 8,
        threshold: 2000,
    },
    FilterArgs {
        len: 16_777_216,
        dict_size: 256,
        bit_width: 8,
        threshold: 2000,
    },
];

/// Build `Dict<BitPacked<u16>>` with dict values `[0, 17, 34, …]` so threshold=2000
/// gives ~50% selectivity (passes if dict[code] > 2000 → roughly half the dict).
fn make_filter_input(args: FilterArgs) -> vortex_array::ArrayRef {
    let dict_values: Vec<i32> = (0..args.dict_size as i32).map(|i| i * 17 + 11).collect();
    let codes: Vec<u16> = (0..args.len)
        .map(|i| (i % args.dict_size) as u16)
        .collect();
    let dict = PrimitiveArray::new(
        Buffer::<i32>::from_iter(dict_values),
        Validity::NonNullable,
    );
    let codes_prim = PrimitiveArray::new(Buffer::<u16>::from_iter(codes), Validity::NonNullable);
    let mut ctx = SESSION.create_execution_ctx();
    let bp = BitPackedData::encode(&codes_prim.into_array(), args.bit_width, &mut ctx)
        .expect("bitpack");
    DictArray::try_new(bp.into_array(), dict.into_array())
        .expect("dict")
        .into_array()
}

#[divan::bench(args = FILTER_ARGS)]
fn filter_canonical_two_pass(bencher: Bencher, args: FilterArgs) {
    let array = make_filter_input(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        // Pass 1: decode the full array into a Buffer<i32>.
        let prim = a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        let slice = prim.as_slice::<i32>();
        // Pass 2: walk the buffer, collecting survivors into a new buffer.
        let mut out = BufferMut::<i32>::with_capacity(slice.len());
        for &v in slice {
            if v > args.threshold {
                out.push(v);
            }
        }
        black_box(out.freeze())
    });
}

#[divan::bench(args = FILTER_ARGS)]
fn filter_chunked_sink(bencher: Bencher, args: FilterArgs) {
    let array = make_filter_input(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        let threshold = args.threshold;
        let sink = FilterSink::<i32, _>::with_capacity(args.len, move |v| v > threshold);
        let buf: Buffer<i32> = drive_into_sink(a.clone(), &DISPATCHER, sink, &mut ctx).unwrap();
        black_box(buf)
    });
}

// ============================================================================
// Cast — Dict<i32> → i64
// ============================================================================

#[derive(Copy, Clone)]
struct CastArgs {
    len: usize,
    dict_size: usize,
}

impl fmt::Display for CastArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "len={} dict={}", self.len, self.dict_size)
    }
}

const CAST_ARGS: &[CastArgs] = &[
    CastArgs {
        len: 1_048_576,
        dict_size: 256,
    },
    CastArgs {
        len: 4_194_304,
        dict_size: 256,
    },
    CastArgs {
        len: 16_777_216,
        dict_size: 256,
    },
];

fn make_cast_input(args: CastArgs) -> vortex_array::ArrayRef {
    let dict_values: Vec<i32> = (0..args.dict_size as i32).map(|i| i * 7 + 13).collect();
    let codes: Vec<u32> = (0..args.len).map(|i| (i % args.dict_size) as u32).collect();
    let dict = PrimitiveArray::new(
        Buffer::<i32>::from_iter(dict_values),
        Validity::NonNullable,
    );
    let codes_prim = PrimitiveArray::new(Buffer::<u32>::from_iter(codes), Validity::NonNullable);
    DictArray::try_new(codes_prim.into_array(), dict.into_array())
        .expect("dict")
        .into_array()
}

#[divan::bench(args = CAST_ARGS)]
fn cast_canonical_two_pass(bencher: Bencher, args: CastArgs) {
    let array = make_cast_input(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        let slice = prim.as_slice::<i32>();
        let mut out = BufferMut::<i64>::with_capacity(slice.len());
        for &v in slice {
            out.push(v as i64);
        }
        black_box(out.freeze())
    });
}

#[divan::bench(args = CAST_ARGS)]
fn cast_chunked_sink(bencher: Bencher, args: CastArgs) {
    let array = make_cast_input(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        let sink = MapSink::<i32, i64, _>::with_capacity(args.len, |v| v as i64);
        let buf: Buffer<i64> = drive_into_sink(a.clone(), &DISPATCHER, sink, &mut ctx).unwrap();
        black_box(buf)
    });
}

// ============================================================================
// Scalar add — Dict<i32>, output i32, `x + 42`
// ============================================================================

#[divan::bench(args = CAST_ARGS)]
fn scalar_add_canonical_two_pass(bencher: Bencher, args: CastArgs) {
    let array = make_cast_input(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        let slice = prim.as_slice::<i32>();
        let mut out = BufferMut::<i32>::with_capacity(slice.len());
        for &v in slice {
            out.push(v.wrapping_add(42));
        }
        black_box(out.freeze())
    });
}

#[divan::bench(args = CAST_ARGS)]
fn scalar_add_chunked_sink(bencher: Bencher, args: CastArgs) {
    let array = make_cast_input(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        let sink =
            MapSink::<i32, i32, _>::with_capacity(args.len, |v| v.wrapping_add(42));
        let buf: Buffer<i32> = drive_into_sink(a.clone(), &DISPATCHER, sink, &mut ctx).unwrap();
        black_box(buf)
    });
}

// ============================================================================
// Scalar mul + add (richer scalar pipeline: `x * 3 + 7`)
// ============================================================================

#[divan::bench(args = CAST_ARGS)]
fn scalar_mul_add_canonical_two_pass(bencher: Bencher, args: CastArgs) {
    let array = make_cast_input(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        let slice = prim.as_slice::<i32>();
        let mut out = BufferMut::<i32>::with_capacity(slice.len());
        for &v in slice {
            out.push(v.wrapping_mul(3).wrapping_add(7));
        }
        black_box(out.freeze())
    });
}

#[divan::bench(args = CAST_ARGS)]
fn scalar_mul_add_chunked_sink(bencher: Bencher, args: CastArgs) {
    let array = make_cast_input(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        let sink = MapSink::<i32, i32, _>::with_capacity(args.len, |v| {
            v.wrapping_mul(3).wrapping_add(7)
        });
        let buf: Buffer<i32> = drive_into_sink(a.clone(), &DISPATCHER, sink, &mut ctx).unwrap();
        black_box(buf)
    });
}
