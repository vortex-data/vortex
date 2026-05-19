// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunked execution engine benchmark.
//!
//! Compares the new chunked decode path (small L1-resident scratch + fused kernels)
//! against the existing canonical-by-canonical executor on several shapes:
//!
//! - `Dict<Primitive>` — the simplest gather-from-small-dict workload.
//! - `RunEnd<Primitive>` — single-encoding streaming.
//! - `Dict<RunEnd<Primitive>>` — the fused stack: the dictionary's values are themselves
//!   RunEnd-encoded, so the chunked Dict kernel materializes the small RunEnd inner once
//!   and then streams the gather. The legacy path does roughly the same work but pays
//!   more allocation overhead in the executor.
//! - `ListView<Primitive>` (canonical + bit-packed offsets) — row-window streaming.
//!
//! Run with `cargo bench -p vortex-runend --bench chunked_exec`.

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::_chunked_exec::listview::build_listview_producer_typed;
use vortex_array::_chunked_exec::primitive::PrimitiveChunkKernelDispatcher;
use vortex_array::_chunked_exec::primitive::decode_to_buffer;
use vortex_array::_chunked_exec::primitive::default_dispatcher;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::BitPackedData;
use vortex_runend::RunEnd;
use vortex_runend::_chunked_exec::register_chunk_kernels as register_runend_chunk_kernels;
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
    register_runend_chunk_kernels(&mut d);
    vortex_fastlanes::_chunked_exec::register_chunk_kernels(&mut d);
    d
});

// ------------------------------------------------------------------------------------
// Dict<Primitive>
// ------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct DictArgs {
    len: usize,
    dict_size: usize,
}

impl fmt::Display for DictArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "len={} dict={}", self.len, self.dict_size)
    }
}

const DICT_ARGS: &[DictArgs] = &[
    DictArgs {
        len: 16_384,
        dict_size: 64,
    },
    DictArgs {
        len: 65_536,
        dict_size: 256,
    },
    DictArgs {
        len: 262_144,
        dict_size: 1024,
    },
    DictArgs {
        len: 1_048_576,
        dict_size: 256,
    },
    DictArgs {
        len: 1_048_576,
        dict_size: 4096,
    },
    // Cache-stress: codes buffer is 4*N bytes (canonical u32 codes).
    // Both paths have the same data flow here (no intermediate to save),
    // so chunked is predicted to stay tied across cache boundaries.
    DictArgs {
        len: 4_194_304, // codes 16 MB, output 16 MB
        dict_size: 256,
    },
    DictArgs {
        len: 16_777_216, // codes 64 MB, output 64 MB
        dict_size: 256,
    },
];

fn make_dict_i32(args: DictArgs) -> vortex_array::ArrayRef {
    let dict_size = args.dict_size;
    let len = args.len;
    let values: Vec<i32> = (0..dict_size as i32).map(|i| i * 17 + 11).collect();
    let codes: Vec<u32> = (0..len).map(|i| (i % dict_size) as u32).collect();
    let values = PrimitiveArray::new(Buffer::<i32>::from_iter(values), Validity::NonNullable);
    let codes = PrimitiveArray::new(Buffer::<u32>::from_iter(codes), Validity::NonNullable);
    DictArray::try_new(codes.into_array(), values.into_array())
        .expect("dict")
        .into_array()
}

#[divan::bench(args = DICT_ARGS)]
fn dict_primitive_chunked(bencher: Bencher, args: DictArgs) {
    let array = make_dict_i32(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        black_box(decode_to_buffer::<i32>(a.clone(), &DISPATCHER, &mut ctx).unwrap())
    });
}

#[divan::bench(args = DICT_ARGS)]
fn dict_primitive_canonical(bencher: Bencher, args: DictArgs) {
    let array = make_dict_i32(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        black_box(a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap())
    });
}

// ------------------------------------------------------------------------------------
// Dict<BitPacked<u16> codes> — the v2 bit-pack fusion case
// ------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct DictBpArgs {
    len: usize,
    dict_size: usize,
    bit_width: u8,
}

impl fmt::Display for DictBpArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "len={} dict={} bw={}", self.len, self.dict_size, self.bit_width)
    }
}

const DICT_BP_ARGS: &[DictBpArgs] = &[
    DictBpArgs {
        len: 65_536,
        dict_size: 256,
        bit_width: 8,
    },
    DictBpArgs {
        len: 262_144,
        dict_size: 256,
        bit_width: 8,
    },
    DictBpArgs {
        len: 1_048_576,
        dict_size: 256,
        bit_width: 8,
    },
    DictBpArgs {
        len: 1_048_576,
        dict_size: 1024,
        bit_width: 10,
    },
    DictBpArgs {
        len: 1_048_576,
        dict_size: 4096,
        bit_width: 12,
    },
    // Cache-stress shapes: intermediate codes Buffer<u16> = 2*N bytes.
    // L2 on the test host is 2 MiB. Below crosses that boundary progressively.
    DictBpArgs {
        len: 4_194_304, // 8 MiB intermediate (4× L2)
        dict_size: 256,
        bit_width: 8,
    },
    DictBpArgs {
        len: 16_777_216, // 32 MiB intermediate (16× L2, still in L3)
        dict_size: 256,
        bit_width: 8,
    },
    DictBpArgs {
        len: 67_108_864, // 128 MiB intermediate (64× L2, half of L3)
        dict_size: 256,
        bit_width: 8,
    },
];

fn make_dict_bp_i32(args: DictBpArgs) -> vortex_array::ArrayRef {
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

#[divan::bench(args = DICT_BP_ARGS)]
fn dict_bp_canonical(bencher: Bencher, args: DictBpArgs) {
    let array = make_dict_bp_i32(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        black_box(a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap())
    });
}

#[divan::bench(args = DICT_BP_ARGS)]
fn dict_bp_chunked(bencher: Bencher, args: DictBpArgs) {
    let array = make_dict_bp_i32(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        black_box(decode_to_buffer::<i32>(a.clone(), &DISPATCHER, &mut ctx).unwrap())
    });
}

// ------------------------------------------------------------------------------------
// RunEnd<Primitive>
// ------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct RunEndArgs {
    len: usize,
    avg_run_len: usize,
}

impl fmt::Display for RunEndArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "len={} run={}", self.len, self.avg_run_len)
    }
}

const RUNEND_ARGS: &[RunEndArgs] = &[
    RunEndArgs {
        len: 65_536,
        avg_run_len: 4,
    },
    RunEndArgs {
        len: 65_536,
        avg_run_len: 64,
    },
    RunEndArgs {
        len: 1_048_576,
        avg_run_len: 16,
    },
    RunEndArgs {
        len: 1_048_576,
        avg_run_len: 256,
    },
];

fn make_runend_i32(args: RunEndArgs) -> vortex_array::ArrayRef {
    let mut values = Vec::with_capacity(args.len);
    let mut run_idx = 0i32;
    let mut pos = 0;
    while pos < args.len {
        let run = args.avg_run_len.min(args.len - pos);
        values.extend(std::iter::repeat(run_idx % 1024).take(run));
        run_idx += 1;
        pos += run;
    }
    let prim = PrimitiveArray::new(Buffer::<i32>::from_iter(values), Validity::NonNullable);
    let mut ctx = SESSION.create_execution_ctx();
    RunEnd::encode(prim.into_array(), &mut ctx).unwrap().into_array()
}

#[divan::bench(args = RUNEND_ARGS)]
fn runend_chunked(bencher: Bencher, args: RunEndArgs) {
    let array = make_runend_i32(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        black_box(decode_to_buffer::<i32>(a.clone(), &DISPATCHER, &mut ctx).unwrap())
    });
}

#[divan::bench(args = RUNEND_ARGS)]
fn runend_canonical(bencher: Bencher, args: RunEndArgs) {
    let array = make_runend_i32(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        black_box(a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap())
    });
}

// ------------------------------------------------------------------------------------
// Dict<RunEnd<Primitive>> (fused)
// ------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct DictRunEndArgs {
    len: usize,
    dict_size: usize,
    /// Average run length *inside the dictionary's values*.
    inner_run_len: usize,
}

impl fmt::Display for DictRunEndArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "len={} dict={} inner_run={}",
            self.len, self.dict_size, self.inner_run_len
        )
    }
}

const DICT_RUNEND_ARGS: &[DictRunEndArgs] = &[
    DictRunEndArgs {
        len: 1_048_576,
        dict_size: 256,
        inner_run_len: 4,
    },
    DictRunEndArgs {
        len: 1_048_576,
        dict_size: 4096,
        inner_run_len: 16,
    },
    DictRunEndArgs {
        len: 4_194_304,
        dict_size: 1024,
        inner_run_len: 8,
    },
];

fn make_dict_runend_i32(args: DictRunEndArgs) -> vortex_array::ArrayRef {
    // Build the inner dictionary values (RunEnd-encoded).
    let mut inner_values = Vec::with_capacity(args.dict_size);
    let mut run_idx = 0i32;
    let mut pos = 0;
    while pos < args.dict_size {
        let run = args.inner_run_len.min(args.dict_size - pos);
        inner_values.extend(std::iter::repeat(run_idx).take(run));
        run_idx += 1;
        pos += run;
    }
    let inner_prim = PrimitiveArray::new(Buffer::<i32>::from_iter(inner_values), Validity::NonNullable);
    let mut ctx = SESSION.create_execution_ctx();
    let inner_re = RunEnd::encode(inner_prim.into_array(), &mut ctx).unwrap();
    let codes: Vec<u32> = (0..args.len as u32).map(|i| i % args.dict_size as u32).collect();
    let codes = PrimitiveArray::new(Buffer::<u32>::from_iter(codes), Validity::NonNullable);
    DictArray::try_new(codes.into_array(), inner_re.into_array())
        .unwrap()
        .into_array()
}

#[divan::bench(args = DICT_RUNEND_ARGS)]
fn dict_runend_fused_chunked(bencher: Bencher, args: DictRunEndArgs) {
    let array = make_dict_runend_i32(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        black_box(decode_to_buffer::<i32>(a.clone(), &DISPATCHER, &mut ctx).unwrap())
    });
}

#[divan::bench(args = DICT_RUNEND_ARGS)]
fn dict_runend_canonical(bencher: Bencher, args: DictRunEndArgs) {
    let array = make_dict_runend_i32(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        black_box(a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap())
    });
}

// Diagnostic: how slow is *just* canonicalizing the inner RunEnd dict (small)?
#[divan::bench(args = DICT_RUNEND_ARGS)]
fn dict_runend_phase_inner_canonical(bencher: Bencher, args: DictRunEndArgs) {
    let array = make_dict_runend_i32(args);
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::arrays::Dict;
    let inner = array.as_::<Dict>().values().clone();
    bencher.with_inputs(|| inner.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        black_box(a.clone().execute::<PrimitiveArray>(&mut ctx).unwrap())
    });
}

// Diagnostic: take_primitive over the already-materialized dict + 1M codes.
#[divan::bench(args = DICT_RUNEND_ARGS)]
fn dict_runend_phase_take(bencher: Bencher, args: DictRunEndArgs) {
    use vortex_array::arrays::dict::DictArraySlotsExt;
    use vortex_array::arrays::Dict;
    use vortex_array::builtins::ArrayBuiltins;
    let array = make_dict_runend_i32(args);
    let dict_view = array.as_::<Dict>();
    let codes = dict_view.codes().clone();
    let values = dict_view.values().clone();
    let mut ctx = SESSION.create_execution_ctx();
    let inner = values.execute::<PrimitiveArray>(&mut ctx).unwrap().into_array();
    bencher
        .with_inputs(|| (inner.clone(), codes.clone()))
        .bench_local_refs(|(inner, codes)| {
            black_box(inner.take(codes.clone()).unwrap())
        });
}

// ------------------------------------------------------------------------------------
// ListView<Primitive> with bit-packed offsets + sizes
// ------------------------------------------------------------------------------------

#[derive(Copy, Clone)]
struct ListViewArgs {
    rows: usize,
    avg_list_len: usize,
    bit_width_offsets: u8,
    bit_width_sizes: u8,
}

impl fmt::Display for ListViewArgs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "rows={} avg_list={} bw_off={} bw_sz={}",
            self.rows, self.avg_list_len, self.bit_width_offsets, self.bit_width_sizes
        )
    }
}

const LIST_ARGS: &[ListViewArgs] = &[
    ListViewArgs {
        rows: 16_384,
        avg_list_len: 8,
        bit_width_offsets: 18, // covers up to 16_384 * 8 == 131_072
        bit_width_sizes: 5,    // up to 31 elements per list
    },
    ListViewArgs {
        rows: 65_536,
        avg_list_len: 4,
        bit_width_offsets: 18,
        bit_width_sizes: 4,
    },
    ListViewArgs {
        rows: 262_144,
        avg_list_len: 4,
        bit_width_offsets: 20,
        bit_width_sizes: 4,
    },
];

/// Build a `ListView<i32>` where:
/// - `elements` is a plain i32 buffer with `rows * avg_list_len` values.
/// - `offsets` and `sizes` are bit-packed via fastlanes (so canonicalize must unpack).
fn make_listview_bp(args: ListViewArgs) -> vortex_array::ArrayRef {
    let n_elements = args.rows * args.avg_list_len;
    let elements: Vec<i32> = (0..n_elements as i32).collect();
    let offsets: Vec<u32> = (0..args.rows as u32).map(|i| i * args.avg_list_len as u32).collect();
    let sizes: Vec<u32> = vec![args.avg_list_len as u32; args.rows];

    let elements_arr = PrimitiveArray::new(Buffer::<i32>::from_iter(elements), Validity::NonNullable);
    let offsets_arr = PrimitiveArray::new(Buffer::<u32>::from_iter(offsets), Validity::NonNullable);
    let sizes_arr = PrimitiveArray::new(Buffer::<u32>::from_iter(sizes), Validity::NonNullable);

    let mut ctx = SESSION.create_execution_ctx();
    let bp_offsets = BitPackedData::encode(
        &offsets_arr.into_array(),
        args.bit_width_offsets,
        &mut ctx,
    )
    .expect("offsets bitpack")
    .into_array();
    let bp_sizes =
        BitPackedData::encode(&sizes_arr.into_array(), args.bit_width_sizes, &mut ctx)
            .expect("sizes bitpack")
            .into_array();

    ListViewArray::new(
        elements_arr.into_array(),
        bp_offsets,
        bp_sizes,
        Validity::NonNullable,
    )
    .into_array()
}

/// Walk the chunked rows summing elements end-to-end. Uses the typed-callback API so
/// there is no dyn dispatch in the inner loop.
#[divan::bench(args = LIST_ARGS)]
fn listview_chunked_sum(bencher: Bencher, args: ListViewArgs) {
    let array = make_listview_bp(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        let mut producer =
            build_listview_producer_typed::<u32, u32, i32>(a.clone(), &mut ctx).unwrap();
        let mut sum: i64 = 0;
        producer.for_each_chunk_typed(|offs, szs, elems| {
            for i in 0..offs.len() {
                let o = offs[i] as usize;
                let s = szs[i] as usize;
                for &v in &elems[o..o + s] {
                    sum = sum.wrapping_add(v as i64);
                }
            }
        });
        black_box(sum)
    });
}

/// Canonicalize the whole `ListView` then sum elements; the apples-to-apples baseline.
#[divan::bench(args = LIST_ARGS)]
fn listview_canonical_sum(bencher: Bencher, args: ListViewArgs) {
    use vortex_array::Canonical;
    use vortex_array::arrays::ListView;
    use vortex_array::arrays::listview::ListViewArrayExt;
    use vortex_array::arrays::primitive::PrimitiveArrayExt;
    use vortex_array::dtype::NativePType;

    let array = make_listview_bp(args);
    bencher.with_inputs(|| array.clone()).bench_local_refs(|a| {
        let mut ctx = SESSION.create_execution_ctx();
        let canonical = a.clone().execute::<Canonical>(&mut ctx).unwrap();
        // Canonical is a ListView; sum the relevant element slice via offsets/sizes.
        let lv = canonical.into_array();
        let view = lv.as_::<ListView>();
        let offsets = view
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let sizes = view
            .sizes()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let elements = view
            .elements()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        let elements = elements.as_slice::<i32>();
        let off = offsets.as_slice::<u32>();
        let sz = sizes.as_slice::<u32>();
        let mut sum: i64 = 0;
        for i in 0..off.len() {
            let o = off[i] as usize;
            let s = sz[i] as usize;
            for &v in &elements[o..o + s] {
                sum = sum.wrapping_add(v as i64);
            }
        }
        // Silence unused if NativePType isn't otherwise referenced.
        let _: i32 = i32::PTYPE.bit_width() as i32;
        black_box(sum)
    });
}
