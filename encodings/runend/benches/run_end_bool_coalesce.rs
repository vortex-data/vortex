// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare decoding comparison-result runs with materializing or directly consuming
//! merged boolean ranges. Input construction and correctness checks are not timed.
//!
//! Inputs model non-nullable comparison results with canonical u32 ends. The
//! comparison itself is not timed. `precoalesced_decode` excludes coalescing and
//! is only a reference; `coalesce_then_decode` includes intermediate allocations.

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_runend::decompress_bool::runend_decode_bools;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    session
});

fn main() {
    divan::main();
}

#[derive(Clone, Copy)]
struct Case {
    length: usize,
    run_length: usize,
    pattern: &'static str,
    offset: usize,
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n{}_run{}_{}_offset{}",
            self.length, self.run_length, self.pattern, self.offset
        )
    }
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for length in [8192, 1_048_576] {
        for run_length in [2, 16, 256] {
            for pattern in [
                "alternating",
                "random50",
                "random10",
                "groups8",
                "groups64",
                "sorted",
                "all_false",
                "all_true",
            ] {
                cases.push(Case {
                    length,
                    run_length,
                    pattern,
                    offset: 0,
                });
            }
        }
    }
    for pattern in ["random50", "groups64", "sorted"] {
        cases.push(Case {
            length: 8192,
            run_length: 16,
            pattern,
            offset: 7,
        });
    }
    cases
}

struct Input {
    ends: PrimitiveArray,
    values: BoolArray,
    expected: BoolArray,
}

fn input(case: Case) -> Input {
    let mut ends = BufferMut::<u32>::empty();
    let mut values = BitBufferMut::empty();
    let mut expected = BitBufferMut::empty();
    let mut state = 0x1234_5678_usize;
    let mut end = 0;
    let mut index = 0usize;
    while end < case.length + case.offset {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = match case.pattern {
            "alternating" => index.is_multiple_of(2),
            "random50" => state % 100 < 50,
            "random10" => state % 100 < 10,
            "groups8" => (index / 8).is_multiple_of(2),
            "groups64" => (index / 64).is_multiple_of(2),
            "sorted" => end >= (case.length + case.offset) / 2,
            "all_false" => false,
            "all_true" => true,
            _ => unreachable!(),
        };
        // Irregular lengths also exercise non-byte-aligned output boundaries.
        let run_length = (case.run_length / 2 + state % case.run_length).max(1);
        let next = (end + run_length).min(case.length + case.offset);
        expected.append_n(value, next - end);
        ends.push(u32::try_from(next).vortex_expect("benchmark ends fit u32"));
        values.append(value);
        end = next;
        index += 1;
    }
    Input {
        ends: PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        values: BoolArray::from(values.freeze()),
        expected: BoolArray::from(expected.freeze().slice(case.offset..)),
    }
}

// BitSliceIterator skips whole words and reports maximal true ranges. The gaps
// are maximal false ranges, so no per-bit scan or redundant end reads are needed.
fn coalesce(ends: &PrimitiveArray, values: &BoolArray) -> (PrimitiveArray, BoolArray) {
    let ends = ends.as_slice::<u32>();
    let bits = values.to_bit_buffer();
    let mut merged_ends = BufferMut::<u32>::empty();
    let mut merged_values = BitBufferMut::empty();
    let mut previous = 0;
    for (start, stop) in bits.set_slices() {
        if start > previous {
            merged_ends.push(ends[start - 1]);
            merged_values.append(false);
        }
        merged_ends.push(ends[stop - 1]);
        merged_values.append(true);
        previous = stop;
    }
    if previous < ends.len() {
        merged_ends.push(ends[ends.len() - 1]);
        merged_values.append(false);
    }
    (
        PrimitiveArray::new(merged_ends.freeze(), Validity::NonNullable),
        BoolArray::from(merged_values.freeze()),
    )
}

// Consume the same merged true ranges without materializing an intermediate REE.
fn fused_ranges(ends: &PrimitiveArray, values: &BoolArray, case: Case) -> ArrayRef {
    let ends = ends.as_slice::<u32>();
    let bits = values.to_bit_buffer();
    let mut decoded = BitBufferMut::full(false, case.length);
    for (start, stop) in bits.set_slices() {
        let start = if start == 0 {
            0
        } else {
            ends[start - 1] as usize
        };
        let stop = ends[stop - 1] as usize;
        let start = start.saturating_sub(case.offset).min(case.length);
        let stop = stop.saturating_sub(case.offset).min(case.length);
        decoded.fill_range(start, stop, true);
    }
    BoolArray::from(decoded.freeze()).into_array()
}

#[derive(Clone, Copy)]
enum Method {
    Raw,
    CoalesceThenDecode,
    PrecoalescedDecode,
    FusedRanges,
}

fn decode(input: &Input, case: Case, ctx: &mut ExecutionCtx) -> ArrayRef {
    runend_decode_bools(
        input.ends.clone(),
        input.values.clone(),
        case.offset,
        case.length,
        ctx,
    )
    .vortex_expect("decode benchmark input")
}

fn bench(bencher: Bencher, case: Case, method: Method) {
    let mut input = input(case);
    let mut ctx = SESSION.create_execution_ctx();
    let (ends, values) = coalesce(&input.ends, &input.values);
    let merged = runend_decode_bools(
        ends.clone(),
        values.clone(),
        case.offset,
        case.length,
        &mut ctx,
    )
    .vortex_expect("decode coalesced input");
    assert_arrays_eq!(decode(&input, case, &mut ctx), input.expected, &mut ctx);
    assert_arrays_eq!(merged, input.expected, &mut ctx);
    assert_arrays_eq!(
        fused_ranges(&input.ends, &input.values, case),
        input.expected,
        &mut ctx
    );
    if matches!(method, Method::PrecoalescedDecode) {
        input.ends = ends;
        input.values = values;
    }
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            let input = divan::black_box(&input);
            match method {
                Method::Raw | Method::PrecoalescedDecode => decode(input, case, ctx),
                Method::CoalesceThenDecode => {
                    let (ends, values) = coalesce(&input.ends, &input.values);
                    runend_decode_bools(ends, values, case.offset, case.length, ctx)
                        .vortex_expect("decode coalesced benchmark input")
                }
                Method::FusedRanges => fused_ranges(&input.ends, &input.values, case),
            }
        });
}

#[divan::bench(args = cases())]
fn raw(bencher: Bencher, case: Case) {
    bench(bencher, case, Method::Raw);
}

#[divan::bench(args = cases())]
fn coalesce_then_decode(bencher: Bencher, case: Case) {
    bench(bencher, case, Method::CoalesceThenDecode);
}

#[divan::bench(args = cases())]
fn precoalesced_decode(bencher: Bencher, case: Case) {
    bench(bencher, case, Method::PrecoalescedDecode);
}

#[divan::bench(args = cases())]
fn fused(bencher: Bencher, case: Case) {
    bench(bencher, case, Method::FusedRanges);
}
