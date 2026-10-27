// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare an already-packed `BitPackedArray` against a constant value. Compares the
//! out-of-range fast path (constant outside `[0, 2^bit_width - 1]`) against an explicit
//! "decompress, then compare" baseline.
//!
//! Sized to finish quickly. Run with `cargo bench -p vortex-fastlanes --bench bitpack_compare`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedData;
use vortex_fastlanes::BitWidths;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    session
});

const LEN: usize = 64 * 1024;
const BW: u8 = 16;

/// Rebuild the array with its packed buffer copied to a page-aligned allocation.
///
/// `bitpack_encode` aligns the packed buffer only to the element type, so its cache-line
/// placement depends on allocator state, which shifts with any change to the bench binary.
/// CodSpeed's simulated cache misses are deterministic in those addresses, which made these
/// benches flip ~30% between two layout modes across unrelated commits. Pinning the buffer to a
/// page boundary makes the layout, and therefore the measurement, reproducible.
fn page_aligned(array: BitPackedArray) -> BitPackedArray {
    let ptype = array.dtype().as_ptype();
    let parts = BitPacked::into_parts(array);
    let BitWidths::Global(bit_width) = parts.bit_widths else {
        unreachable!("bitpack_encode packs every block at one bit width")
    };
    BitPacked::try_new(
        parts.packed.ensure_aligned(Alignment::new(4096)).unwrap(),
        ptype,
        parts.validity,
        parts.patches,
        bit_width,
        parts.len,
        parts.offset,
    )
    .unwrap()
}

/// Build a packed array of varied in-range values, plus an out-of-range constant RHS for
/// the fast-path benches.
fn build_inputs() -> (ArrayRef, ArrayRef, ExecutionCtx) {
    let mut ctx = SESSION.create_execution_ctx();
    let buf: BufferMut<u32> = (0..LEN).map(|i| (i as u32) % (1 << BW)).collect();
    let array = page_aligned(
        BitPackedData::encode(
            &PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array(),
            BW,
            &mut ctx,
        )
        .unwrap(),
    )
    .into_array();
    // 1 << BW is just past the packable range, so the out-of-range fast path fires.
    let constant = 1u32 << BW;
    let rhs = ConstantArray::new(constant, LEN).into_array();
    (array, rhs, ctx)
}

#[divan::bench]
fn fast_eq_out_of_range(bencher: Bencher) {
    let (array, rhs, mut ctx) = build_inputs();
    bencher.bench_local(|| {
        array
            .clone()
            .binary(rhs.clone(), Operator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench]
fn baseline_eq(bencher: Bencher) {
    let (array, rhs, mut ctx) = build_inputs();
    bencher.bench_local(|| {
        // What the fallback would do: materialize the unpacked primitive, then run Arrow
        // compare on it.
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        primitive
            .into_array()
            .binary(rhs.clone(), Operator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench]
fn fast_lt_out_of_range(bencher: Bencher) {
    let (array, rhs, mut ctx) = build_inputs();
    bencher.bench_local(|| {
        array
            .clone()
            .binary(rhs.clone(), Operator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench]
fn baseline_lt(bencher: Bencher) {
    let (array, rhs, mut ctx) = build_inputs();
    bencher.bench_local(|| {
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        primitive
            .into_array()
            .binary(rhs.clone(), Operator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}
