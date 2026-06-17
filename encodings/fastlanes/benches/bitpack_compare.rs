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
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedData;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const LENS: &[usize] = &[1024, 64 * 1024];
const BIT_WIDTHS: &[u8] = &[4, 16];

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
    BitPacked::try_new(
        parts.packed.ensure_aligned(Alignment::new(4096)).unwrap(),
        ptype,
        parts.validity,
        parts.patches,
        parts.bit_width,
        parts.len,
        parts.offset,
    )
    .unwrap()
}

/// Build a packed array of varied in-range values, plus an out-of-range constant RHS for
/// the fast-path benches.
fn build_inputs<const BW: u8>(len: usize) -> (ArrayRef, ArrayRef, ExecutionCtx) {
    let mut ctx = SESSION.create_execution_ctx();
    let buf: BufferMut<u32> = (0..len).map(|i| (i as u32) % (1 << BW)).collect();
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
    let rhs = ConstantArray::new(constant, len).into_array();
    (array, rhs, ctx)
}

#[divan::bench(args = LENS, consts = BIT_WIDTHS)]
fn fast_eq_out_of_range<const BW: u8>(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs::<BW>(len);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        array
            .clone()
            .binary(rhs.clone(), Operator::Eq)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS, consts = BIT_WIDTHS)]
fn baseline_eq<const BW: u8>(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs::<BW>(len);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
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

#[divan::bench(args = LENS, consts = BIT_WIDTHS)]
fn fast_lt_out_of_range<const BW: u8>(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs::<BW>(len);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        array
            .clone()
            .binary(rhs.clone(), Operator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}

#[divan::bench(args = LENS, consts = BIT_WIDTHS)]
fn baseline_lt<const BW: u8>(bencher: Bencher, len: usize) {
    let (array, rhs, mut ctx) = build_inputs::<BW>(len);
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
        primitive
            .into_array()
            .binary(rhs.clone(), Operator::Lt)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap()
    });
}
