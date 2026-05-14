// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare the fast constant bit-packing path against the standard `bitpack_encode`
//! pipeline on a uniform-constant input.
//!
//! Sized to finish quickly. Run with `cargo bench -p vortex-fastlanes --bench bitpack_constant`.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::black_box;
use divan::counter::ItemsCount;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_fastlanes::bitpack_compress::bitpack_encode_constant;

fn main() {
    divan::main();
}

const LENS: &[usize] = &[1024, 64 * 1024];
const BIT_WIDTHS: &[u8] = &[4, 16];

const CONSTANT: u32 = 7;

#[divan::bench(args = LENS, consts = BIT_WIDTHS)]
fn full_encode<const BW: u8>(bencher: Bencher, len: usize) {
    let buf: BufferMut<u32> = (0..len).map(|_| CONSTANT).collect();
    let arr = PrimitiveArray::new(buf.freeze(), Validity::NonNullable);
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    bencher
        .counter(ItemsCount::new(len))
        .bench_local(|| bitpack_encode(black_box(&arr), black_box(BW), None, &mut ctx).unwrap());
}

#[divan::bench(args = LENS, consts = BIT_WIDTHS)]
fn fast_encode<const BW: u8>(bencher: Bencher, len: usize) {
    bencher.counter(ItemsCount::new(len)).bench_local(|| {
        bitpack_encode_constant::<u32>(
            black_box(CONSTANT),
            black_box(BW),
            black_box(len),
            Validity::NonNullable,
        )
        .unwrap()
    });
}
