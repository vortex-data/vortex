// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding a `BitPacked` array whose 1024-value blocks each have their own bit width.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use num_traits::AsPrimitive;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_fastlanes::BitPackedArraySlotsExt;
use vortex_fastlanes::bitpack_compress::bitpack_encode_blocked;
use vortex_fastlanes::bitpack_decompress::unpack_array_blocked;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

/// 64 blocks per array: shorter iterations are noisy on the walltime legs.
const NUM_BLOCKS: usize = 64;

#[vortex_bench_support::cpu_features]
#[divan::bench(types = [u32, u64])]
fn bitpack_blocked_decompress<T>(bencher: Bencher)
where
    T: NativePType,
    u64: AsPrimitive<T>,
{
    // Block widths cycle from 1 to 16 bits, so the blocks take different unpacking kernels.
    let bit_widths: Vec<u8> = (1..=16).cycle().take(NUM_BLOCKS).collect();
    let values = PrimitiveArray::from_iter(bit_widths.iter().flat_map(|&bit_width| {
        (0..1024u64)
            .map(move |i| AsPrimitive::<T>::as_(i.wrapping_mul(7919) & ((1 << bit_width) - 1)))
    }));
    let array = bitpack_encode_blocked(
        &values,
        &bit_widths,
        Some(0),
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();
    let offsets = array.block_offsets().unwrap().clone();

    bencher.counter(ItemsCount::new(array.len())).bench(|| {
        unpack_array_blocked(
            array.as_view(),
            &offsets,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
    });
}
