// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks decoding a `BitPacked` array whose 1024-value blocks each have their own bit width.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use fastlanes::BitPacking;
use mimalloc::MiMalloc;
use num_traits::AsPrimitive;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::bitpack_compress::bitpack_primitive;
use vortex_fastlanes::bitpack_decompress::unpack_array;
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

/// `u32` unpacks full blocks straight into the output, while multi-block `u64` arrays unpack
/// through scratch.
#[vortex_bench_support::cpu_features]
#[divan::bench(types = [u32, u64])]
fn bitpack_blocked_decompress<T>(bencher: Bencher)
where
    T: NativePType + BitPacking,
    u64: AsPrimitive<T>,
{
    // Block widths cycle from 1 to 16 bits, so the blocks take different unpacking kernels.
    let mut packed = BufferMut::<T>::with_capacity(NUM_BLOCKS * FL_CHUNK_SIZE);
    let mut offsets = vec![0u32];
    for bit_width in (1..=16u8).cycle().take(NUM_BLOCKS) {
        let block: Vec<T> = (0..1024u64)
            .map(|i| (i.wrapping_mul(7919) & ((1 << bit_width) - 1)).as_())
            .collect();
        packed.extend_from_slice(&bitpack_primitive(&block, bit_width));
        offsets.push(u32::try_from(packed.len() * size_of::<T>()).unwrap());
    }
    let array = BitPacked::try_new_with_block_offsets(
        BufferHandle::new_host(packed.freeze().into_byte_buffer()),
        T::PTYPE,
        Validity::NonNullable,
        None,
        PrimitiveArray::from_iter(offsets).into_array(),
        NUM_BLOCKS * FL_CHUNK_SIZE,
        0,
    )
    .unwrap();

    bencher.counter(ItemsCount::new(array.len())).bench(|| {
        unpack_array(array.as_view(), &mut SESSION.create_execution_ctx()).unwrap()
    });
}
