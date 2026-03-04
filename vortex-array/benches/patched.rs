// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use divan::counter::BytesCount;
use divan::counter::ItemsCount;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::arrays::PatchedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::patches::Patches;
use vortex_buffer::buffer;

fn main() {
    divan::main()
}

#[divan::bench(args = [1, 10, 100, 1024, 2048, 65_536])]
fn bench_patch_transpose(bencher: Bencher, n_patches: usize) {
    const N: u32 = 1024 * 512;
    let numbers = PrimitiveArray::from_iter(0u32..N).into_array();

    let patch_indices =
        PrimitiveArray::from_iter((0..N).step_by(N as usize / n_patches)).into_array();
    let patch_values = buffer![u32::MAX; patch_indices.len()].into_array();

    let patches = Patches::new(N as usize, 0, patch_indices, patch_values, None).unwrap();

    let mut ctx = ExecutionCtx::new(LEGACY_SESSION.clone());

    bencher.counter(ItemsCount::new(n_patches)).bench_local(|| {
        PatchedArray::from_array_and_patches(numbers.clone(), &patches, &mut ctx).unwrap()
    });
}
