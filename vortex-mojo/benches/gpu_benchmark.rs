// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;

fn main() {
    divan::main();
}

/// Benchmark for GPU frame_of_reference kernel.
#[divan::bench]
fn frame_of_reference_benchmark(bencher: Bencher) {
    bencher.bench_local(|| unsafe {
        vortex_mojo::mojo_frame_of_reference();
    });
}
