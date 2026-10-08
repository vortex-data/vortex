// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks filtering `FixedSizeList<i32>` arrays, where the list-level selection is expanded
//! into a selection of elements.

#![expect(clippy::cast_possible_truncation, clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const NUM_ELEMENTS: usize = 1 << 17;

/// List size 2 covers the plain-bitmap expansion with many short runs, and 16 covers the
/// cached-slices expansion.
#[divan::bench(consts = [2, 16], args = [0.01, 0.5])]
fn fsl_i32<const LIST_SIZE: u32>(bencher: Bencher, density: f64) {
    // Short lists produce many more runs per element and dense selections copy far more
    // elements, so both use smaller arrays to keep each iteration short.
    let mut num_elements = NUM_ELEMENTS;
    if LIST_SIZE == 2 {
        num_elements /= 2;
    }
    if density > 0.1 {
        num_elements /= 8;
    }
    let len = num_elements / LIST_SIZE as usize;
    let elements = PrimitiveArray::from_iter(0..(len * LIST_SIZE as usize) as i32).into_array();
    let array =
        FixedSizeListArray::new(elements, LIST_SIZE, Validity::NonNullable, len).into_array();

    let mut rng = StdRng::seed_from_u64(0);
    let bits = BitBuffer::from_iter((0..len).map(|_| rng.random_bool(density)));

    // Build a fresh mask for every iteration so that representations cached on the mask are not
    // reused across iterations.
    bencher
        .with_inputs(|| {
            (
                &array,
                Mask::from_buffer(bits.clone()),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_values(|(array, mask, mut ctx)| {
            array
                .filter(mask)
                .unwrap()
                .execute::<RecursiveCanonical>(&mut ctx)
                .unwrap()
        });
}
