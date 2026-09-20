// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar reads out of a PCO array. Cases are `(access_count, nullable, scattered)`; clustered
//! indices stay inside one page so a retained decode can be reused, scattered ones cross pages.

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_pco::Pco;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const LEN: usize = 16_384;
const CASES: &[(usize, bool, bool)] = &[
    (1, false, false),
    (1024, false, false),
    (1024, true, false),
    (1024, false, true),
];

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

fn pco(nullable: bool) -> ArrayRef {
    let validity = if nullable {
        Validity::from_iter((0..LEN).map(|i| i % 11 != 0))
    } else {
        Validity::NonNullable
    };
    let input = PrimitiveArray::new(
        (0..LEN)
            .map(|i| u32::try_from(i / 16).vortex_expect("fixture values fit u32"))
            .collect::<Vec<_>>(),
        validity,
    );
    let mut ctx = SESSION.create_execution_ctx();
    Pco::from_primitive(input.as_view(), 8, 1024, &mut ctx)
        .vortex_expect("PCO compression")
        .into_array()
}

fn indices(count: usize, scattered: bool) -> Vec<usize> {
    let span = if scattered { LEN } else { 256 };
    let base = if scattered { 0 } else { 4096 };
    let mut seed = 42u64;
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            base + ((seed >> 32) as usize % span)
        })
        .collect()
}

#[divan::bench(args = CASES)]
fn scalar_access(bencher: Bencher, (count, nullable, scattered): (usize, bool, bool)) {
    let array = pco(nullable);
    let indices = indices(count, scattered);
    bencher
        .with_inputs(|| (SESSION.create_execution_ctx(), Vec::with_capacity(count)))
        .bench_refs(|(ctx, scalars)| {
            for &index in &indices {
                scalars.push(
                    array
                        .execute_scalar(index, ctx)
                        .vortex_expect("scalar access"),
                );
            }
        });
}
