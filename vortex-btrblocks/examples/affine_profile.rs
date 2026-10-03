// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Break down where sliced Affine decode time goes on one real column.
//!
//! `cargo run --release -p vortex-btrblocks --example affine_profile -- <file.i64>`

#![expect(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::schemes::integer::AffineScheme;
use vortex_buffer::Buffer;
use vortex_fastlanes::Affine;
use vortex_fastlanes::AffineArraySlotsExt;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});
static AFFINE: AffineScheme = AffineScheme::auto();
const SLICE: usize = 65_536;

fn sliced(array: &ArrayRef) -> Duration {
    let mut ctx = SESSION.create_execution_ctx();
    (0..7)
        .map(|_| {
            let start = Instant::now();
            for begin in (0..array.len()).step_by(SLICE) {
                let part = array.slice(begin..(begin + SLICE).min(array.len())).unwrap();
                black_box(part.execute::<PrimitiveArray>(&mut ctx).unwrap());
            }
            start.elapsed()
        })
        .min()
        .unwrap()
}

fn report(name: &str, array: &ArrayRef, n: usize) {
    let t = sliced(array);
    println!(
        "{name:<40} {:>7.3} ns/value {:>8.0} M values/s",
        t.as_secs_f64() * 1e9 / n as f64,
        n as f64 / t.as_secs_f64() / 1e6
    );
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: affine_profile <file.i64>");
    let bytes = std::fs::read(path).unwrap();
    let values: Buffer<i64> = bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n = values.len();
    let input = PrimitiveArray::new(values, Validity::NonNullable).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let base = || BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted();

    let default = base().build().compress(&input, &mut ctx).unwrap();
    let affine = base()
        .with_new_scheme(&AFFINE)
        .build()
        .compress(&input, &mut ctx)
        .unwrap();
    println!("default: {}", default.display_tree());
    println!("affine:  {}", affine.display_tree());

    let part = affine.slice(65_536..131_072).unwrap();
    println!("affine slice: {}", part.display_tree());
    let view = part.as_opt::<Affine>().expect("affine slice");
    println!(
        "slice encoded child is {} (BitPacked: {})",
        view.encoded().encoding_id(),
        view.encoded().as_opt::<vortex_fastlanes::BitPacked>().is_some()
    );
    report("default, whole tree", &default, n);
    report("affine, whole tree", &affine, n);
    let view = affine.as_opt::<Affine>().expect("affine at the top");
    report("affine encoded child only", view.encoded(), n);
    for (name, child) in [
        ("references", view.references()),
        ("scales", view.scales()),
        ("slopes", view.slopes()),
    ] {
        let mut ctx = SESSION.create_execution_ctx();
        let start = Instant::now();
        for _ in 0..100 {
            black_box(child.clone().execute::<PrimitiveArray>(&mut ctx).unwrap());
        }
        println!(
            "{name:<40} full decode {:>8.1} µs ({} chunks)",
            start.elapsed().as_secs_f64() * 1e6 / 100.0,
            child.len()
        );
    }
}
