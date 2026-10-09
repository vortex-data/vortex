// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sweeps take shapes and prints the time of the take kernel against decoding every value and
//! gathering views, the path without a kernel. Run with `cargo bench --bench take_sweep`.

#![allow(
    clippy::many_single_char_names,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::hint::black_box;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = array_session();
    vortex_fsst::initialize(&session);
    session
});

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 16
    }
}

fn strings(n: usize, shape: &str) -> VarBinArray {
    let mut next = lcg(7);
    let v: Vec<String> = (0..n)
        .map(|i| match shape {
            "short" => format!("{:06x}", next() % 4096),
            "url" => format!("https://www.example.com/products/{i:08x}?ref={:012x}", next()),
            _ => format!(
                "2026-05-14T12:34:56.789012Z INFO request_id={:016x} method=GET path=/api/v1/users/{i:08x}/profile status=200 bytes={}",
                next(), next() % 100000
            ),
        })
        .collect();
    VarBinArray::from_iter(
        v.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    )
}

fn indices(n: usize, k: usize, range: usize, null_pct: u64) -> ArrayRef {
    let mut next = lcg(42);
    let mult = 0x9e37_79b1u64 | 1;
    let it = (0..k).map(|_| {
        let r = next();
        let idx = ((r % range as u64) * mult % n as u64) as u32;
        (next() % 100 >= null_pct).then_some(idx)
    });
    if null_pct == 0 {
        PrimitiveArray::from_iter(it.map(|v| v.unwrap())).into_array()
    } else {
        PrimitiveArray::from_option_iter(it).into_array()
    }
}

fn time(mut f: impl FnMut(), budget: Duration) -> Duration {
    f();
    let mut best = Duration::MAX;
    let start = Instant::now();
    let mut iters = 0;
    while iters < 5 || (start.elapsed() < budget && iters < 2000) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        iters += 1;
    }
    best
}

fn main() {
    let budget = Duration::from_millis(60);
    println!("shape n k range null% kernel_us nokernel_us ratio");
    for shape in ["short", "url", "long"] {
        for n in [64usize, 1024, 65_536] {
            let values = fsst(&strings(n, shape).into_array());
            for kf in [0.1, 0.3, 0.45, 0.49, 0.5, 0.75, 1.0, 2.0, 3.9, 8.0] {
                for rf in [0.02, 0.1, 0.19, 0.21, 0.5, 1.0] {
                    for null_pct in [0u64, 50] {
                        let k = ((n as f64 * kf) as usize).max(1);
                        let range = ((n as f64 * rf) as usize).max(1);
                        let idx = indices(n, k, range, null_pct);
                        let kern = time(
                            || {
                                let mut ctx = SESSION.create_execution_ctx();
                                black_box(
                                    DictArray::try_new(idx.clone(), values.clone())
                                        .unwrap()
                                        .into_array()
                                        .execute::<VarBinViewArray>(&mut ctx)
                                        .unwrap(),
                                );
                            },
                            budget,
                        );
                        let base = time(
                            || {
                                let mut ctx = SESSION.create_execution_ctx();
                                let d = values
                                    .clone()
                                    .execute::<Canonical>(&mut ctx)
                                    .unwrap()
                                    .into_array();
                                black_box(
                                    DictArray::try_new(idx.clone(), d)
                                        .unwrap()
                                        .into_array()
                                        .execute::<VarBinViewArray>(&mut ctx)
                                        .unwrap(),
                                );
                            },
                            budget,
                        );
                        println!(
                            "{shape} {n} {k} {range} {null_pct} {:.2} {:.2} {:.2}",
                            kern.as_secs_f64() * 1e6,
                            base.as_secs_f64() * 1e6,
                            kern.as_secs_f64() / base.as_secs_f64()
                        );
                    }
                }
            }
        }
    }
}

fn fsst(a: &ArrayRef) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let c = vortex_fsst::fsst_train_compressor(a, &mut ctx).unwrap();
    vortex_fsst::fsst_compress(a, &c, &mut ctx)
        .unwrap()
        .into_array()
}
