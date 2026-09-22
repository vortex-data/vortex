// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native wall-time probes. Fixture construction is outside the timed calls.

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::scalar_fn::unstable::row::InputElement;
use vortex_array::scalar_fn::unstable::row::Utf8Column;
use vortex_error::VortexResult;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Warm each case, then calibrate repeated calls to at least two milliseconds per sample.
fn measure<T>(name: &str, rows: usize, mut run: impl FnMut() -> T) {
    for _ in 0..200 {
        drop(black_box(run()));
    }
    let mut iterations = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            drop(black_box(run()));
        }
        if start.elapsed() >= Duration::from_millis(2) {
            break;
        }
        iterations *= 2;
    }
    for sample in 0..12 {
        let start = Instant::now();
        for _ in 0..iterations {
            drop(black_box(run()));
        }
        println!(
            "{name},{rows},{sample},{iterations},{:.3}",
            start.elapsed().as_nanos() as f64 / iterations as f64
        );
    }
}

fn main() -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    println!("case,rows,sample,iterations,ns");
    for rows in [0, 1, 64, 1024, 16384] {
        for (kind, value) in [
            ("inline", "hello"),
            ("external", "a longer external UTF-8 string λ"),
        ] {
            let array =
                VarBinViewArray::from_iter_str(std::iter::repeat_n(value, rows)).into_array();
            let decoded = Utf8Column::decode(array.clone(), &mut ctx)?;
            if rows > 0 {
                assert_eq!(Utf8Column::get(&decoded, rows - 1).as_str(), value);
            }
            measure(&format!("utf8_decode_{kind}"), rows, || {
                Utf8Column::decode(black_box(array.clone()), &mut ctx).unwrap()
            });
        }
    }
    Ok(())
}
