// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting decimal arrays across storage widths, lengths, and validity paths.

mod common;

use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::dtype::DecimalType;
use vortex_array::validity::Validity;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexExpect;

use crate::common::cases;
use crate::common::decimal_array;

fn main() {
    divan::main();
}

#[divan::bench(args = cases())]
fn dbp_split_all_valid(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
    bench_split(bencher, values_type, len, Validity::AllValid);
}

#[divan::bench(args = cases())]
fn dbp_split_mixed_null(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
    let mut rng = StdRng::seed_from_u64(42);
    let validity = Validity::from_iter((0..len).map(|_| rng.random_bool(0.5)));
    bench_split(bencher, values_type, len, validity);
}

fn bench_split(bencher: Bencher, values_type: DecimalType, len: usize, validity: Validity) {
    let decimal = decimal_array(values_type, len, validity);
    let session = array_session();
    bencher
        .with_inputs(|| session.create_execution_ctx())
        .bench_refs(|ctx| {
            split_decimal(black_box(&decimal), ctx).vortex_expect("split decimal array")
        });
}
