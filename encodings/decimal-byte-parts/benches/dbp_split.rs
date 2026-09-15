// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting integer slices, including output-buffer allocation.
//! Array benchmarks also include validity execution and array construction.

mod common;

use divan::Bencher;
use divan::black_box;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::dtype::DecimalType;
use vortex_decimal_byte_parts::_benchmarking::i128_to_parts;
use vortex_decimal_byte_parts::_benchmarking::i256_to_parts;
use vortex_decimal_byte_parts::_benchmarking::split_wide;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::common::cases;
use crate::common::i128_values;
use crate::common::i256_values;

fn main() {
    divan::main();
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = cases())]
fn dbp_split_kernel_all_valid(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
    bench_split(bencher, values_type, len, Mask::new_true(len));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = cases())]
fn dbp_split_kernel_mixed_null(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
    let mut rng = StdRng::seed_from_u64(42);
    let validity = Mask::from_iter((0..len).map(|_| rng.random_bool(0.5)));
    bench_split(bencher, values_type, len, validity);
}

fn bench_split(bencher: Bencher, values_type: DecimalType, len: usize, validity: Mask) {
    match values_type {
        DecimalType::I128 => {
            let values = i128_values(len);
            bencher.bench(|| {
                split_wide(
                    black_box(values.as_slice()),
                    black_box(&validity),
                    i128_to_parts,
                )
            });
        }
        DecimalType::I256 => {
            let values = i256_values(len);
            bencher.bench(|| {
                split_wide(
                    black_box(values.as_slice()),
                    black_box(&validity),
                    i256_to_parts,
                )
            });
        }
        _ => vortex_panic!("unsupported benchmark storage type: {values_type}"),
    }
}

/// Benchmarks of full end-to-end dbp assembly, including kernel. These
/// are more volatile and should not be run in CI due to noise.
#[cfg(not(codspeed))]
mod arrays {
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

    use crate::common::arrays::cases;
    use crate::common::arrays::decimal_array;

    #[divan::bench(args = cases())]
    fn dbp_split_all_valid(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
        bench_split(bencher, values_type, len, Validity::AllValid);
    }

    #[divan::bench(args = cases())]
    fn dbp_split_all_null(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
        bench_split(bencher, values_type, len, Validity::AllInvalid);
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
}
