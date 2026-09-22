// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reassembling integer slices, including output-buffer allocation.
//! Array benchmarks also include validation, part execution, and casts.

mod common;

use divan::Bencher;
use divan::black_box;
use mimalloc::MiMalloc;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::i256;
use vortex_buffer::buffer;
use vortex_decimal_byte_parts::_benchmarking::assemble_wide_decimal;
use vortex_decimal_byte_parts::_benchmarking::i128_to_parts;
use vortex_decimal_byte_parts::_benchmarking::i256_to_parts;
use vortex_decimal_byte_parts::_benchmarking::split_wide;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::common::cases;
use crate::common::i128_values;
use crate::common::i256_values;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = cases())]
fn dbp_assemble_kernel(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
    let validity = Mask::new_true(len);
    match values_type {
        DecimalType::I128 => {
            let values = i128_values(len);
            let (msp, [lower]) = split_wide(values.as_slice(), &validity, i128_to_parts);
            bencher.bench(|| {
                assemble_wide_decimal::<i128, i64, 1>(
                    black_box(msp.as_slice()),
                    black_box(lower.as_slice()).iter().map(|&word| [word]),
                )
            });
        }
        DecimalType::I256 => {
            let values = i256_values(len);
            let (msp, [first, second, third]) =
                split_wide(values.as_slice(), &validity, i256_to_parts);
            bencher.bench(|| {
                assemble_wide_decimal::<i256, i64, 3>(
                    black_box(msp.as_slice()),
                    black_box(first.as_slice())
                        .iter()
                        .zip(black_box(second.as_slice()))
                        .zip(black_box(third.as_slice()))
                        .map(|((&a, &b), &c)| [a, b, c]),
                )
            });
        }
        _ => vortex_panic!("unsupported benchmark storage type: {values_type}"),
    }
}

// The kernel widens the MSP while assembling; lower parts are already `u64`.
#[vortex_bench_support::cpu_features]
#[divan::bench(args = cases())]
fn dbp_assemble_kernel_narrow_msp(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
    let msp = buffer![-1i8; len];
    match values_type {
        DecimalType::I128 => {
            let lower = buffer![u64::from(u8::MAX); len];
            bencher.bench(|| {
                assemble_wide_decimal::<i128, i8, 1>(
                    black_box(msp.as_slice()),
                    black_box(lower.as_slice()).iter().map(|&word| [word]),
                )
            });
        }
        DecimalType::I256 => {
            let first = buffer![u64::from(u8::MAX); len];
            let second = buffer![u64::from(u16::MAX); len];
            let third = buffer![u64::from(u32::MAX); len];
            bencher.bench(|| {
                assemble_wide_decimal::<i256, i8, 3>(
                    black_box(msp.as_slice()),
                    black_box(first.as_slice())
                        .iter()
                        .zip(black_box(second.as_slice()))
                        .zip(black_box(third.as_slice()))
                        .map(|((&a, &b), &c)| [a, b, c]),
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::DecimalType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_decimal_byte_parts::_benchmarking::assemble_decimal;
    use vortex_decimal_byte_parts::split_decimal;
    use vortex_error::VortexExpect;
    use vortex_error::vortex_panic;

    use crate::common::arrays::cases;
    use crate::common::arrays::decimal_array;

    #[divan::bench(args = cases())]
    fn dbp_assemble(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
        let decimal = decimal_array(values_type, len, Validity::NonNullable);
        let mut ctx = array_session().create_execution_ctx();
        let parts = split_decimal(&decimal, &mut ctx).vortex_expect("split benchmark input");
        bench_parts(
            bencher,
            parts.msp,
            parts.lower_parts,
            decimal.decimal_dtype(),
        );
    }

    #[divan::bench(args = cases())]
    fn dbp_assemble_narrowed(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
        let (precision, lower_parts) = match values_type {
            DecimalType::I64 => (18, vec![]),
            DecimalType::I128 => (38, vec![buffer![u8::MAX; len].into_array()]),
            DecimalType::I256 => (
                76,
                vec![
                    buffer![u8::MAX; len].into_array(),
                    buffer![u16::MAX; len].into_array(),
                    buffer![u32::MAX; len].into_array(),
                ],
            ),
            _ => vortex_panic!("unsupported benchmark storage type: {values_type}"),
        };
        bench_parts(
            bencher,
            buffer![-1i8; len].into_array(),
            lower_parts,
            DecimalDType::new(precision, 2),
        );
    }

    fn bench_parts(
        bencher: Bencher,
        msp: ArrayRef,
        lower_parts: Vec<ArrayRef>,
        decimal_dtype: DecimalDType,
    ) {
        let session = array_session();
        bencher
            .with_inputs(|| session.create_execution_ctx())
            .bench_refs(|ctx| {
                assemble_decimal(black_box(&msp), black_box(&lower_parts), decimal_dtype, ctx)
                    .vortex_expect("assemble decimal byte parts")
            });
    }
}
