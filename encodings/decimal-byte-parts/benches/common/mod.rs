// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared inputs for splitting and assembly benchmarks.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::i256;
use vortex_buffer::Buffer;

/// Decimal widths under test, with the input plus output bytes each row moves.
///
/// Splitting reads one decimal and writes a most significant part plus lower parts of the same
/// total width, and assembly does the reverse, so both move twice the decimal's size per row.
const WIDTHS: [(DecimalType, usize); 2] = [
    (DecimalType::I128, 2 * size_of::<i128>()),
    (DecimalType::I256, 2 * size_of::<i256>()),
];

/// Working set per kernel iteration.
///
/// The kernel benchmarks run on the walltime legs, where a case that takes under a microsecond
/// reports mostly per-iteration jitter: at 1,024 rows the `i128` kernels took 0.4 to 1.2 µs and
/// moved by 10 to 19% on pull requests that changed no decimal code. Budgeting by bytes puts
/// every case in the microsecond range, and the largest one still fits the 1 MiB L2 cache of the
/// Graviton leg, so these stay measurements of kernel code rather than of memory bandwidth.
const WORKING_SET_BYTES: [usize; 2] = [256 * 1024, 1024 * 1024];

pub(super) fn cases() -> Vec<(DecimalType, usize)> {
    WIDTHS
        .into_iter()
        .flat_map(|(values_type, bytes_per_row)| {
            WORKING_SET_BYTES.map(|bytes| (values_type, bytes / bytes_per_row))
        })
        .collect()
}

pub(super) fn i128_values(len: usize) -> Buffer<i128> {
    let mut rng = StdRng::seed_from_u64(42);
    let max = 10i128.pow(38) - 1;
    (0..len).map(|_| rng.random_range(-max..=max)).collect()
}

pub(super) fn i256_values(len: usize) -> Buffer<i256> {
    let mut rng = StdRng::seed_from_u64(42);
    // Keep the magnitude below 10^76 while exercising all four signed/unsigned words.
    (0..len)
        .map(|_| i256::from_parts(rng.random(), rng.random::<i128>() >> 4))
        .collect()
}

#[cfg(not(codspeed))]
pub(super) mod arrays {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::DecimalType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::vortex_panic;

    use super::i128_values;
    use super::i256_values;

    /// These benchmarks never run in CI, so they keep the small row counts that make a local
    /// `cargo bench` quick, rather than the kernel benchmarks' walltime-safe sizes.
    pub(crate) fn cases() -> Vec<(DecimalType, usize)> {
        [DecimalType::I64, DecimalType::I128, DecimalType::I256]
            .into_iter()
            .flat_map(|values_type| [1_024, 8_192].map(|len| (values_type, len)))
            .collect()
    }

    pub(crate) fn decimal_array(
        values_type: DecimalType,
        len: usize,
        validity: Validity,
    ) -> DecimalArray {
        match values_type {
            DecimalType::I64 => {
                let mut rng = StdRng::seed_from_u64(42);
                let max = 10i64.pow(18) - 1;
                let values: Buffer<i64> = (0..len).map(|_| rng.random_range(-max..=max)).collect();
                DecimalArray::new(values, DecimalDType::new(18, 2), validity)
            }
            DecimalType::I128 => {
                DecimalArray::new(i128_values(len), DecimalDType::new(38, 2), validity)
            }
            DecimalType::I256 => {
                DecimalArray::new(i256_values(len), DecimalDType::new(76, 2), validity)
            }
            _ => vortex_panic!("unsupported benchmark storage type: {values_type}"),
        }
    }
}
