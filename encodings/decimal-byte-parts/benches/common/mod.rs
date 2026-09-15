// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared inputs for splitting and assembly benchmarks.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::i256;
use vortex_buffer::Buffer;

pub(super) fn cases() -> Vec<(DecimalType, usize)> {
    [DecimalType::I128, DecimalType::I256]
        .into_iter()
        .flat_map(|values_type| [1_024, 8_192].map(|len| (values_type, len)))
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
