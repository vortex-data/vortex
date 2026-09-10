// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared decimal inputs for splitting and assembly benchmarks.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::arrays::DecimalArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::vortex_panic;

pub(super) fn cases() -> Vec<(DecimalType, usize)> {
    [DecimalType::I64, DecimalType::I128, DecimalType::I256]
        .into_iter()
        .flat_map(|values_type| [1_024, 8_192].map(|len| (values_type, len)))
        .collect()
}

pub(super) fn decimal_array(
    values_type: DecimalType,
    len: usize,
    validity: Validity,
) -> DecimalArray {
    let mut rng = StdRng::seed_from_u64(42);

    macro_rules! decimal {
        ($T:ty, $precision:literal) => {{
            let max = <$T>::pow(10, $precision) - 1;
            let values: Buffer<$T> = (0..len).map(|_| rng.random_range(-max..=max)).collect();
            DecimalArray::new(values, DecimalDType::new($precision, 2), validity)
        }};
    }

    match values_type {
        DecimalType::I64 => decimal!(i64, 18),
        DecimalType::I128 => decimal!(i128, 38),
        DecimalType::I256 => {
            // Keep the magnitude below 10^76 while exercising all four signed/unsigned words.
            let values: Buffer<i256> = (0..len)
                .map(|_| i256::from_parts(rng.random(), rng.random::<i128>() >> 4))
                .collect();
            DecimalArray::new(values, DecimalDType::new(76, 2), validity)
        }
        _ => vortex_panic!("unsupported benchmark storage type: {values_type}"),
    }
}
