// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

pub use bitpacking::*;
pub use delta::*;
pub use r#for::*;

mod bitpacking;
mod delta;
mod r#for;

#[cfg(test)]
mod test {
    use arrow_buffer::BooleanBuffer;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_array::pipeline::export_canonical_pipeline_expr;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::BufferMut;
    use vortex_mask::Mask;

    use crate::bitpack_to_best_bit_width;

    #[test]
    fn test_fastlanes() {
        let fraction_kept = 0.01;

        let mut rng = StdRng::seed_from_u64(0);
        let values = (0..100_000)
            .map(|_| u16::try_from(rng.random_range(0..100)).unwrap())
            .collect::<BufferMut<u16>>()
            .into_array()
            .to_primitive();
        let array = bitpack_to_best_bit_width(&values).unwrap();

        let mask = Mask::from_buffer(
            (0..100_000)
                .map(|_| rng.random_bool(fraction_kept))
                .collect::<BooleanBuffer>(),
        );

        let tc = mask.true_count();

        let array = export_canonical_pipeline_expr(array.dtype(), array.len(), &array, &mask)
            .unwrap()
            .into_primitive();
        assert_eq!(array.len(), tc);
    }
}
