// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant vector quantization encoding for Vortex.
//!
//! Implements the TurboQuant algorithm for lossy compression of high-dimensional vector data.
//! Supports two variants:
//! - **MSE**: Optimal for mean-squared error reconstruction
//! - **Prod**: Optimal for inner product preservation (unbiased)
//!
//! The encoding operates on `FixedSizeList` arrays of floats (the storage format of
//! `Vector` and `FixedShapeTensor` extension types).

pub use array::TurboQuant;
pub use array::TurboQuantArray;
pub use array::TurboQuantVariant;
pub use compress::TurboQuantConfig;
pub use compress::turboquant_encode;

mod array;
pub mod centroids;
mod compress;
mod decompress;
pub mod rotation;
mod rules;

use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize the TurboQuant encoding in the given session.
pub fn initialize(session: &mut VortexSession) {
    session.arrays().register(TurboQuant);
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::TurboQuantConfig;
    use crate::TurboQuantVariant;
    use crate::turboquant_encode;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// Create a FixedSizeListArray of random f32 vectors.
    fn make_fsl(num_rows: usize, dim: usize, seed: u64) -> FixedSizeListArray {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::Distribution;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0).unwrap();

        let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim);
        for _ in 0..(num_rows * dim) {
            buf.push(normal.sample(&mut rng));
        }

        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
        FixedSizeListArray::try_new(
            elements.into_array(),
            dim as u32,
            Validity::NonNullable,
            num_rows,
        )
        .unwrap()
    }

    /// Compute MSE between original and reconstructed vectors.
    fn compute_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
        assert_eq!(original.len(), reconstructed.len());
        let n = original.len() as f32;
        original
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / n
    }

    #[rstest]
    #[case(32, 1)]
    #[case(32, 2)]
    #[case(32, 3)]
    #[case(32, 4)]
    #[case(128, 2)]
    #[case(128, 4)]
    #[case(256, 2)]
    fn roundtrip_mse(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 10;
        let fsl = make_fsl(num_rows, dim, 42);
        let original_elements: Vec<f32> = {
            let prim = fsl.elements().to_canonical().unwrap().into_primitive();
            prim.as_slice::<f32>().to_vec()
        };

        let config = TurboQuantConfig {
            bit_width,
            variant: TurboQuantVariant::Mse,
            seed: Some(123),
        };

        let encoded = turboquant_encode(&fsl, &config)?;
        assert_eq!(encoded.dimension(), dim as u32);
        assert_eq!(encoded.bit_width(), bit_width);

        // Decode.
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = encoded
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        assert_eq!(decoded.len(), num_rows);

        let decoded_elements: Vec<f32> = {
            let prim = decoded.elements().to_canonical().unwrap().into_primitive();
            prim.as_slice::<f32>().to_vec()
        };

        // Verify MSE is bounded. Higher bit_width = lower error.
        let mse = compute_mse(&original_elements, &decoded_elements);
        let avg_norm: f32 = (0..num_rows)
            .map(|i| {
                let row = &original_elements[i * dim..(i + 1) * dim];
                row.iter().map(|&v| v * v).sum::<f32>().sqrt()
            })
            .sum::<f32>()
            / num_rows as f32;

        // Normalized MSE should decrease with more bits.
        let normalized_mse = mse / (avg_norm * avg_norm + 1e-10);
        // Generous bound: normalized MSE should be < 1 for any bit_width >= 1.
        assert!(
            normalized_mse < 1.0,
            "Normalized MSE too high: {normalized_mse} for dim={dim}, bits={bit_width}"
        );

        Ok(())
    }

    #[rstest]
    #[case(32, 2)]
    #[case(32, 3)]
    #[case(128, 2)]
    #[case(128, 4)]
    fn roundtrip_prod(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 10;
        let fsl = make_fsl(num_rows, dim, 42);

        let config = TurboQuantConfig {
            bit_width,
            variant: TurboQuantVariant::Prod,
            seed: Some(456),
        };

        let encoded = turboquant_encode(&fsl, &config)?;
        assert_eq!(encoded.variant(), TurboQuantVariant::Prod);

        // Decode.
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = encoded
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        assert_eq!(decoded.len(), num_rows);

        Ok(())
    }

    #[test]
    fn roundtrip_empty() -> VortexResult<()> {
        let fsl = make_fsl(0, 128, 0);
        let config = TurboQuantConfig {
            bit_width: 2,
            variant: TurboQuantVariant::Mse,
            seed: Some(0),
        };

        let encoded = turboquant_encode(&fsl, &config)?;
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = encoded
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        assert_eq!(decoded.len(), 0);

        Ok(())
    }

    #[test]
    fn higher_bits_lower_error() -> VortexResult<()> {
        let dim = 128;
        let num_rows = 50;
        let fsl = make_fsl(num_rows, dim, 99);
        let original: Vec<f32> = {
            let prim = fsl.elements().to_canonical().unwrap().into_primitive();
            prim.as_slice::<f32>().to_vec()
        };

        let mut prev_mse = f32::MAX;
        for bit_width in 1..=4u8 {
            let config = TurboQuantConfig {
                bit_width,
                variant: TurboQuantVariant::Mse,
                seed: Some(123),
            };

            let encoded = turboquant_encode(&fsl, &config)?;
            let mut ctx = SESSION.create_execution_ctx();
            let decoded = encoded
                .into_array()
                .execute::<FixedSizeListArray>(&mut ctx)?;
            let decoded_elements: Vec<f32> = {
                let prim = decoded.elements().to_canonical().unwrap().into_primitive();
                prim.as_slice::<f32>().to_vec()
            };

            let mse = compute_mse(&original, &decoded_elements);
            assert!(
                mse <= prev_mse,
                "MSE should decrease with more bits: {bit_width}-bit MSE={mse} > previous={prev_mse}"
            );
            prev_mse = mse;
        }

        Ok(())
    }
}
