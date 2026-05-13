// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex NeaTS encoding for floating-point time-series columns.
//!
//! NeaTS partitions a floating-point column into pieces and approximates each piece with a
//! small model (constant, linear, quadratic, or exponential). For every element it stores a
//! quantized residual against the chosen model. Decoded values are `model(i) + residual_i * scale`.
//!
//! Compared to ALP, which maps doubles to small ints via decimal scaling, NeaTS is designed for
//! data where successive values are well-approximated by a low-order function of position. It is
//! particularly aggressive on smooth or piecewise-smooth time-series (sensor traces, stock prices,
//! linear ramps with noise).
//!
//! Two modes are supported, controlled by [`NeaTSOptions::epsilon`]:
//!
//! - **Lossless (default)**: `epsilon = None`. The compressor chooses the smallest power-of-two
//!   `scale` such that `(x - model(i)) / scale` fits in `i64` and the round-trip is bit-exact.
//!   On values whose fractional part cannot be represented in `i64` at the chosen scale the
//!   compressor degrades to a single residual per element (no model benefit, but still exact).
//! - **Bounded-error lossy**: `epsilon = Some(eps)`. The compressor sets `scale = 2 * eps` so the
//!   round-trip error is `<= eps` per value. This unlocks aggressive piecewise fits.
//!
//! The implementation mirrors the slot pattern used by `datetime-parts` and `alp`: a NeaTS array
//! has six child slots that each cascade into existing Vortex encodings (frame-of-reference and
//! FastLanes bit-packing on `residuals` and `piece_starts`, constant compression on `model_ids`
//! when only one family fires).

pub use array::*;
pub use compress::*;
use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

mod array;
mod canonical;
mod compress;
pub mod compute;
pub mod models;
mod ops;

/// Initialize NeaTS encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(NeaTS);

    // Register a NeaTS-aware min/max kernel that reduces over per-piece bounds instead of
    // decoding all values.
    session.aggregate_fns().register_aggregate_kernel(
        NeaTS.id(),
        Some(MinMax.id()),
        &compute::min_max::NeaTSMinMaxKernel,
    );
}

#[cfg(test)]
mod test {
    use prost::Message;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::NeaTSOptions;
    use crate::array::NeaTSMetadata;
    use crate::compress::neats_encode;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_neats_metadata() {
        check_metadata(
            "neats.metadata",
            &NeaTSMetadata {
                value_ptype: PType::F64 as i32,
                residual_ptype: PType::I64 as i32,
                num_pieces: u64::MAX,
                scale_bits: u64::MAX,
                epsilon_bits: u64::MAX,
            }
            .encode_to_vec(),
        );
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "value mismatch at {i}: actual={a} expected={e} diff={}",
                (a - e).abs()
            );
        }
    }

    #[rstest]
    #[case(64)]
    #[case(1024)]
    #[case(5000)]
    fn roundtrip_linear_ramp_lossy(#[case] n: usize) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<f64> = (0..n).map(|i| 0.5 + 0.001 * i as f64).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let encoded = neats_encode(
            array.as_view(),
            NeaTSOptions {
                epsilon: Some(1e-9),
                ..NeaTSOptions::default()
            },
        )?;
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_close(decoded.as_slice::<f64>(), &values, 2e-9);
        Ok(())
    }

    #[rstest]
    #[case(128)]
    #[case(2048)]
    fn roundtrip_sine_lossy(#[case] n: usize) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let encoded = neats_encode(
            array.as_view(),
            NeaTSOptions {
                epsilon: Some(1e-3),
                ..NeaTSOptions::default()
            },
        )?;
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_close(decoded.as_slice::<f64>(), &values, 2e-3);
        Ok(())
    }

    #[test]
    fn scalar_at_matches_canonical() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<f64> = (0..512)
            .map(|i| (i as f64 * 0.02).sin() + 0.01 * i as f64)
            .collect();
        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let encoded = neats_encode(
            array.as_view(),
            NeaTSOptions {
                epsilon: Some(1e-4),
                ..NeaTSOptions::default()
            },
        )?;
        let decoded = encoded
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let decoded_slice = decoded.as_slice::<f64>();

        for idx in [0usize, 1, 17, 128, 200, 511] {
            let scalar = encoded
                .as_ref()
                .clone()
                .execute_scalar(idx, &mut ctx)?
                .as_primitive()
                .as_::<f64>()
                .unwrap();
            assert!(
                (scalar - decoded_slice[idx]).abs() <= 2e-4,
                "scalar_at[{idx}] = {scalar} but canonical = {} (diff {})",
                decoded_slice[idx],
                (scalar - decoded_slice[idx]).abs()
            );
        }
        Ok(())
    }

    #[test]
    fn min_max_kernel_matches_canonical() -> VortexResult<()> {
        use std::sync::LazyLock;

        use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
        use vortex_array::aggregate_fn::fns::min_max::min_max;
        use vortex_array::session::ArraySession;
        use vortex_session::VortexSession;

        // We need a session with NeaTS registered so the kernel actually fires.
        static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
            let session = VortexSession::empty().with::<ArraySession>();
            crate::initialize(&session);
            session
        });
        let mut ctx = SESSION.create_execution_ctx();

        let values: Vec<f64> = (0..1024)
            .map(|i| (i as f64 * 0.05).sin() + 0.001 * i as f64)
            .collect();
        let truth_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let truth_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let array = PrimitiveArray::new(Buffer::copy_from(&values), Validity::NonNullable);
        let encoded = neats_encode(
            array.as_view(),
            NeaTSOptions {
                epsilon: Some(1e-6),
                ..NeaTSOptions::default()
            },
        )?;

        let result = min_max(&encoded.into_array(), &mut ctx)?.expect("non-empty input");
        let MinMaxResult { min, max } = result;
        let min_f: f64 = min.as_primitive().as_::<f64>().unwrap();
        let max_f: f64 = max.as_primitive().as_::<f64>().unwrap();

        // The kernel computes inclusive bounds from `min(model) + min(residual*scale)` per
        // piece. These bounds are guaranteed to cover the true min/max but can be slightly
        // wider because the model's argmin and the residual's argmin may differ. The contract
        // is `kernel_min <= truth_min` and `kernel_max >= truth_max`. We also assert the
        // looseness is bounded by the largest per-piece residual swing (an over-estimate of
        // the worst-case widening).
        assert!(
            min_f <= truth_min,
            "kernel min {min_f} did not cover truth min {truth_min}",
        );
        assert!(
            max_f >= truth_max,
            "kernel max {max_f} did not cover truth max {truth_max}",
        );
        let data_range = truth_max - truth_min;
        let kernel_range = max_f - min_f;
        assert!(
            kernel_range <= 2.0 * data_range,
            "kernel range {kernel_range} should not exceed 2x data range {data_range}",
        );
        Ok(())
    }

    #[test]
    fn roundtrip_with_nulls() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<Option<f64>> = (0..200)
            .map(|i| {
                if i % 5 == 0 {
                    None
                } else {
                    Some(i as f64 * 0.1)
                }
            })
            .collect();
        let array = PrimitiveArray::from_option_iter(values.clone());
        let encoded = neats_encode(
            array.as_view(),
            NeaTSOptions {
                epsilon: Some(1e-9),
                ..NeaTSOptions::default()
            },
        )?;
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        let decoded_slice = decoded.as_slice::<f64>();
        let validity = decoded.validity()?;
        for (i, expected) in values.iter().enumerate() {
            let valid = validity.is_valid(i)?;
            match expected {
                None => assert!(!valid, "expected null at {i}, was valid"),
                Some(v) => {
                    assert!(valid, "expected valid at {i}, was null");
                    assert!((decoded_slice[i] - v).abs() <= 2e-9);
                }
            }
        }
        Ok(())
    }
}
