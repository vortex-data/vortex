// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_error::VortexResult;

use crate::CompareOp;
use crate::HszConfig;
use crate::stage::Hsz;

fn smooth_signal(n: usize) -> Vec<f64> {
    use std::f64::consts::TAU;
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            10.0 + 5.0 * (t * TAU).sin() + 0.1 * (t * 5.0 * TAU).cos()
        })
        .collect()
}

#[rstest]
#[case(0)]
#[case(1)]
#[case(7)]
#[case(1023)]
#[case(1024)]
#[case(1025)]
#[case(10_000)]
fn roundtrip_within_eps(#[case] n: usize) -> VortexResult<()> {
    let values = smooth_signal(n);
    let eps = 1e-3;
    let hsz = Hsz::compress(&values, HszConfig { eps })?;
    assert_eq!(hsz.len(), n);
    let decoded = hsz.decompress();
    assert_eq!(decoded.len(), n);
    for (i, (a, b)) in values.iter().zip(decoded.as_slice()).enumerate() {
        assert!(
            (a - b).abs() <= eps,
            "position {i}: original {a} vs decoded {b}, diff {} > eps {eps}",
            (a - b).abs()
        );
    }
    Ok(())
}

#[test]
fn rejects_invalid_config() {
    let values = vec![1.0, 2.0, 3.0];
    assert!(Hsz::compress(&values, HszConfig { eps: 0.0 }).is_err());
    assert!(Hsz::compress(&values, HszConfig { eps: -1.0 }).is_err());
    assert!(Hsz::compress(&values, HszConfig { eps: f64::NAN }).is_err());
    assert!(Hsz::compress(&values, HszConfig { eps: f64::INFINITY }).is_err());
}

#[test]
fn sum_matches_reference() -> VortexResult<()> {
    let values = smooth_signal(4096);
    let hsz = Hsz::compress(&values, HszConfig::default())?;
    let reference: f64 = values.iter().sum();
    let homomorphic = hsz.sum();
    let from_residuals = hsz.sum_from_residuals();
    assert!(
        (reference - homomorphic).abs() < 1e-9,
        "Stage-0 sum drifted: ref={reference} hsz={homomorphic}"
    );
    let bound = (values.len() as f64) * hsz.eps();
    assert!(
        (reference - from_residuals).abs() <= bound,
        "Stage-1 sum exceeded eps*len bound: ref={reference} hsz={from_residuals} bound={bound}"
    );
    Ok(())
}

#[test]
fn mean_matches_reference() -> VortexResult<()> {
    let values = smooth_signal(2048);
    let hsz = Hsz::compress(&values, HszConfig::default())?;
    let reference: f64 = values.iter().sum::<f64>() / values.len() as f64;
    assert!((hsz.mean() - reference).abs() < 1e-9);
    Ok(())
}

#[test]
fn min_max_match_reference() -> VortexResult<()> {
    let values = smooth_signal(2048);
    let hsz = Hsz::compress(&values, HszConfig::default())?;
    let min_ref = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ref = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(hsz.min(), min_ref);
    assert_eq!(hsz.max(), max_ref);
    Ok(())
}

#[test]
fn between_mask_is_exact() -> VortexResult<()> {
    let values = smooth_signal(8 * 1024);
    let hsz = Hsz::compress(&values, HszConfig { eps: 1e-4 })?;
    let (mask, stats) = hsz.between_mask(7.0, 12.0);
    let expected: usize = values.iter().filter(|v| **v >= 7.0 && **v <= 12.0).count();
    assert_eq!(mask.true_count(), expected);
    // We expect *some* blocks to be answered without descending into Stage 1
    // for a smooth signal.
    assert!(
        stats.blocks_all_true + stats.blocks_all_false > 0,
        "expected zone-map skipping, got {stats:?}"
    );
    Ok(())
}

#[test]
fn between_mask_full_range_is_all_true() -> VortexResult<()> {
    let values = smooth_signal(1024);
    let hsz = Hsz::compress(&values, HszConfig::default())?;
    let (mask, stats) = hsz.between_mask(f64::NEG_INFINITY, f64::INFINITY);
    assert_eq!(mask.true_count(), values.len());
    assert_eq!(stats.blocks_descended, 0);
    Ok(())
}

#[test]
fn between_mask_disjoint_range_is_all_false() -> VortexResult<()> {
    let values = smooth_signal(1024);
    let hsz = Hsz::compress(&values, HszConfig::default())?;
    let (mask, stats) = hsz.between_mask(1000.0, 2000.0);
    assert_eq!(mask.true_count(), 0);
    assert_eq!(stats.blocks_descended, 0);
    Ok(())
}

#[test]
fn slice_roundtrips() -> VortexResult<()> {
    let values = smooth_signal(2500);
    let hsz = Hsz::compress(&values, HszConfig { eps: 1e-3 })?;
    for &(s, e) in &[
        (0usize, 0usize),
        (0, 1024),
        (0, 2500),
        (137, 1500),
        (2499, 2500),
    ] {
        let sliced = hsz.slice(s..e)?;
        assert_eq!(sliced.len(), e - s);
        let decoded = sliced.decompress();
        for (i, &v) in decoded.as_slice().iter().enumerate() {
            assert!(
                (values[s + i] - v).abs() <= hsz.eps(),
                "slice {s}..{e} pos {i}: {} vs {v}",
                values[s + i]
            );
        }
    }
    Ok(())
}

#[test]
fn filter_roundtrips() -> VortexResult<()> {
    let values: Vec<f64> = (0..1024).map(|i| i as f64).collect();
    let hsz = Hsz::compress(&values, HszConfig { eps: 0.5 })?;
    let mask = vortex_mask::Mask::from_iter(values.iter().map(|v| (*v as i64) % 3 == 0));
    let filtered = hsz.filter(&mask)?;
    let expected: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| (*v as i64) % 3 == 0)
        .collect();
    assert_eq!(filtered.len(), expected.len());
    let decoded = filtered.decompress();
    for (i, &v) in decoded.as_slice().iter().enumerate() {
        assert!((expected[i] - v).abs() <= 0.5);
    }
    Ok(())
}

#[test]
fn take_uses_outliers_correctly() -> VortexResult<()> {
    // Construct data where one value is so far out it forces an outlier.
    let mut values: Vec<f64> = (0..1024).map(|i| i as f64).collect();
    values[42] = 1e20;
    let hsz = Hsz::compress(&values, HszConfig { eps: 0.5 })?;
    assert!(
        !hsz.outlier_indices().is_empty(),
        "expected the giant value to become an outlier"
    );
    let taken = hsz.take(&[0, 42, 100, 1023])?;
    assert!((taken.as_slice()[0] - 0.0).abs() <= 0.5);
    assert_eq!(taken.as_slice()[1], 1e20);
    assert!((taken.as_slice()[2] - 100.0).abs() <= 0.5);
    assert!((taken.as_slice()[3] - 1023.0).abs() <= 0.5);
    Ok(())
}

#[test]
fn outliers_roundtrip_through_decompress() -> VortexResult<()> {
    let mut values: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.1).collect();
    values[10] = 1e15;
    values[1000] = -1e15;
    let hsz = Hsz::compress(&values, HszConfig { eps: 1e-3 })?;
    let decoded = hsz.decompress();
    assert_eq!(decoded.as_slice()[10], 1e15);
    assert_eq!(decoded.as_slice()[1000], -1e15);
    for i in [0usize, 1, 50, 500, 1023] {
        assert!((decoded.as_slice()[i] - values[i]).abs() <= hsz.eps());
    }
    Ok(())
}

#[rstest]
#[case(CompareOp::Lt, 8.0)]
#[case(CompareOp::Le, 8.0)]
#[case(CompareOp::Gt, 8.0)]
#[case(CompareOp::Ge, 8.0)]
#[case(CompareOp::Eq, 8.0)]
#[case(CompareOp::Ne, 8.0)]
fn compare_mask_matches_naive(#[case] op: CompareOp, #[case] value: f64) -> VortexResult<()> {
    let values = smooth_signal(4096);
    let hsz = Hsz::compress(&values, HszConfig { eps: 1e-4 })?;
    let (mask, _stats) = hsz.compare_mask(op, value);
    let expected: usize = values
        .iter()
        .filter(|v| match op {
            CompareOp::Lt => **v < value,
            CompareOp::Le => **v <= value,
            CompareOp::Gt => **v > value,
            CompareOp::Ge => **v >= value,
            CompareOp::Eq => **v == value,
            CompareOp::Ne => **v != value,
        })
        .count();
    // Strict equality on f64 is tight; with `eps = 1e-4` and a smooth signal
    // we shouldn't hit a value exactly equal to 8.0 except by chance, so
    // both Eq and Ne should agree with the naive computation.
    let diff = mask.true_count().abs_diff(expected);
    assert!(
        diff <= 2,
        "op={op:?} value={value}: mask={} naive={} diff={diff}",
        mask.true_count(),
        expected
    );
    Ok(())
}

#[test]
fn compare_skips_blocks_for_smooth_signal() -> VortexResult<()> {
    let values = smooth_signal(8 * 1024);
    let hsz = Hsz::compress(&values, HszConfig { eps: 1e-4 })?;
    let (_, stats) = hsz.compare_mask(CompareOp::Gt, 100.0);
    assert!(stats.blocks_descended == 0);
    assert!(stats.blocks_all_false > 0);
    let (_, stats) = hsz.compare_mask(CompareOp::Lt, -100.0);
    assert!(stats.blocks_descended == 0);
    assert!(stats.blocks_all_false > 0);
    let (_, stats) = hsz.compare_mask(CompareOp::Ge, -100.0);
    assert!(stats.blocks_descended == 0);
    assert!(stats.blocks_all_true > 0);
    Ok(())
}

#[test]
fn is_nan_finds_nan_outliers() -> VortexResult<()> {
    let mut values: Vec<f64> = (0..1024).map(|i| i as f64).collect();
    values[42] = f64::NAN;
    values[500] = f64::INFINITY;
    let hsz = Hsz::compress(&values, HszConfig { eps: 0.5 })?;
    let nan_mask = hsz.is_nan_mask();
    assert_eq!(nan_mask.true_count(), 1);
    assert!(nan_mask.value(42));
    assert!(!nan_mask.value(500));
    let fin_mask = hsz.is_finite_mask();
    assert_eq!(fin_mask.true_count(), values.len() - 2);
    assert!(!fin_mask.value(42));
    assert!(!fin_mask.value(500));
    assert_eq!(hsz.nan_count(), 1);
    assert_eq!(hsz.non_finite_count(), 2);
    Ok(())
}

#[test]
fn count_in_range_matches_naive() -> VortexResult<()> {
    let values = smooth_signal(4096);
    let hsz = Hsz::compress(&values, HszConfig { eps: 1e-4 })?;
    let expected = values.iter().filter(|v| **v >= 7.0 && **v <= 12.0).count();
    assert_eq!(hsz.count_in_range(7.0, 12.0), expected);
    assert_eq!(
        hsz.count_in_range(f64::NEG_INFINITY, f64::INFINITY),
        values.len()
    );
    assert_eq!(hsz.count_in_range(1000.0, 2000.0), 0);
    Ok(())
}

#[test]
fn is_constant_detects_constant_column() -> VortexResult<()> {
    let constant: Vec<f64> = vec![42.0; 2048];
    let hsz = Hsz::compress(&constant, HszConfig { eps: 1e-3 })?;
    assert!(hsz.is_constant());

    let almost_constant: Vec<f64> = (0..2048).map(|i| 42.0 + (i as f64) * 1e-4).collect();
    let hsz = Hsz::compress(&almost_constant, HszConfig { eps: 1e-3 })?;
    assert!(!hsz.is_constant());

    let empty: Vec<f64> = vec![];
    let hsz = Hsz::compress(&empty, HszConfig { eps: 1e-3 })?;
    assert!(!hsz.is_constant());
    Ok(())
}
