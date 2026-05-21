// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for `compute_referenced_elements_mask`, `compute_density`, and
//! `estimate_density` on `ListViewArray`.

use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::common::create_basic_listview;
use super::common::create_empty_lists_listview;
use super::common::create_large_listview;
use super::common::create_overlapping_listview;
use crate::IntoArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::scalar::ScalarValue;
use crate::validity::Validity;

const EPS: f32 = 1e-6;

#[test]
fn full_density_no_overlap() -> VortexResult<()> {
    let lv = create_basic_listview();
    let exact = lv.compute_density().expect("non-empty elements");
    let est = lv.estimate_density()?.expect("non-empty elements");

    assert!((exact - 1.0).abs() < EPS, "exact density {exact}");
    assert!((est - 1.0).abs() < EPS, "estimate density {est}");
    Ok(())
}

#[test]
fn sparse_no_overlap_matches_exact() -> VortexResult<()> {
    let lv = create_large_listview();
    let exact = lv.compute_density().expect("non-empty");
    let est = lv.estimate_density()?.expect("non-empty");

    assert!((exact - 0.5).abs() < EPS, "exact density {exact}");
    assert!((est - 0.5).abs() < EPS, "estimate density {est}");
    Ok(())
}

#[test]
fn all_empty_lists_is_zero_density() -> VortexResult<()> {
    let lv = create_empty_lists_listview();
    let exact = lv.compute_density().expect("elements has length 1");
    let est = lv.estimate_density()?.expect("elements has length 1");

    assert_eq!(exact, 0.0);
    assert_eq!(est, 0.0);
    Ok(())
}

#[test]
fn overlap_full_coverage_clamps_estimate() -> VortexResult<()> {
    let lv = create_overlapping_listview();
    let exact = lv.compute_density().expect("non-empty");
    let est = lv.estimate_density()?.expect("non-empty");

    assert!((exact - 1.0).abs() < EPS, "exact density {exact}");
    assert!((est - 1.0).abs() < EPS, "estimate density {est}");
    Ok(())
}

#[test]
fn overlap_differential_exact_lower_than_estimate() -> VortexResult<()> {
    // Two rows both pointing at elements[0..5) over a 20-element buffer.
    // Unique referenced = 5  → exact = 0.25
    // sum(sizes) = 10        → estimate = 0.50
    let elements = PrimitiveArray::from_iter(0i32..20).into_array();
    let offsets = buffer![0u32, 0].into_array();
    let sizes = buffer![5u32, 5].into_array();
    let lv = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)?;

    let exact = lv.compute_density().expect("non-empty");
    let est = lv.estimate_density()?.expect("non-empty");

    assert!((exact - 0.25).abs() < EPS, "exact density {exact}");
    assert!((est - 0.50).abs() < EPS, "estimate density {est}");
    assert!(est > exact, "estimate must overcount overlapping views");
    Ok(())
}

#[test]
fn empty_elements_returns_none() -> VortexResult<()> {
    let elements = PrimitiveArray::from_iter::<[i32; 0]>([]).into_array();
    let offsets = buffer![0u32; 0].into_array();
    let sizes = buffer![0u32; 0].into_array();
    let lv = ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)?;

    assert!(lv.compute_density().is_none());
    assert!(lv.estimate_density()?.is_none());
    Ok(())
}

#[test]
fn estimate_uses_cached_sum_stat() -> VortexResult<()> {
    let lv = create_basic_listview();
    // Pre-populate Stat::Sum with a deliberately-wrong 5 so we can prove
    // estimate_density reads from the cache instead of computing fresh.
    lv.sizes()
        .statistics()
        .set(Stat::Sum, Precision::Exact(ScalarValue::from(5u64)));

    let est = lv.estimate_density()?.expect("non-empty");
    assert!(
        (est - 0.5).abs() < EPS,
        "estimate {est} should reflect cached Sum=5, not computed Sum=10",
    );
    Ok(())
}

#[test]
fn referenced_mask_set_bits_match_views() -> VortexResult<()> {
    // create_large_listview: 10 lists of size 50 at offsets [0,100,200,...,900].
    // Bits [i*100 .. i*100+50) set, the rest unset.
    let lv = create_large_listview();
    let mask = lv
        .compute_referenced_elements_mask()
        .expect("non-empty elements");
    let bits = match mask {
        Mask::Values(v) => v,
        other => panic!("expected Values mask for partial coverage, got {other:?}"),
    };

    assert_eq!(bits.true_count(), 500);
    // Spot-check the boundaries of the first and last views.
    let bb = bits.bit_buffer();
    assert!(bb.value(0));
    assert!(bb.value(49));
    assert!(!bb.value(50));
    assert!(!bb.value(99));
    assert!(bb.value(100));
    assert!(bb.value(949));
    assert!(!bb.value(950));
    Ok(())
}
