// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for `compute_referenced_elements_mask`, `compute_density`, and
//! `estimate_density` on `ListViewArray`.

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::common::create_basic_listview;
use super::common::create_empty_lists_listview;
use super::common::create_large_listview;
use super::common::create_overlapping_listview;
use super::common::create_sparse_overlapping_listview;
use crate::ExecutionCtx;
use crate::VortexSessionExecute;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::tests::common::create_empty_elements_listview;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::scalar::ScalarValue;
use crate::session::ArraySession;

const EPS: f32 = 1e-6;

fn test_execution_ctx() -> ExecutionCtx {
    let session = VortexSession::empty().with::<ArraySession>();
    session.create_execution_ctx()
}

#[test]
fn full_density_no_overlap() -> VortexResult<()> {
    let mut ctx = test_execution_ctx();
    let lv = create_basic_listview();
    let exact = lv.compute_density(&mut ctx)?;
    let est = lv.upper_bound_density(&mut ctx)?;

    assert!((exact - 1.0).abs() < EPS);
    assert!((est - 1.0).abs() < EPS);
    Ok(())
}

#[test]
fn sparse_no_overlap_matches_exact() -> VortexResult<()> {
    let mut ctx = test_execution_ctx();
    let lv = create_large_listview();
    let exact = lv.compute_density(&mut ctx)?;
    let est = lv.upper_bound_density(&mut ctx)?;

    assert!((exact - 0.5).abs() < EPS);
    assert!((est - 0.5).abs() < EPS);
    Ok(())
}

#[test]
fn all_empty_lists_is_zero_density() -> VortexResult<()> {
    let mut ctx = test_execution_ctx();
    let lv = create_empty_lists_listview();
    let exact = lv.compute_density(&mut ctx)?;
    let est = lv.upper_bound_density(&mut ctx)?;

    assert_eq!(exact, 0.0);
    assert_eq!(est, 0.0);
    Ok(())
}

#[test]
fn overlap_full_coverage_clamps_estimate() -> VortexResult<()> {
    let mut ctx = test_execution_ctx();
    let lv = create_overlapping_listview();
    let exact = lv.compute_density(&mut ctx)?;
    let est = lv.upper_bound_density(&mut ctx)?;

    assert!((exact - 1.0).abs() < EPS);
    assert!((est - 1.0).abs() < EPS);
    Ok(())
}

#[test]
fn overlap_differential_exact_lower_than_estimate() -> VortexResult<()> {
    let mut ctx = test_execution_ctx();
    let lv = create_sparse_overlapping_listview();

    let exact = lv.compute_density(&mut ctx)?;
    let est = lv.upper_bound_density(&mut ctx)?;

    assert!((exact - 0.25).abs() < EPS);
    assert!((est - 0.40).abs() < EPS);
    Ok(())
}

#[test]
fn empty_elements_returns_one() -> VortexResult<()> {
    let mut ctx = test_execution_ctx();
    let lv = create_empty_elements_listview();

    let exact = lv.compute_density(&mut ctx)?;
    let est = lv.upper_bound_density(&mut ctx)?;

    assert!((exact - 1.0).abs() < EPS);
    assert!((est - 1.0).abs() < EPS);
    Ok(())
}

#[test]
fn estimate_uses_cached_sum_stat() -> VortexResult<()> {
    let mut ctx = test_execution_ctx();
    let lv = create_basic_listview();
    // Pre-populate Stat::Sum with a deliberately-wrong 5 so we can prove
    // estimate_density reads from the cache instead of computing fresh.
    lv.sizes()
        .statistics()
        .set(Stat::Sum, Precision::Exact(ScalarValue::from(5u64)));

    let est = lv.upper_bound_density(&mut ctx)?;
    assert!((est - 0.5).abs() < EPS);
    Ok(())
}

#[test]
fn referenced_mask_set_bits_match_views() -> VortexResult<()> {
    let mut ctx = test_execution_ctx();
    let lv = create_sparse_overlapping_listview();
    let mask = lv.compute_referenced_elements_mask(&mut ctx)?;
    let bits = match mask {
        Mask::Values(v) => v,
        _ => panic!("expected Values mask"),
    };

    assert_eq!(bits.true_count(), 5);
    let bb = bits.bit_buffer();
    for i in 0..3 {
        assert!(bb.value(i));
    }
    assert!(bb.value(18));
    assert!(bb.value(19));
    Ok(())
}
