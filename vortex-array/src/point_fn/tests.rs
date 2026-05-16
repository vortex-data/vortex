// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the `point_fn` module: runtime, session, generic search_sorted.

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::point_fn::BlockKey;
use crate::point_fn::PointDispatch;
use crate::point_fn::PointDispatchExt;
use crate::point_fn::PointRuntime;
use crate::point_fn::PointSession;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

fn sorted_primitive() -> crate::ArrayRef {
    PrimitiveArray::new(
        buffer![0i32, 1, 2, 2, 2, 3, 5, 5, 8, 13, 21, 34, 55, 89, 144],
        crate::validity::Validity::NonNullable,
    )
    .into_array()
}

#[test]
fn runtime_scalar_at_matches_execute_scalar() -> VortexResult<()> {
    let arr = sorted_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut rt = PointRuntime::new(&mut ctx);

    for i in 0..arr.len() {
        let via_runtime = rt.scalar_at(&arr, i)?;
        let mut ctx2 = LEGACY_SESSION.create_execution_ctx();
        let via_legacy = arr.execute_scalar(i, &mut ctx2)?;
        assert_eq!(via_runtime, via_legacy, "mismatch at idx {i}");
    }
    Ok(())
}

#[test]
fn session_scalar_at_caches_repeated_lookups() -> VortexResult<()> {
    let arr = sorted_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut session = PointSession::new(&mut ctx);

    let first = session.scalar_at(&arr, 3)?;
    assert_eq!(session.scalar_cache_len(), 1);

    let again = session.scalar_at(&arr, 3)?;
    assert_eq!(session.scalar_cache_len(), 1, "no new entry on cache hit");
    assert_eq!(first, again);

    let _scalar = session.scalar_at(&arr, 7)?;
    assert_eq!(session.scalar_cache_len(), 2);
    Ok(())
}

#[test]
fn session_scalar_cache_evicts_when_full() -> VortexResult<()> {
    let arr = sorted_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut session = PointSession::with_capacities(&mut ctx, 3, 8);

    for i in 0..5 {
        let _scalar = session.scalar_at(&arr, i)?;
    }
    assert_eq!(
        session.scalar_cache_len(),
        3,
        "FIFO eviction caps at capacity"
    );
    Ok(())
}

#[test]
fn runtime_search_sorted_matches_legacy_left() -> VortexResult<()> {
    let arr = sorted_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut rt = PointRuntime::new(&mut ctx);

    // Spot-check a few values: exact hit, in-gap, before-start, after-end.
    let cases = [
        (
            Scalar::from(2i32),
            SearchSortedSide::Left,
            SearchResult::Found(2),
        ),
        (
            Scalar::from(2i32),
            SearchSortedSide::Right,
            SearchResult::Found(5),
        ),
        (
            Scalar::from(4i32),
            SearchSortedSide::Left,
            SearchResult::NotFound(6),
        ),
        (
            Scalar::from(-1i32),
            SearchSortedSide::Left,
            SearchResult::NotFound(0),
        ),
        (
            Scalar::from(200i32),
            SearchSortedSide::Left,
            SearchResult::NotFound(15),
        ),
    ];

    for (target, side, expected) in cases {
        let got = rt.search_sorted(&arr, &target, side)?;
        assert_eq!(got, expected, "target={target:?} side={side:?}");
    }
    Ok(())
}

#[test]
fn session_search_sorted_hits_scalar_cache() -> VortexResult<()> {
    let arr = sorted_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut session = PointSession::new(&mut ctx);

    // search for an exact value; the binary search probes ~log2(15) ≈ 4 unique
    // indices, plus the side-refinement may revisit some — those revisits should
    // be cache hits.
    let _result = session.search_sorted(&arr, &Scalar::from(2i32), SearchSortedSide::Left)?;

    // After one search, the cache should hold at most as many entries as unique
    // indices probed, capped by the default capacity.
    let len_after_search = session.scalar_cache_len();
    assert!(len_after_search > 0, "search populated the cache");

    // A second identical search should re-use everything from the cache:
    // cache size should not grow.
    let _result = session.search_sorted(&arr, &Scalar::from(2i32), SearchSortedSide::Left)?;
    assert_eq!(
        session.scalar_cache_len(),
        len_after_search,
        "second identical search reused cache entirely"
    );
    Ok(())
}

#[test]
fn runtime_cached_block_is_noop() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut rt = PointRuntime::new(&mut ctx);

    let key = (0usize, BlockKey::new(0, 0));
    // Each call to runtime.cached_block re-runs the closure. Two calls = two runs.
    let mut runs = 0usize;
    let _: i32 = rt.cached_block(key, || {
        runs += 1;
        Ok(42)
    })?;
    let _: i32 = rt.cached_block(key, || {
        runs += 1;
        Ok(42)
    })?;
    assert_eq!(runs, 2, "runtime never caches");
    Ok(())
}

#[test]
fn session_cached_block_decodes_once_per_key() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut session = PointSession::new(&mut ctx);

    let key = (12345usize, BlockKey::new(7, 3));
    let mut runs = 0usize;
    let v1: i32 = session.cached_block(key, || {
        runs += 1;
        Ok(99)
    })?;
    let v2: i32 = session.cached_block(key, || {
        runs += 1;
        Ok(99)
    })?;
    assert_eq!(runs, 1, "session decodes once and caches");
    assert_eq!(v1, 99);
    assert_eq!(v2, 99);
    assert_eq!(session.block_cache_len(), 1);
    Ok(())
}

#[test]
fn slice_recurses_through_dispatch() -> VortexResult<()> {
    use crate::arrays::SliceArray;

    let inner = sorted_primitive();
    // slice covers indices 4..10 of the inner array
    let slice = SliceArray::new(inner, 4..10).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut session = PointSession::new(&mut ctx);

    // scalar_at on the slice at local idx 0 should equal inner's idx 4 (= 2).
    let v = session.scalar_at(&slice, 0)?;
    assert_eq!(v, Scalar::from(2i32));

    // After the slice scalar_at, the cache should hold entries at BOTH levels:
    // one for the slice array and one for the inner array (the recursion target).
    assert_eq!(
        session.scalar_cache_len(),
        2,
        "recursion populated caches at slice and inner levels"
    );

    // Re-fetching slice[0] should hit the slice-level cache (no recursion).
    let v2 = session.scalar_at(&slice, 0)?;
    assert_eq!(v2, Scalar::from(2i32));
    assert_eq!(session.scalar_cache_len(), 2, "no new entries");
    Ok(())
}

#[test]
fn slice_search_sorted_recurses_correctly() -> VortexResult<()> {
    use crate::arrays::SliceArray;

    let inner = sorted_primitive();
    // inner: [0, 1, 2, 2, 2, 3, 5, 5, 8, 13, 21, 34, 55, 89, 144]
    // slice covers indices 5..12 → [3, 5, 5, 8, 13, 21, 34]
    let slice = SliceArray::new(inner, 5..12).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut rt = PointRuntime::new(&mut ctx);

    // Search for 5 (Left side) — should return Found(1) in slice-local coords.
    let r = rt.search_sorted(&slice, &Scalar::from(5i32), SearchSortedSide::Left)?;
    assert_eq!(r, SearchResult::Found(1));

    // Search for 100 — should return NotFound(7) (past end of slice).
    let r = rt.search_sorted(&slice, &Scalar::from(100i32), SearchSortedSide::Left)?;
    assert_eq!(r, SearchResult::NotFound(7));
    Ok(())
}

#[test]
fn session_cached_block_evicts_oldest() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut session = PointSession::with_capacities(&mut ctx, 8, 2);

    let k1 = (1usize, BlockKey::new(0, 1));
    let k2 = (1usize, BlockKey::new(0, 2));
    let k3 = (1usize, BlockKey::new(0, 3));

    let _: i32 = session.cached_block(k1, || Ok(1))?;
    let _: i32 = session.cached_block(k2, || Ok(2))?;
    assert_eq!(session.block_cache_len(), 2);

    // Insert third: evicts k1 (oldest).
    let _: i32 = session.cached_block(k3, || Ok(3))?;
    assert_eq!(session.block_cache_len(), 2);

    // k1 should now be a miss → re-decode.
    let mut k1_runs = 0usize;
    let _: i32 = session.cached_block(k1, || {
        k1_runs += 1;
        Ok(1)
    })?;
    assert_eq!(k1_runs, 1, "k1 was evicted; re-decoded");
    Ok(())
}
