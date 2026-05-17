// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for the `point_fn` module: PointRuntime / PointSession behavior,
//! generic_search_sorted, and per-encoding override correctness.

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

/// A non-FAST array suitable for testing the scalar cache. Built as a Dict
/// over a primitive: Dict's scalar_at is non-trivial (read code, read dict)
/// so the session does cache it.
fn cached_array() -> crate::ArrayRef {
    use crate::arrays::DictArray;
    let dict = PrimitiveArray::new(
        buffer![10i32, 20, 30, 40, 50],
        crate::validity::Validity::NonNullable,
    )
    .into_array();
    let codes = PrimitiveArray::new(
        buffer![0u32, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        crate::validity::Validity::NonNullable,
    )
    .into_array();
    DictArray::try_new(codes, dict).unwrap().into_array()
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
fn session_bypasses_cache_for_fast_leaves() -> VortexResult<()> {
    // Primitive has FAST_SCALAR_AT = true, so the session never inserts.
    let arr = sorted_primitive();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut session = PointSession::new(&mut ctx);

    let first = session.scalar_at(&arr, 3)?;
    assert_eq!(
        session.scalar_cache_len(),
        0,
        "fast leaf bypasses scalar cache"
    );

    let again = session.scalar_at(&arr, 3)?;
    assert_eq!(first, again, "values agree across calls (no cache needed)");
    assert_eq!(session.scalar_cache_len(), 0);
    Ok(())
}

#[test]
fn session_scalar_at_caches_repeated_lookups() -> VortexResult<()> {
    // Dict has FAST_SCALAR_AT = false (default), so the session does cache.
    let arr = cached_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut session = PointSession::new(&mut ctx);

    let first = session.scalar_at(&arr, 3)?;
    assert!(session.scalar_cache_len() >= 1);

    let cache_after_first = session.scalar_cache_len();
    let again = session.scalar_at(&arr, 3)?;
    assert_eq!(
        session.scalar_cache_len(),
        cache_after_first,
        "no new entry on cache hit"
    );
    assert_eq!(first, again);
    Ok(())
}

#[test]
fn session_scalar_cache_evicts_when_full() -> VortexResult<()> {
    // Use Dict so caching is exercised. Note Dict's point_scalar_at also
    // caches the recursive codes/values reads, so we look at the dict
    // array's own cache entries only (its addr is what the loop reuses).
    let arr = cached_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    // Capacity 3 across both dict-level entries and the inner non-fast
    // recursions. Verify the cache never grows past 3.
    let mut session = PointSession::with_capacities(&mut ctx, 3, 8);

    for i in 0..arr.len() {
        let _scalar = session.scalar_at(&arr, i)?;
    }
    assert!(
        session.scalar_cache_len() <= 3,
        "FIFO eviction caps at capacity ({} > 3)",
        session.scalar_cache_len()
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
fn session_caches_non_fast_scalar_at_calls() -> VortexResult<()> {
    let arr = cached_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut session = PointSession::new(&mut ctx);

    // Direct scalar_at on the Dict (non-FAST) populates the cache.
    let first = session.scalar_at(&arr, 5)?;
    let len_after_first = session.scalar_cache_len();
    assert!(
        len_after_first >= 1,
        "Dict scalar_at populates the scalar cache"
    );

    // Same index again — cache hit, no new entry.
    let again = session.scalar_at(&arr, 5)?;
    assert_eq!(first, again);
    assert_eq!(
        session.scalar_cache_len(),
        len_after_first,
        "cache hit produces no new entry"
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

    // Both Slice and Primitive are FAST_SCALAR_AT — the session bypasses the
    // scalar cache at every level, avoiding redundant wrapper-level entries.
    assert_eq!(
        session.scalar_cache_len(),
        0,
        "fast wrapper over fast leaf skips both cache levels"
    );

    // The recursion still produces the right value on a second call.
    let v2 = session.scalar_at(&slice, 0)?;
    assert_eq!(v2, Scalar::from(2i32));
    assert_eq!(session.scalar_cache_len(), 0);
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
fn constant_search_sorted_o1() -> VortexResult<()> {
    use crate::arrays::ConstantArray;

    // Constant array of 1000 copies of 42.
    let arr = ConstantArray::new(Scalar::from(42i32), 1000).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut rt = PointRuntime::new(&mut ctx);

    // Found semantics — Left → 0, Right → len.
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(42i32), SearchSortedSide::Left)?,
        SearchResult::Found(0),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(42i32), SearchSortedSide::Right)?,
        SearchResult::Found(1000),
    );
    // Less than constant: goes before everything.
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(0i32), SearchSortedSide::Left)?,
        SearchResult::NotFound(0),
    );
    // Greater than constant: goes at the end.
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(100i32), SearchSortedSide::Left)?,
        SearchResult::NotFound(1000),
    );
    Ok(())
}

#[test]
fn slice_search_sorted_clamps_to_slice_bounds() -> VortexResult<()> {
    use crate::arrays::SliceArray;

    let inner = sorted_primitive();
    // inner: [0, 1, 2, 2, 2, 3, 5, 5, 8, 13, 21, 34, 55, 89, 144]
    // slice covers indices 5..12 → [3, 5, 5, 8, 13, 21, 34]
    let slice = SliceArray::new(inner, 5..12).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut rt = PointRuntime::new(&mut ctx);

    // Value 1 is in the inner array but before the slice — NotFound(0).
    let r = rt.search_sorted(&slice, &Scalar::from(1i32), SearchSortedSide::Left)?;
    assert_eq!(r, SearchResult::NotFound(0));

    // Value 144 is the last in the inner array, after the slice — NotFound(len).
    let r = rt.search_sorted(&slice, &Scalar::from(144i32), SearchSortedSide::Left)?;
    assert_eq!(r, SearchResult::NotFound(7));

    // Value 8 is in the slice at local index 3.
    let r = rt.search_sorted(&slice, &Scalar::from(8i32), SearchSortedSide::Left)?;
    assert_eq!(r, SearchResult::Found(3));

    Ok(())
}

#[test]
fn dict_search_sorted_through_dispatch() -> VortexResult<()> {
    use crate::arrays::DictArray;

    // Sorted dict: [10, 20, 30, 40], sorted codes: [0, 0, 1, 1, 2, 3, 3, 3]
    // Logical (sorted): [10, 10, 20, 20, 30, 40, 40, 40]
    let dict = PrimitiveArray::new(
        vortex_buffer::buffer![10i32, 20, 30, 40],
        crate::validity::Validity::NonNullable,
    )
    .into_array();
    let codes = PrimitiveArray::new(
        vortex_buffer::buffer![0u32, 0, 1, 1, 2, 3, 3, 3],
        crate::validity::Validity::NonNullable,
    )
    .into_array();
    let arr = DictArray::try_new(codes, dict)?.into_array();

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut rt = PointRuntime::new(&mut ctx);

    // Found cases.
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(10i32), SearchSortedSide::Left)?,
        SearchResult::Found(0),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(20i32), SearchSortedSide::Left)?,
        SearchResult::Found(2),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(20i32), SearchSortedSide::Right)?,
        SearchResult::Found(4),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(40i32), SearchSortedSide::Right)?,
        SearchResult::Found(8),
    );

    // NotFound cases.
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(5i32), SearchSortedSide::Left)?,
        SearchResult::NotFound(0),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(25i32), SearchSortedSide::Left)?,
        SearchResult::NotFound(4),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(50i32), SearchSortedSide::Left)?,
        SearchResult::NotFound(8),
    );

    Ok(())
}

#[test]
fn chunked_search_sorted_routes_to_chunk() -> VortexResult<()> {
    use crate::arrays::ChunkedArray;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    // Three chunks, cross-chunk monotonic ascending:
    //   chunk 0: [1, 3, 5]
    //   chunk 1: [7, 9, 11]
    //   chunk 2: [13, 15, 17]
    let chunks = vec![
        PrimitiveArray::new(
            vortex_buffer::buffer![1i32, 3, 5],
            crate::validity::Validity::NonNullable,
        )
        .into_array(),
        PrimitiveArray::new(
            vortex_buffer::buffer![7i32, 9, 11],
            crate::validity::Validity::NonNullable,
        )
        .into_array(),
        PrimitiveArray::new(
            vortex_buffer::buffer![13i32, 15, 17],
            crate::validity::Validity::NonNullable,
        )
        .into_array(),
    ];
    let arr = ChunkedArray::try_new(
        chunks,
        crate::dtype::DType::Primitive(PType::I32, Nullability::NonNullable),
    )?
    .into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let mut rt = PointRuntime::new(&mut ctx);

    // Found cases (one per chunk).
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(3i32), SearchSortedSide::Left)?,
        SearchResult::Found(1),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(7i32), SearchSortedSide::Left)?,
        SearchResult::Found(3),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(15i32), SearchSortedSide::Left)?,
        SearchResult::Found(7),
    );

    // NotFound — values that fall in gaps.
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(0i32), SearchSortedSide::Left)?,
        SearchResult::NotFound(0),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(4i32), SearchSortedSide::Left)?,
        SearchResult::NotFound(2),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(6i32), SearchSortedSide::Left)?,
        SearchResult::NotFound(3),
    );
    assert_eq!(
        rt.search_sorted(&arr, &Scalar::from(20i32), SearchSortedSide::Left)?,
        SearchResult::NotFound(9),
    );

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
