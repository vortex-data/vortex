// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Barrier;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;

use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_error::vortex_err;
use vortex_session::registry::ReadContext;

use super::*;
use crate::plan::physical::SegmentScanPlan;
use crate::segments::SegmentId;

fn plan() -> PlanRef {
    SegmentScanPlan::new(
        PType::I32.into(),
        10,
        SegmentId::from(0),
        ReadContext::new(ArrayContext::empty().to_ids()),
        None,
    )
    .into_plan()
}

#[test]
fn a_finished_later_segment_does_not_wait_for_an_earlier_chunk() -> VortexResult<()> {
    let cache = SegmentCache::new(0..30, true);
    let first = cache.lease(0..10);
    let second = cache.lease(10..20);
    let second_projection = Arc::clone(&second);
    let reader = cache.reader(plan(), 10..20);
    reader.get_or_try_init(|| Ok(PrimitiveArray::from_iter(10i32..20).into_array()))?;
    drop(second);
    assert!(reader.get().is_some());
    drop(second_projection);
    assert!(reader.get().is_none());
    assert_eq!(
        cache.shared.state.lock().pending,
        BTreeMap::from([(0, 10), (20, 30)])
    );
    drop(first);
    Ok(())
}

#[test]
fn an_entry_survives_until_all_overlapping_consumers_retire() -> VortexResult<()> {
    let cache = SegmentCache::new(2..38, true);
    let reader = cache.reader(plan(), 10..20);
    let array = PrimitiveArray::from_iter(10i32..20).into_array();
    let retained = reader.get_or_try_init(|| Ok(array.clone()))?;
    assert!(ArrayRef::ptr_eq(&retained, &array));
    cache.complete(12..38);
    assert!(reader.get().is_some());
    cache.complete(2..10);
    assert!(reader.get().is_some());
    cache.complete(10..12);
    assert!(reader.get().is_none());
    assert!(cache.shared.state.lock().pending.is_empty());
    reader.get_or_try_init(|| Ok(array))?;
    assert!(
        reader.get().is_none(),
        "retired rows cannot repopulate the cache"
    );
    Ok(())
}

#[test]
fn cancellation_can_retire_a_range_containing_completed_holes() {
    let cache = SegmentCache::new(0..100, true);
    cache.complete(10..20);
    cache.complete(40..60);
    cache.complete(5..90);
    assert_eq!(
        cache.shared.state.lock().pending,
        BTreeMap::from([(0, 5), (90, 100)])
    );
    cache.complete(0..100);
    assert!(cache.shared.state.lock().pending.is_empty());
}

#[test]
fn plan_identity_and_row_domain_isolate_entries() -> VortexResult<()> {
    let cache = SegmentCache::new(0..20, true);
    let plan = plan();
    let reader = cache.reader(plan.clone(), 0..10);
    let first = reader.get_or_try_init(|| Ok(PrimitiveArray::from_iter(0i32..10).into_array()))?;
    let winner = reader.get_or_try_init(|| Err(vortex_err!("must reuse the first array")))?;
    assert!(ArrayRef::ptr_eq(&winner, &first));
    assert!(cache.reader(plan, 10..20).get().is_none());
    assert!(cache.reader(self::plan(), 0..10).get().is_none());
    Ok(())
}

#[test]
fn concurrent_readers_initialize_a_segment_once() -> VortexResult<()> {
    let cache = SegmentCache::new(0..10, true);
    let plan = plan();
    let array = PrimitiveArray::from_iter(0i32..10).into_array();
    let starts = Barrier::new(4);
    let initializations = AtomicUsize::new(0);
    let arrays = thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let reader = cache.reader(plan.clone(), 0..10);
                let (starts, initializations, array) = (&starts, &initializations, &array);
                scope.spawn(move || {
                    starts.wait();
                    reader.get_or_try_init(|| {
                        initializations.fetch_add(1, Ordering::Relaxed);
                        Ok(array.clone())
                    })
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| vortex_err!("segment initialization panicked"))?
            })
            .collect::<VortexResult<Vec<_>>>()
    })?;
    assert_eq!(initializations.load(Ordering::Relaxed), 1);
    assert!(arrays.iter().all(|result| ArrayRef::ptr_eq(result, &array)));
    Ok(())
}

#[test]
fn a_failed_initialization_can_be_retried() -> VortexResult<()> {
    let cache = SegmentCache::new(0..10, true);
    let reader = cache.reader(plan(), 0..10);
    assert!(
        reader
            .get_or_try_init(|| Err(vortex_err!("invalid segment")))
            .is_err()
    );
    assert!(reader.get().is_none());
    let array = PrimitiveArray::from_iter(0i32..10).into_array();
    let result = reader.get_or_try_init(|| Ok(array.clone()))?;
    assert!(ArrayRef::ptr_eq(&result, &array));
    assert!(reader.get().is_some());
    Ok(())
}

#[test]
fn retirement_during_initialization_does_not_repopulate_the_cache() -> VortexResult<()> {
    let cache = SegmentCache::new(0..10, true);
    let reader = cache.reader(plan(), 0..10);
    let array = PrimitiveArray::from_iter(0i32..10).into_array();
    let result = reader.get_or_try_init(|| {
        // This also requires initialization to release the global cache lock.
        cache.complete(0..10);
        Ok(array.clone())
    })?;
    assert!(ArrayRef::ptr_eq(&result, &array));
    assert!(reader.get().is_none());
    assert!(cache.shared.state.lock().arrays.is_empty());
    Ok(())
}
