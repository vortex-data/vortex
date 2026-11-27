// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::get_item;
use vortex_expr::gt;
use vortex_expr::lit;
use vortex_expr::root;

use super::*;
use crate::plan::LazyPlanRef;
use crate::test::NoSegments;
use crate::test::SESSION;
use crate::test::Store;

fn struct_input(store: &Arc<Store>, rows: usize) -> VortexResult<ScanInput> {
    let ints = PrimitiveArray::from_iter(0..i32::try_from(rows)?).into_array();
    let flags = BoolArray::from_iter((0..rows).map(|i| i % 2 == 0)).into_array();
    let array = StructArray::from_fields(&[("ints", ints), ("flags", flags)])?.into_array();
    let plan = store.flat(&array)?;
    Ok(store.input(plan))
}

#[test]
fn builder_dtype_tracks_projection_and_row_range_clamps_inverted_bounds() -> VortexResult<()> {
    let store = Arc::new(Store::default());
    let input = struct_input(&store, 10)?;
    let root_dtype = input.dtype.clone();
    let builder = ScanBuilder::new(SESSION.clone(), input);
    assert_eq!(builder.dtype(), &root_dtype);

    let projection = get_item("ints", root()).bind(&root_dtype)?;
    let builder = builder
        .with_projection(projection.clone())
        .with_row_range(Range { start: 8, end: 3 })
        .with_row_offset(42)
        .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![1u64, 4]))?);
    assert_eq!(builder.dtype(), projection.dtype());

    let prepared = builder.prepare()?;
    assert_eq!(prepared.rows(), &(8..8));
    assert_eq!(prepared.row_offset, 42);
    assert!(matches!(
        prepared.selection(),
        Selection::IncludeByIndex(indices) if indices.as_slice() == [1, 4]
    ));
    Ok(())
}

#[test]
fn prepare_defaults_row_range_to_full_root_and_caches_lazy_root() -> VortexResult<()> {
    let store = Arc::new(Store::default());
    let base = struct_input(&store, 12)?;
    let plan = base.root.get()?.clone();
    let inits = Arc::new(AtomicUsize::new(0));
    let cache = ScanCache::default();

    let make_input = || {
        let plan = plan.clone();
        let inits = Arc::clone(&inits);
        ScanInput {
            dtype: base.dtype.clone(),
            root: LazyPlanRef::lazy(move || {
                inits.fetch_add(1, Ordering::SeqCst);
                Ok(plan.clone())
            }),
            locations: Arc::clone(&base.locations),
            segments: Arc::new(NoSegments),
            cache: Some(cache.clone()),
        }
    };

    let first = ScanBuilder::new(SESSION.clone(), make_input()).prepare()?;
    let second = ScanBuilder::new(SESSION.clone(), make_input()).prepare()?;
    assert_eq!(first.rows(), &(0..12));
    assert_eq!(second.rows(), &(0..12));
    assert_eq!(inits.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn prepare_propagates_lazy_root_and_dtype_mismatch_errors() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let failing_input = ScanInput {
        dtype,
        root: LazyPlanRef::lazy(|| Err(vortex_err!("corrupt footer"))),
        locations: Arc::new([]),
        segments: Arc::new(NoSegments),
        cache: None,
    };
    let err = ScanBuilder::new(SESSION.clone(), failing_input)
        .prepare()
        .err()
        .ok_or_else(|| vortex_err!("failing lazy root must error"))?;
    assert!(err.to_string().contains("corrupt footer"), "{err}");

    let store = Arc::new(Store::default());
    let input = struct_input(&store, 4)?;
    let other_struct = DType::struct_(
        [(
            "missing",
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )],
        Nullability::NonNullable,
    );
    let wrong_proj = get_item("missing", root()).bind(&other_struct)?;
    let err = ScanBuilder::new(SESSION.clone(), input.clone())
        .with_projection(wrong_proj)
        .prepare()
        .err()
        .ok_or_else(|| vortex_err!("mismatched projection field must error"))?;
    assert!(
        err.to_string()
            .contains("Couldn't find the missing field in the input scope"),
        "{err}"
    );

    let wrong_dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let wrong_filter = gt(root(), lit(0i64)).bind(&wrong_dtype)?;
    let err = ScanBuilder::new(SESSION.clone(), input)
        .with_some_filter(Some(wrong_filter))
        .prepare()
        .err()
        .ok_or_else(|| vortex_err!("mismatched filter dtype must error"))?;
    assert!(
        err.to_string()
            .contains("Comparison requires equal dtypes ignoring nullability"),
        "{err}"
    );
    Ok(())
}

#[test]
fn filter_cache_respects_skip_unselective_dynamic_flag() -> VortexResult<()> {
    let store = Arc::new(Store::default());
    let mut input = struct_input(&store, 10)?;
    let cache = ScanCache::default();
    input.cache = Some(cache.clone());
    let filter = gt(get_item("ints", root()), lit(5i32)).bind(&input.dtype)?;

    let first = ScanBuilder::new(SESSION.clone(), input.clone())
        .with_some_filter(Some(filter.clone()))
        .with_skip_unselective_dynamic(false)
        .prepare()?;
    let second = ScanBuilder::new(SESSION.clone(), input.clone())
        .with_some_filter(Some(filter.clone()))
        .with_skip_unselective_dynamic(false)
        .prepare()?;
    let skipping = ScanBuilder::new(SESSION.clone(), input)
        .with_some_filter(Some(filter.clone()))
        .with_skip_unselective_dynamic(true)
        .prepare()?;

    let first_plans = first
        .filter
        .as_ref()
        .ok_or_else(|| vortex_err!("missing first filter plans"))?;
    let second_plans = second
        .filter
        .as_ref()
        .ok_or_else(|| vortex_err!("missing second filter plans"))?;
    let skipping_plans = skipping
        .filter
        .as_ref()
        .ok_or_else(|| vortex_err!("missing skipping filter plans"))?;
    assert!(Arc::ptr_eq(first_plans, second_plans));
    assert!(!Arc::ptr_eq(first_plans, skipping_plans));
    assert!(!first_plans.skips_unselective_dynamic());
    assert!(skipping_plans.skips_unselective_dynamic());
    let cached_strict = cache
        .filter_plans(&filter, false)
        .ok_or_else(|| vortex_err!("cached strict filter entry missing"))?;
    assert!(Arc::ptr_eq(&cached_strict, first_plans));
    let cached_skipping = cache
        .filter_plans(&filter, true)
        .ok_or_else(|| vortex_err!("cached skipping filter entry missing"))?;
    assert!(Arc::ptr_eq(&cached_skipping, skipping_plans));
    Ok(())
}

#[test]
fn unfiltered_chunk_rows_scales_only_for_identity_projection() -> VortexResult<()> {
    let dtype = DType::struct_(
        [(
            "ints",
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )],
        Nullability::NonNullable,
    );
    let root_plan = crate::test::struct_plan(
        1_000_000,
        dtype.clone(),
        vec![crate::test::flat(
            1_000_000,
            DType::Primitive(PType::I32, Nullability::NonNullable),
            0,
        )],
    );
    // 384 bytes/row shrinks identity chunk sizing to 16_384 rows.
    let locations: Arc<[SegmentLocation]> = Arc::from([SegmentLocation {
        offset: 0,
        length: 384_000_000,
        alignment: Alignment::none(),
    }]);
    let make_input = || ScanInput {
        dtype: dtype.clone(),
        root: LazyPlanRef::ready(root_plan.clone()),
        locations: Arc::clone(&locations),
        segments: Arc::new(NoSegments),
        cache: None,
    };

    let identity = ScanBuilder::new(SESSION.clone(), make_input()).prepare()?;
    assert_eq!(identity.chunk_rows, 16_384);

    let projected = ScanBuilder::new(SESSION.clone(), make_input())
        .with_projection(get_item("ints", root()).bind(&dtype)?)
        .prepare()?;
    assert_eq!(projected.chunk_rows, MAX_ROWS_PER_CHUNK);
    Ok(())
}

#[test]
fn filtered_rows_per_chunk_edge_cases() -> VortexResult<()> {
    let store = Arc::new(Store::default());
    let input = struct_input(&store, 4)?;
    let cases = [
        (
            get_item("flags", root()),
            get_item("flags", root()),
            128 * ROW_BLOCK_SIZE,
        ),
        (
            get_item("ints", root()),
            get_item("flags", root()),
            FILTERED_ROWS_PER_CHUNK,
        ),
        (lit(1i32), lit(true), FILTERED_ROWS_PER_CHUNK),
    ];
    for (projection, filter, expected) in cases {
        let actual = filtered_rows_per_chunk(
            &input.dtype,
            &projection.bind(&input.dtype)?,
            &filter.bind(&input.dtype)?,
        )?;
        assert_eq!(actual, expected);
    }
    Ok(())
}
