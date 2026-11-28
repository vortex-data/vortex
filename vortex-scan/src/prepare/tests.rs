// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::TryStreamExt;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
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
use vortex_io::session::RuntimeSessionExt;

use super::*;
use crate::execution::driver::ScanDriver;
use crate::plan::LazyPlanRef;
use crate::row_idx::row_idx;
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

#[tokio::test]
async fn builder_dtype_tracks_projection_and_row_range_clamps_inverted_bounds() -> VortexResult<()>
{
    let store = Arc::new(Store::default());
    let input = struct_input(&store, 10)?;
    let root_dtype = input.dtype.clone();
    let builder = ScanBuilder::new(SESSION.clone(), input.clone());
    assert_eq!(builder.dtype(), &root_dtype);

    let projection = get_item("ints", root()).bind(&root_dtype)?;
    let clamped = builder
        .with_projection(projection.clone())
        .with_row_range(Range { start: 8, end: 3 });
    assert_eq!(clamped.dtype(), projection.dtype());
    let prepared = clamped.prepare()?;
    assert_eq!(prepared.rows(), &(8..8));

    // Verify row_offset and selection end-to-end on a nonempty scan projecting `row_idx()`.
    let session = SESSION.clone().with_tokio();
    let row_idx_proj = row_idx().bind(&root_dtype)?;
    let nonempty = ScanBuilder::new(session.clone(), input)
        .with_projection(row_idx_proj.clone())
        .with_row_offset(42)
        .with_row_indices(StrictSortedBuffer::try_new(Buffer::from(vec![1u64, 4]))?);
    assert_eq!(nonempty.dtype(), row_idx_proj.dtype());
    let batches: Vec<_> = ScanDriver::new(session.handle(), 1, true)
        .run(nonempty)
        .try_collect()
        .await?;
    let actual = ChunkedArray::try_new(batches, row_idx_proj.dtype().clone())?.into_array();
    assert_arrays_eq!(
        actual,
        Buffer::from(vec![43u64, 46]).into_array(),
        &mut SESSION.create_execution_ctx()
    );
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
