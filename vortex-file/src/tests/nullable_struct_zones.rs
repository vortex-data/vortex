// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File round trips for the zone maps of fields beneath a nullable struct.

use std::sync::Arc;

use futures::TryStreamExt;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::FieldNames;
use vortex_array::expr::Expression;
use vortex_array::expr::fill_null;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::zoned::Zoned;
use vortex_layout::scan::metrics::ScanUsage;
use vortex_layout::scan::v2;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_scan::execution::driver::ScanDriver;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;
use crate::strategy::WriteStrategyBuilder;

/// Scans `file` on the classic or the planner-driven executor, returning the rows that match
/// `filter` projected by `projection`, and the number of segments the scan read.
async fn scan_counting_segments(
    file: &VortexFile,
    planner: bool,
    projection: &Expression,
    filter: &Expression,
) -> VortexResult<(ArrayRef, u64)> {
    let usage = Arc::new(ScanUsage::default());
    let file = file.clone().with_usage(Arc::clone(&usage));
    let projection = projection.bind(file.dtype())?;
    let filter = filter.bind(file.dtype())?;
    let dtype = projection.dtype().clone();
    let array = if planner {
        let scan = v2::ScanBuilder::new(file.session().clone(), file.scan_file())
            .with_projection(projection)
            .with_some_filter(Some(filter));
        let chunks: Vec<ArrayRef> = ScanDriver::new(file.session().handle(), 1, true)
            .run(scan)
            .try_collect()
            .await?;
        ChunkedArray::try_new(chunks, dtype)?.into_array()
    } else {
        file.scan()?
            .with_projection(projection)
            .with_filter(filter)
            .into_array_stream()?
            .read_all()
            .await?
    };
    Ok((array, usage.snapshot().segments_requested))
}

/// A float field beneath a nullable struct, written with the default editions only, records its
/// bounds and its NaN count, which float bounds need to prune and every core edition permits.
/// Filters on it return every row they keep in memory on both scans.
///
/// Where `s` is null, `f` holds a value above every visible one, so the bounds of every zone are
/// looser than the values read through `s`. Neither `s.f > 1e8` nor `s.f < 0` keeps a row, but
/// only the second is ruled out by the bounds, so it reads fewer segments.
#[rstest]
#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn float_beneath_a_nullable_struct_prunes_with_default_editions(
    #[values(false, true)] planner: bool,
) -> VortexResult<()> {
    const ZONE_ROWS: u32 = 1024;
    let session = VortexSession::empty()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with_tokio();
    crate::register_default_encodings(&session);
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    session.enable_edition(CORE_2026_08_3)?;

    let rows = 0..4 * ZONE_ROWS;
    let valid = |row: u32| !row.is_multiple_of(3);
    let f: Buffer<f64> = rows
        .clone()
        .map(|row| {
            if valid(row) {
                f64::from(row) + 0.5
            } else {
                1e9
            }
        })
        .collect();
    let s = StructArray::try_new(
        FieldNames::from(["f"]),
        vec![f.into_array()],
        rows.len(),
        Validity::from_iter(rows.clone().map(valid)),
    )?;
    let ids: Buffer<i64> = rows.map(i64::from).collect();
    let table =
        StructArray::from_fields(&[("id", ids.into_array()), ("s", s.into_array())])?.into_array();

    let strategy = WriteStrategyBuilder::from_session(&session)
        .with_row_block_size(usize::try_from(ZONE_ROWS)?)
        .with_data_block_target_bytes(None)
        .build();
    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(strategy)
        .write(&mut buffer, table.clone().to_array_stream())
        .await?;
    let file = session.open_options().open_buffer(buffer)?;

    let mut recorded = Vec::new();
    for layout in file.footer().layout().depth_first_traversal() {
        let layout = layout?;
        if layout.is::<Zoned>() && layout.dtype().is_float() {
            recorded.push(layout.as_::<Zoned>().present_aggregates());
        }
    }
    let expected: Arc<[String]> = [
        Max.bind(NumericalAggregateOpts::skip_nans()).to_string(),
        Min.bind(NumericalAggregateOpts::skip_nans()).to_string(),
        NanCount.bind(EmptyOptions).to_string(),
    ]
    .into();
    assert_eq!(recorded, [expected]);

    let id = get_item("id", root());
    let f = get_item("f", get_item("s", root()));
    let mut ctx = session.create_execution_ctx();
    let mut segments = Vec::new();
    for filter in [
        lt(f.clone(), lit(100.0f64)),
        gt(f.clone(), lit(1e8f64)),
        lt(f, lit(0.0f64)),
    ] {
        let (actual, read) = scan_counting_segments(&file, planner, &id, &filter).await?;
        let keep = table
            .clone()
            .apply(&fill_null(filter, lit(false)))?
            .execute::<Mask>(&mut ctx)?;
        let expected = table.clone().filter(keep)?.apply(&id)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        segments.push(read);
    }
    assert!(segments[2] < segments[1], "segments read: {segments:?}");
    Ok(())
}
