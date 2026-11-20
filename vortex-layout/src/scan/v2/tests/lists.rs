// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! List layouts on the planner-driven scan, mirroring the layout reader tests of the list layout.

use std::ops::Range;
use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::expr::Expression;
use vortex_array::expr::Lambda;
use vortex_array::expr::checked_add;
use vortex_array::expr::get_item;
use vortex_array::expr::is_not_null;
use vortex_array::expr::is_null;
use vortex_array::expr::list_contains;
use vortex_array::expr::list_length;
use vortex_array::expr::list_transform;
use vortex_array::expr::lit;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::var;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::VortexSession;

use super::builder;
use super::collect_scan;
use super::write_chunks;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::list::writer::ListLayoutStrategy;
use crate::layouts::struct_::writer::StructStrategy;
use crate::segments::CountingSegmentSource;
use crate::segments::SegmentSource;
use crate::test::new_session;

/// Lists `[{a, 1}, null]`, null (hiding `[{c, i32::MAX}]`), `[]`, `[{null, 3}, {e, 4}]` and
/// `[{f, i32::MAX}]`, of nullable `{sku: utf8?, qty: i32}` elements, as the only chunk of a file
/// whatever row it starts at.
fn items(_start: usize) -> VortexResult<ArrayRef> {
    let mut ctx = new_session().create_execution_ctx();
    let elements = StructArray::try_new(
        FieldNames::from(["sku", "qty"]),
        vec![
            VarBinViewArray::from_iter_nullable_str([
                Some("a"),
                Some("b"),
                Some("c"),
                None,
                Some("e"),
                Some("f"),
            ])
            .into_array(),
            buffer![1i32, 2, i32::MAX, 3, 4, i32::MAX].into_array(),
        ],
        6,
        Validity::from_iter([true, false, true, true, true, true]),
    )?
    .into_array();
    Ok(ListArray::try_new(
        elements,
        buffer![0u32, 2, 3, 3, 5, 6].into_array(),
        Validity::from_iter([true, false, true, true, true]),
        &mut ctx,
    )?
    .into_array())
}

/// Writes struct elements as a struct layout, so the list has five segments: its offsets, its
/// validity, the element validity, `sku` and `qty`.
fn struct_elements_list_strategy() -> ListLayoutStrategy {
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    ListLayoutStrategy::default()
        .with_elements(Arc::new(StructStrategy::new(Arc::clone(&flat), flat)))
}

fn sku() -> VortexResult<Expression> {
    Ok(list_transform(
        root(),
        Lambda::new(["x"], get_item("sku", var("x")))?,
    ))
}

fn qty() -> VortexResult<Expression> {
    Ok(list_transform(
        root(),
        Lambda::new(["x"], get_item("qty", var("x")))?,
    ))
}

/// `list_transform(root, x -> x.qty + 1)`, which overflows on `i32::MAX`.
fn qty_plus_one() -> VortexResult<Expression> {
    Ok(list_transform(
        root(),
        Lambda::new(["x"], checked_add(get_item("qty", var("x")), lit(1i32)))?,
    ))
}

/// A list written with `strategy` from `list`, its segments, and a session for scanning it.
async fn write(
    strategy: &dyn LayoutStrategy,
    list: fn(usize) -> VortexResult<ArrayRef>,
) -> VortexResult<(VortexSession, Arc<dyn SegmentSource>, LayoutRef)> {
    let session = new_session().with_tokio();
    let (segments, layout) = write_chunks(&session, strategy, vec![list(0)?]).await?;
    Ok((session, segments, layout))
}

/// Scans `expr` over the rows of `row_range` that `mask` selects, filtered by `filter`, and counts
/// the segments the scan requests.
async fn scan_counting(
    strategy: &dyn LayoutStrategy,
    list: fn(usize) -> VortexResult<ArrayRef>,
    expr: &Expression,
    filter: Option<&Expression>,
    row_range: Range<u64>,
    mask: &Mask,
) -> VortexResult<(ArrayRef, usize)> {
    let (session, segments, layout) = write(strategy, list).await?;
    let source = Arc::new(CountingSegmentSource::new(segments));
    let segments: Arc<dyn SegmentSource> = Arc::<CountingSegmentSource>::clone(&source);
    let projection = expr.bind(layout.dtype())?;
    let dtype = projection.dtype().clone();
    let start = row_range.start;
    let rows = (0..mask.len())
        .filter(|&row| mask.value(row))
        .map(|row| start + row as u64);
    let scan = builder(&session, &segments, &layout)?
        .with_projection(projection)
        .with_some_filter(
            filter
                .map(|filter| filter.bind(layout.dtype()))
                .transpose()?,
        )
        .with_row_range(row_range)
        .with_row_indices(StrictSortedBuffer::try_new(Buffer::from_iter(rows))?);
    let result = collect_scan(dtype, scan).await?;
    // Execute so lazy results surface their errors before the count is read.
    let result = result
        .execute::<RecursiveCanonical>(&mut session.create_execution_ctx())?
        .0
        .into_array();
    Ok((result, source.requests()))
}

/// `expr` over the rows of `list` in `row_range` that `mask` selects.
fn expected(
    list: &ArrayRef,
    expr: &Expression,
    row_range: &Range<u64>,
    mask: &Mask,
) -> VortexResult<ArrayRef> {
    list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?
        .filter(mask.clone())?
        .apply(expr)
}

/// A projection reads only the list children it needs: the validity for a null check, the offsets
/// and the validity for the lengths, and for `list_transform` lambdas the element columns they
/// read, the element validity, the offsets and the validity. These are the reads of the layout
/// reader.
#[rstest]
#[case::root(root(), 5)]
#[case::is_null(is_null(root()), 1)]
#[case::length(list_length(root()), 2)]
#[case::sku(sku()?, 4)]
#[case::qty(qty()?, 4)]
#[case::both(
    pack([("skus", sku()?), ("qtys", qty()?)], Nullability::NonNullable),
    5
)]
#[case::with_length(
    pack([("skus", sku()?), ("len", list_length(root()))], Nullability::NonNullable),
    4
)]
#[case::predicate(list_contains(sku()?, lit("e")), 4)]
#[tokio::test]
async fn element_projection_reads_needed_fields(
    #[case] expr: Expression,
    #[case] segments: usize,
) -> VortexResult<()> {
    let (result, requests) = scan_counting(
        &struct_elements_list_strategy(),
        items,
        &expr,
        None,
        0..5,
        &Mask::new_true(5),
    )
    .await?;
    let expected = items(0)?.apply(&expr)?;
    assert_arrays_eq!(result, expected, &mut new_session().create_execution_ctx());
    assert_eq!(requests, segments);
    Ok(())
}

/// Lists and their pushed-down lambdas keep the semantics of `list_transform` for null lists,
/// empty lists and null elements over any rows: whole, a range, or a selection.
#[rstest]
#[case::full(0..5, Mask::new_true(5))]
#[case::subrange(1..4, Mask::new_true(3))]
#[case::sparse(0..5, Mask::from_iter([true, false, true, false, true]))]
#[case::null_list_only(1..2, Mask::new_true(1))]
#[case::empty_list_only(2..3, Mask::new_true(1))]
#[case::none_selected(0..5, Mask::new_false(5))]
#[case::empty_range(2..2, Mask::new_true(0))]
#[tokio::test]
async fn element_projection_round_trips(
    #[case] row_range: Range<u64>,
    #[case] mask: Mask,
    #[values(
        root(),
        sku()?,
        qty()?,
        list_length(root()),
        pack(
            [("skus", sku()?), ("valid", is_not_null(root()))],
            Nullability::NonNullable,
        )
    )]
    expr: Expression,
) -> VortexResult<()> {
    let expected = expected(&items(0)?, &expr, &row_range, &mask)?;
    let (result, _) = scan_counting(
        &struct_elements_list_strategy(),
        items,
        &expr,
        None,
        row_range,
        &mask,
    )
    .await?;
    assert_arrays_eq!(result, expected, &mut new_session().create_execution_ctx());
    Ok(())
}

/// A lambda that can fail runs only on the elements of the selected valid lists, as in memory:
/// the `i32::MAX` under the null list never reaches it, nor does the one in the last list when the
/// selection leaves that list out. Only `qty`, the element validity, the offsets and the validity
/// are read, and no element at all when no selected list is valid.
#[rstest]
#[case::full_without_last(0..4, Mask::new_true(4), 4)]
#[case::unselected_last(0..5, Mask::from_iter([true, true, true, true, false]), 4)]
#[case::sparse(0..5, Mask::from_iter([false, true, false, true, false]), 4)]
#[case::null_list_only(1..2, Mask::new_true(1), 2)]
#[tokio::test]
async fn fallible_lambda_runs_on_selected_valid_lists(
    #[case] row_range: Range<u64>,
    #[case] mask: Mask,
    #[case] segments: usize,
) -> VortexResult<()> {
    let expr = qty_plus_one()?;
    let expected = expected(&items(0)?, &expr, &row_range, &mask)?;
    let (result, requests) = scan_counting(
        &struct_elements_list_strategy(),
        items,
        &expr,
        None,
        row_range,
        &mask,
    )
    .await?;
    assert_arrays_eq!(result, expected, &mut new_session().create_execution_ctx());
    assert_eq!(requests, segments);
    Ok(())
}

/// The scan fails where the lambda fails in memory: on the `i32::MAX` of a selected valid list.
#[tokio::test]
async fn fallible_lambda_fails_on_selected_overflow() -> VortexResult<()> {
    let expr = qty_plus_one()?;
    let mut ctx = new_session().create_execution_ctx();
    assert!(
        items(0)?
            .apply(&expr)?
            .execute::<RecursiveCanonical>(&mut ctx)
            .is_err()
    );
    let scan = scan_counting(
        &struct_elements_list_strategy(),
        items,
        &expr,
        None,
        4..5,
        &Mask::new_true(1),
    )
    .await;
    assert!(scan.is_err());
    Ok(())
}

/// A filter through a pushed-down lambda reads the element column it tests and keeps the rows
/// the filter keeps in memory. The projection of the null checks shares the filter's read of the
/// validity.
#[rstest]
#[case::dense(Mask::new_true(5))]
#[case::sparse(Mask::from_iter([false, false, false, true, false]))]
#[tokio::test]
async fn filter_pushes_lambda(#[case] mask: Mask) -> VortexResult<()> {
    let filter = list_contains(sku()?, lit("e"));
    let projection = is_not_null(root());
    let (result, requests) = scan_counting(
        &struct_elements_list_strategy(),
        items,
        &projection,
        Some(&filter),
        0..5,
        &mask,
    )
    .await?;

    let list = items(0)?.filter(mask)?;
    let mut ctx = new_session().create_execution_ctx();
    let keep = list
        .clone()
        .apply(&filter)?
        .fill_null(false)?
        .execute::<Mask>(&mut ctx)?;
    assert_arrays_eq!(result, list.filter(keep)?.apply(&projection)?, &mut ctx);
    assert_eq!(requests, 4);
    Ok(())
}

/// `[[1, 2], null, []]`, `null`, `[[3]]`, `[]` and `[[], [4, 5, 6]]`, lists of nullable lists, as
/// the only chunk of a file whatever row it starts at.
fn nested(_start: usize) -> VortexResult<ArrayRef> {
    let mut ctx = new_session().create_execution_ctx();
    let inner = ListArray::try_new(
        buffer![1i64, 2, 3, 4, 5, 6].into_array(),
        buffer![0u32, 2, 2, 2, 3, 3, 6].into_array(),
        Validity::from_iter([true, false, true, true, true, true]),
        &mut ctx,
    )?
    .into_array();
    Ok(ListArray::try_new(
        inner,
        buffer![0u32, 3, 3, 4, 4, 6].into_array(),
        Validity::from_iter([true, false, true, true, true]),
        &mut ctx,
    )?
    .into_array())
}

/// A list of lists reads the inner lists of its selected lists, and their elements through their
/// own offsets.
#[rstest]
#[case::full(0..5, Mask::new_true(5))]
#[case::subrange(2..5, Mask::new_true(3))]
#[case::sparse(0..5, Mask::from_iter([true, false, false, false, true]))]
#[tokio::test]
async fn nested_lists_round_trip(
    #[case] row_range: Range<u64>,
    #[case] mask: Mask,
    #[values(
        root(),
        list_transform(root(), Lambda::new(["x"], list_length(var("x")))?)
    )]
    expr: Expression,
) -> VortexResult<()> {
    let chunked: Arc<dyn LayoutStrategy> =
        Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
    let strategy = ListLayoutStrategy::default()
        .with_elements(Arc::new(ListLayoutStrategy::default()))
        .with_offsets(Arc::clone(&chunked))
        .with_validity(chunked);
    let expected = expected(&nested(0)?, &expr, &row_range, &mask)?;
    let (result, _) = scan_counting(&strategy, nested, &expr, None, row_range, &mask).await?;
    assert_arrays_eq!(result, expected, &mut new_session().create_execution_ctx());
    Ok(())
}
