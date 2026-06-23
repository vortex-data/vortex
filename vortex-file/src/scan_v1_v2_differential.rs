// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential tests that scan the same [`ScanRequest`] through both the V1
//! (LayoutReader-based) and V2 (ScanPlan-based) scan paths and assert the
//! outputs are identical.
//!
//! V1 is driven through [`VortexFile::scan`] +
//! [`ScanBuilder::into_array_stream`]; V2 is driven directly through
//! [`VortexFile::scan_plan_stream`]. Neither side flips the process-global
//! `VORTEX_SCAN_IMPL` env var, so the two implementations run side by side in
//! the same test process.

// Nested struct fixtures use short field names (a, b, c, s) that mirror the v1
// regression tests; single-char names are clearest here.
#![allow(clippy::many_single_char_names)]

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::merge;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::stats::PRUNING_STATS;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_layout::layouts::row_idx::row_idx;
use vortex_scan::ScanRequest;
use vortex_session::VortexSession;

use crate::OpenOptionsSessionExt;
use crate::VortexFile;
use crate::WriteOptionsSessionExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::tests::new_test_session);

/// Write `array` to an in-memory Vortex file, optionally with file statistics
/// (which exercises the V2 `FileStatsScanPlan` path and V1 `FileStatsLayoutReader`).
async fn write_file(array: ArrayRef, with_stats: bool) -> VortexResult<VortexFile> {
    let mut buf = ByteBufferMut::empty();
    if with_stats {
        let mut writer = SESSION
            .write_options()
            .with_file_statistics(PRUNING_STATS.to_vec())
            .writer(&mut buf, array.dtype().clone());
        writer.push(array).await?;
        writer.finish().await?;
    } else {
        SESSION
            .write_options()
            .write(&mut buf, array.to_array_stream())
            .await?;
    }
    SESSION.open_options().open_buffer(buf.freeze())
}

/// Scan `file` through the V1 LayoutReader path.
async fn scan_v1(file: &VortexFile, request: &ScanRequest) -> VortexResult<ArrayRef> {
    let mut builder = file
        .scan()?
        .with_projection(request.projection.clone())
        .with_ordered(true);
    if let Some(filter) = &request.filter {
        builder = builder.with_filter(filter.clone());
    }
    builder.into_array_stream()?.read_all().await
}

/// Scan `file` through the V2 ScanPlan path.
async fn scan_v2(file: &VortexFile, request: &ScanRequest) -> VortexResult<ArrayRef> {
    file.scan_plan_stream(request.clone())?.read_all().await
}

/// Scan the same request through both paths and assert the outputs are equal.
async fn assert_v1_eq_v2(file: &VortexFile, request: ScanRequest) -> VortexResult<()> {
    let v1 = scan_v1(file, &request).await?;
    let v2 = scan_v2(file, &request).await?;
    assert_eq!(
        v1.dtype(),
        v2.dtype(),
        "V1/V2 dtype mismatch for projection {} filter {:?}",
        request.projection,
        request.filter
    );
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(v1, v2, &mut ctx);
    Ok(())
}

/// Build an ordered V2 scan request from a projection and optional filter.
fn request(projection: Expression, filter: Option<Expression>) -> ScanRequest {
    ScanRequest {
        projection,
        filter,
        ordered: true,
        ..Default::default()
    }
}

// ---- Fixtures ----

/// Flat primitive column, both nullable and non-nullable variants.
fn flat_primitive(nullable: bool) -> ArrayRef {
    let numbers = if nullable {
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None, Some(6)])
            .into_array()
    } else {
        buffer![1i32, 2, 3, 4, 5, 6].into_array()
    };
    StructArray::from_fields(&[("numbers", numbers)])
        .unwrap()
        .into_array()
}

/// A chunked primitive column.
fn chunked() -> ArrayRef {
    let numbers = ChunkedArray::from_iter([
        buffer![1i32, 2, 3, 4].into_array(),
        buffer![5i32, 6, 7, 8].into_array(),
        buffer![9i32, 10, 11, 12].into_array(),
    ])
    .into_array();
    StructArray::from_fields(&[("numbers", numbers)])
        .unwrap()
        .into_array()
}

/// A low-cardinality string column that the writer dictionary-encodes.
fn dict_encoded() -> ArrayRef {
    let n = 4096usize;
    let values: Vec<&str> = (0..n).map(|i| ["alpha", "beta", "gamma"][i % 3]).collect();
    let strings = VarBinViewArray::from_iter_str(values).into_array();
    StructArray::from_fields(&[("letters", strings)])
        .unwrap()
        .into_array()
}

/// A wide-range numeric column over many rows so the writer emits zone stats.
fn zoned() -> ArrayRef {
    let n = 100_000i32;
    let numbers = PrimitiveArray::from_iter(0..n).into_array();
    StructArray::from_fields(&[("numbers", numbers)])
        .unwrap()
        .into_array()
}

/// A `keep` flag column plus a `name` string column, for multi-conjunct filter tests:
/// `id != 0` is a cheap, selective predicate; `name LIKE '%match%'` is the expensive
/// residual that should run filter-first once `id` has narrowed the demanded rows.
fn id_and_name(keep: &[u32], names: &[&str]) -> ArrayRef {
    StructArray::from_fields(&[
        (
            "id",
            PrimitiveArray::from_iter(keep.iter().copied()).into_array(),
        ),
        (
            "name",
            VarBinViewArray::from_iter_str(names.iter().copied()).into_array(),
        ),
    ])
    .unwrap()
    .into_array()
}

/// 16 names where most rows contain the `match` needle (decoys), so a residual `LIKE`
/// that ignored the cheaper predicate would diverge from V1.
const MULTI_CONJUNCT_NAMES: [&str; 16] = [
    "row0_match",
    "row1_match",
    "no_hit_here",
    "row3_match",
    "row4_match",
    "row5_match",
    "row6_match",
    "row7_match",
    "row8_match",
    "has_match_inside",
    "row10_match",
    "row11_match",
    "row12_match",
    "row13_match",
    "row14_match",
    "row15_match",
];

fn multi_conjunct_filter() -> Expression {
    vortex_array::expr::and(
        vortex_array::expr::not_eq(get_item("id", root()), lit(0u32)),
        vortex_array::expr::like(get_item("name", root()), lit("%match%")),
    )
}

/// Outer struct is non-nullable (so the file writes), but it contains a nullable
/// nested struct `a` with a non-nullable field `b.c`. Projecting `a.b.c` (or
/// selecting `c` out of `a.b`) must preserve the nulls of the nullable `a.b`
/// struct.
///
/// |      a.b          |
/// |-------------------|
/// | `{ "c": 4 }`      |
/// |     `NULL`        |
/// | `{ "c": 6 }`      |
/// |     `NULL`        |
/// | `{ "c": 10 }`     |
fn nested_nullable_struct() -> ArrayRef {
    let c = buffer![4i32, 5, 6, 8, 10].into_array();
    let b = StructArray::try_from_iter_with_validity([("c", c)], Validity::NonNullable)
        .unwrap()
        .into_array();
    let a = StructArray::try_from_iter_with_validity(
        [("b", b)],
        Validity::Array(BoolArray::from_iter([true, false, true, false, true]).into_array()),
    )
    .unwrap()
    .into_array();
    StructArray::try_from_iter_with_validity([("a", a)], Validity::NonNullable)
        .unwrap()
        .into_array()
}

// ---- Differential cases ----

#[rstest]
#[case::flat_primitive_nonnull(flat_primitive(false))]
#[case::flat_primitive_nullable(flat_primitive(true))]
#[case::chunked(chunked())]
#[case::dict_encoded(dict_encoded())]
#[tokio::test]
async fn differential_full_scan(#[case] array: ArrayRef) -> VortexResult<()> {
    let file = write_file(array, false).await?;
    assert_v1_eq_v2(&file, request(root(), None)).await
}

#[rstest]
#[case::flat_primitive_nonnull(flat_primitive(false))]
#[case::flat_primitive_nullable(flat_primitive(true))]
#[case::chunked(chunked())]
#[tokio::test]
async fn differential_project_numbers(#[case] array: ArrayRef) -> VortexResult<()> {
    let file = write_file(array, false).await?;
    assert_v1_eq_v2(&file, request(select(["numbers"], root()), None)).await
}

#[rstest]
#[case::flat_primitive_nonnull(flat_primitive(false))]
#[case::flat_primitive_nullable(flat_primitive(true))]
#[case::chunked(chunked())]
#[tokio::test]
async fn differential_filter_numbers(#[case] array: ArrayRef) -> VortexResult<()> {
    let file = write_file(array, false).await?;
    let filter = gt(get_item("numbers", root()), lit(3i32));
    assert_v1_eq_v2(&file, request(root(), Some(filter))).await
}

#[tokio::test]
async fn differential_dict_filter() -> VortexResult<()> {
    let file = write_file(dict_encoded(), false).await?;
    let filter = vortex_array::expr::eq(get_item("letters", root()), lit("beta"));
    assert_v1_eq_v2(&file, request(root(), Some(filter))).await
}

#[tokio::test]
async fn differential_zoned_full() -> VortexResult<()> {
    let file = write_file(zoned(), true).await?;
    assert_v1_eq_v2(&file, request(root(), None)).await
}

#[tokio::test]
async fn differential_zoned_filter() -> VortexResult<()> {
    let file = write_file(zoned(), true).await?;
    // Filter that zone stats can partially prune.
    let filter = gt(get_item("numbers", root()), lit(99_990i32));
    assert_v1_eq_v2(&file, request(root(), Some(filter))).await
}

/// Low-density multi-conjunct filter: `id != 0` keeps 2/16 rows (density 0.125 < 0.2),
/// so the expensive `name LIKE '%match%'` runs filter-first over only the demanded rows
/// and its compacted verdict is scattered back. Asserted against the V1 reference, which
/// catches any off-by-rank error in the scatter-back.
#[tokio::test]
async fn differential_multi_conjunct_filter_first() -> VortexResult<()> {
    let mut keep = [0u32; 16];
    keep[2] = 1;
    keep[9] = 1;
    let file = write_file(id_and_name(&keep, &MULTI_CONJUNCT_NAMES), false).await?;
    assert_v1_eq_v2(&file, request(root(), Some(multi_conjunct_filter()))).await
}

/// High-density multi-conjunct filter: `id != 0` keeps 14/16 rows (density 0.875 > 0.2),
/// so the residual takes the dense path. Must still match V1.
#[tokio::test]
async fn differential_multi_conjunct_dense() -> VortexResult<()> {
    let mut keep = [1u32; 16];
    keep[2] = 0;
    keep[9] = 0;
    let file = write_file(id_and_name(&keep, &MULTI_CONJUNCT_NAMES), false).await?;
    assert_v1_eq_v2(&file, request(root(), Some(multi_conjunct_filter()))).await
}

#[tokio::test]
async fn differential_single_field_merge_select_projection() -> VortexResult<()> {
    let file = write_file(flat_primitive(false), true).await?;
    let projection = merge([
        pack([("file_row_number", row_idx())], Nullability::NonNullable),
        select(["numbers"], root()),
    ]);
    assert_v1_eq_v2(&file, request(projection, None)).await
}

/// Reproduces the struct-null bug: projecting a single deep field out of a
/// nullable nested struct must apply the parent struct's validity. The V2
/// single-field fast path previously bypassed `self.validity`.
#[tokio::test]
async fn differential_nested_nullable_struct_get_item() -> VortexResult<()> {
    let file = write_file(nested_nullable_struct(), false).await?;
    // SELECT a.b.c (single deep field access)
    let projection = get_item("c", get_item("b", get_item("a", root())));
    assert_v1_eq_v2(&file, request(projection, None)).await
}

/// Same bug via `select(["c"], a.b)`: selecting a single field out of the
/// nullable nested struct must preserve the struct's nulls.
#[tokio::test]
async fn differential_nested_nullable_struct_select() -> VortexResult<()> {
    let file = write_file(nested_nullable_struct(), false).await?;
    let projection = select(["c"], get_item("b", get_item("a", root())));
    assert_v1_eq_v2(&file, request(projection, None)).await
}

/// Projecting the nullable nested struct `a.b` itself (a struct value) must also
/// preserve its nulls.
#[tokio::test]
async fn differential_nested_nullable_struct_project_struct() -> VortexResult<()> {
    let file = write_file(nested_nullable_struct(), false).await?;
    let projection = get_item("b", get_item("a", root()));
    assert_v1_eq_v2(&file, request(projection, None)).await
}

// ---- Ported V1 regression tests (struct nulls), exercised through V2 ----

/// Port of `vortex-layout` `test_struct_layout_nulls`: a nullable struct, when a
/// single field is projected, must mask the field with the parent struct's
/// validity. Reachable on the V2 file path through a non-nullable outer struct
/// wrapping a nullable inner struct.
#[tokio::test]
async fn v2_struct_layout_nulls() -> VortexResult<()> {
    // inner struct `a` is nullable with fields a, b, c; row 0 is null.
    let inner = StructArray::try_from_iter_with_validity(
        [
            ("a", buffer![7i32, 2, 3].into_array()),
            ("b", buffer![4i32, 5, 6].into_array()),
            ("c", buffer![4i32, 5, 6].into_array()),
        ],
        Validity::Array(BoolArray::from_iter([false, true, true]).into_array()),
    )?
    .into_array();
    let outer = StructArray::try_from_iter_with_validity([("s", inner)], Validity::NonNullable)?
        .into_array();

    let file = write_file(outer, false).await?;

    // SELECT s.a -> the result must be masked with s's validity (row 0 null).
    let projection = get_item("a", get_item("s", root()));
    let v2 = scan_v2(&file, &request(projection, None)).await?;

    assert_eq!(
        v2.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );

    let expected = PrimitiveArray::from_option_iter([None, Some(2i32), Some(3)]).into_array();
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(v2, expected, &mut ctx);
    Ok(())
}

/// Port of `vortex-layout` `test_struct_layout_nested`: projecting `c` out of a
/// nullable nested struct `s.a.b` must preserve the nested struct's nulls.
#[tokio::test]
async fn v2_struct_layout_nested() -> VortexResult<()> {
    // s.a.b is nullable (true, false, true); s.a.b.c is non-nullable.
    let c = buffer![4i32, 5, 6].into_array();
    let b =
        StructArray::try_from_iter_with_validity([("c", c)], Validity::NonNullable)?.into_array();
    let a = StructArray::try_from_iter_with_validity(
        [("b", b)],
        Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
    )?
    .into_array();
    let s =
        StructArray::try_from_iter_with_validity([("a", a)], Validity::NonNullable)?.into_array();
    let outer =
        StructArray::try_from_iter_with_validity([("s", s)], Validity::NonNullable)?.into_array();

    let file = write_file(outer, false).await?;

    // SELECT c from s.a.b
    let projection = select(["c"], get_item("b", get_item("a", get_item("s", root()))));
    let v2 = scan_v2(&file, &request(projection, None)).await?;

    // Result is a nullable struct (because s.a.b is nullable) with a
    // non-nullable field "c".
    assert_eq!(
        v2.dtype(),
        &DType::Struct(
            vortex_array::dtype::StructFields::from_iter([(
                "c",
                DType::Primitive(PType::I32, Nullability::NonNullable)
            )]),
            Nullability::Nullable,
        )
    );

    // Cross-check against V1 producing the same masked output.
    let v1 = scan_v1(
        &file,
        &request(
            select(["c"], get_item("b", get_item("a", get_item("s", root())))),
            None,
        ),
    )
    .await?;
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(v1, v2, &mut ctx);

    // Build the expected struct directly: rows 0 and 2 valid, row 1 null.
    let expected = StructArray::try_from_iter_with_validity(
        [("c", buffer![4i32, 5, 6].into_array())],
        Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
    )?
    .into_array();
    assert_arrays_eq!(v2, expected, &mut ctx);
    Ok(())
}
