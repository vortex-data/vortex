// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::Metadata;
use arrow_schema::Schema;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use datafusion_common::Result as DFResult;
use datafusion_common::config::ConfigOptions;
use datafusion_expr::ScalarUDF;
use datafusion_functions::core::getfield::GetFieldFunc;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::expressions as df_expr;
use rstest::rstest;
use vortex::dtype::DType;
use vortex::expr::get_item;
use vortex::expr::root;
use vortex::expr::transform::replace;
use vortex_arrow::ArrowSessionExt;

use super::column_uses;
use crate::common_tests::VX_SESSION;
use crate::convert::exprs::ProcessedProjection;
use crate::convert::exprs::raw_projection;
use crate::convert::schema::calculate_physical_schema;
use crate::row_index;

/// Arrow field metadata that tags a field, so a test can see that a pruned field keeps it.
fn metadata(value: &str) -> Metadata {
    [("source".to_owned(), value.to_owned())]
        .into_iter()
        .collect()
}

fn int(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Int32, nullable)
}

/// The struct `s` and its child `inner` are nullable. Of the other nested columns, only `t` can be
/// pruned: `ext` is an extension type, `dup` has duplicate child names, and `m` is a map.
static SCHEMA: LazyLock<Schema> = LazyLock::new(|| {
    let inner = Fields::from(vec![
        Field::new("label", DataType::Utf8, true).with_metadata(metadata("label")),
        int("other", false),
    ]);
    let s = Fields::from(vec![
        Field::new("inner", DataType::Struct(inner), true).with_metadata(metadata("inner")),
        int("x", false),
        int("y", true),
    ]);
    let entries = Field::new(
        "entries",
        DataType::Struct(vec![Field::new("key", DataType::Utf8, false), int("value", true)].into()),
        false,
    );

    Schema::new(vec![
        int("id", false),
        Field::new("s", DataType::Struct(s), true).with_metadata(metadata("s")),
        Field::new("t", DataType::Struct(vec![int("a", false)].into()), false),
        Field::new(
            "ext",
            DataType::Struct(vec![int("a", false), int("b", false)].into()),
            true,
        )
        .with_metadata(
            [(EXTENSION_TYPE_NAME_KEY.to_owned(), "test.ext".to_owned())]
                .into_iter()
                .collect::<Metadata>(),
        ),
        Field::new(
            "dup",
            DataType::Struct(vec![int("a", false), int("a", false), int("b", false)].into()),
            true,
        ),
        Field::new("m", DataType::Map(Arc::new(entries), false), true),
    ])
});

/// The Vortex dtype of `id` and `s`, as the source stores them.
fn stored_dtype() -> anyhow::Result<DType> {
    Ok(VX_SESSION
        .arrow()
        .from_arrow_schema(&SCHEMA.project(&[0, 1])?)?)
}

fn get_field(source: Arc<dyn PhysicalExpr>, keys: &[&str]) -> DFResult<Arc<dyn PhysicalExpr>> {
    let mut args = vec![source];
    args.extend(keys.iter().map(|key| df_expr::lit(*key)));
    Ok(Arc::new(ScalarFunctionExpr::try_new(
        Arc::new(ScalarUDF::from(GetFieldFunc::new())),
        args,
        &SCHEMA,
        Arc::new(ConfigOptions::new()),
    )?))
}

/// Read a dotted `path` such as `s.inner.label` with one flattened `get_field` call, or read its
/// column if it has no keys.
fn read(path: &str) -> DFResult<Arc<dyn PhysicalExpr>> {
    let mut names = path.split('.');
    let column = names.next().unwrap_or_default();
    let keys = names.collect::<Vec<_>>();
    let column = Arc::new(df_expr::Column::new_with_schema(column, &SCHEMA)?);
    if keys.is_empty() {
        return Ok(column);
    }

    get_field(column, &keys)
}

/// Run [`raw_projection`] over [`SCHEMA`] for `exprs`, which it names `c0`, `c1`, and so on.
fn run_raw_projection(
    exprs: Vec<Arc<dyn PhysicalExpr>>,
    residual_filter: Option<Arc<dyn PhysicalExpr>>,
) -> DFResult<ProcessedProjection> {
    let projection = exprs
        .into_iter()
        .enumerate()
        .map(|(idx, expr)| ProjectionExpr::new(expr, format!("c{idx}")))
        .collect::<Vec<_>>();

    raw_projection(
        &SCHEMA,
        ProjectionExprs::from(projection),
        residual_filter.as_ref(),
    )
}

#[rstest]
#[case::leaf(&["s.inner.label"], "pack(s: pack(inner: $.s.inner{label}))")]
#[case::two_leaves(
    &["s.inner.label", "s.x"],
    "pack(s: pack(inner: $.s.inner{label}, x: $.s.x))"
)]
#[case::whole_children(&["s.x", "s.y"], "pack(s: $.s{x, y})")]
#[case::every_child(&["s.inner", "s.x", "s.y"], "pack(s: $.s)")]
#[case::every_leaf(&["s.inner.label", "s.inner.other", "s.x", "s.y"], "pack(s: $.s)")]
#[case::bare_and_field(&["s", "s.x"], "pack(s: $.s)")]
#[case::two_structs(&["s.x", "t.a"], "pack(s: $.s{x}, t: $.t)")]
#[case::extension(&["ext.a"], "pack(ext: $.ext)")]
#[case::duplicate_names(&["dup.b"], "pack(dup: $.dup)")]
#[case::map(&["m.key"], "pack(m: $.m)")]
fn test_raw_projection_reads_used_children(
    #[case] paths: &[&str],
    #[case] expected: &str,
) -> DFResult<()> {
    let exprs = paths
        .iter()
        .map(|path| read(path))
        .collect::<DFResult<_>>()?;
    assert_eq!(
        run_raw_projection(exprs, None)?.scan_projection.to_string(),
        expected
    );

    Ok(())
}

/// The inner call returns all of `s.inner` to the outer call.
#[test]
fn test_raw_projection_reads_result_of_inner_call_whole() -> DFResult<()> {
    let nested = get_field(read("s.inner")?, &["label"])?;
    assert_eq!(
        run_raw_projection(vec![nested], None)?
            .scan_projection
            .to_string(),
        "pack(s: $.s{inner})"
    );

    Ok(())
}

#[test]
fn test_raw_projection_reads_whole_struct_for_other_uses() -> DFResult<()> {
    let is_null = Arc::new(df_expr::IsNullExpr::new(read("s")?));
    assert_eq!(
        run_raw_projection(vec![read("s.x")?, is_null], None)?
            .scan_projection
            .to_string(),
        "pack(s: $.s)"
    );

    Ok(())
}

#[test]
fn test_raw_projection_reads_children_of_residual_filter() -> DFResult<()> {
    let residual = Arc::new(df_expr::IsNullExpr::new(read("s.y")?));
    assert_eq!(
        run_raw_projection(vec![read("s.x")?], Some(residual))?
            .scan_projection
            .to_string(),
        "pack(s: $.s{x, y})"
    );

    Ok(())
}

#[test]
fn test_raw_projection_reads_row_index_after_pruned_struct() -> DFResult<()> {
    let processed = run_raw_projection(vec![read("s.x")?, row_index()], None)?;

    assert_eq!(
        processed.scan_projection.to_string(),
        "pack(s: $.s{x}, __vortex_row_index: #row_idx)"
    );
    assert_eq!(
        processed.leftover_projection.as_ref()[1].expr.to_string(),
        "__vortex_row_index@1"
    );

    Ok(())
}

#[test]
fn test_column_uses_rejects_column_outside_schema() {
    let column: Arc<dyn PhysicalExpr> =
        Arc::new(df_expr::Column::new("missing", SCHEMA.fields().len()));

    let error = column_uses([&column], &SCHEMA).unwrap_err();

    assert_eq!(
        error.strip_backtrace(),
        "Execution error: Column missing@6 is out of bounds"
    );
}

/// A leaf read through the narrowed scan output, as a filter pushed above the scan reads it,
/// reduces to a read of the same leaf from the source.
#[rstest]
#[case::through_pack(&["s.inner.label", "s.x"])]
#[case::through_select(&["s.x", "s.y"])]
fn test_leaf_read_through_pruned_struct_reduces_to_source_leaf(
    #[case] paths: &[&str],
) -> anyhow::Result<()> {
    let exprs = paths
        .iter()
        .map(|path| read(path))
        .collect::<DFResult<_>>()?;
    let scan_projection = run_raw_projection(exprs, None)?.scan_projection;
    let stored_dtype = stored_dtype()?;

    for path in paths {
        let leaf_read = path
            .split('.')
            .fold(root(), |read, name| get_item(name, read));
        let composed = replace(leaf_read.clone(), &root(), scan_projection.clone());
        assert_eq!(
            composed
                .bind(&stored_dtype)?
                .optimize_recursive()?
                .to_string(),
            leaf_read.to_string()
        );
    }

    Ok(())
}

/// The reference schema keeps the Arrow fields of the kept children, and the scan output schema
/// is the same whether it is computed before or after the scan optimizes its projection.
#[test]
fn test_raw_projection_keeps_arrow_fields_of_kept_children() -> anyhow::Result<()> {
    let processed = run_raw_projection(vec![read("s.inner.label")?, read("s.x")?], None)?;

    let label = Field::new("label", DataType::Utf8, true).with_metadata(metadata("label"));
    let pruned_s = |s_nullable, inner_nullable, x_nullable| {
        let inner = Field::new(
            "inner",
            DataType::Struct(vec![label.clone()].into()),
            inner_nullable,
        )
        .with_metadata(metadata("inner"));
        Field::new(
            "s",
            DataType::Struct(vec![inner, int("x", x_nullable)].into()),
            s_nullable,
        )
        .with_metadata(metadata("s"))
    };
    assert_eq!(
        processed.scan_reference_schema,
        Schema::new(vec![pruned_s(true, true, false)])
    );

    // Reading through the nullable `s` makes `x` nullable, and the pack around the kept children
    // has no nulls of its own.
    let stored_dtype = stored_dtype()?;
    let returned_dtype = processed.scan_projection.return_dtype(&stored_dtype)?;
    let optimized_dtype = processed
        .scan_projection
        .bind(&stored_dtype)?
        .optimize_recursive()?
        .dtype()
        .clone();
    let expected = Schema::new(vec![pruned_s(false, true, true)]);
    for dtype in [returned_dtype, optimized_dtype] {
        assert_eq!(
            calculate_physical_schema(
                &dtype,
                &processed.scan_reference_schema,
                &VX_SESSION.arrow()
            )?,
            expected
        );
    }

    Ok(())
}
