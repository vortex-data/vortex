// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
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
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::execution::pipeline::Scan;
use crate::execution::pipeline::Split;
use crate::execution::pipeline::Turn;
use crate::plan::EvalPlan;
use crate::plan::ListPackPlan;
use crate::plan::PlanRef;
use crate::plan::optimize;
use crate::test::SESSION;
use crate::test::Store;
use crate::test::flat;
use crate::test::struct_plan;

fn i32() -> DType {
    DType::Primitive(PType::I32, Nullability::NonNullable)
}

/// A non-nullable struct of `i32` fields named `names`.
fn struct_of(names: &[&str]) -> DType {
    DType::Struct(
        StructFields::from_iter(names.iter().map(|name| (*name, i32()))),
        Nullability::NonNullable,
    )
}

/// A shape-only list of `{sku: i32, qty: i32, price: i32}` elements whose fields are separate
/// sources, as a list layout with struct elements is lowered.
fn items_plan(nullability: Nullability) -> PlanRef {
    let fields = (0..3).map(|segment| flat(6, i32(), segment)).collect();
    let elements = struct_plan(6, struct_of(&["sku", "qty", "price"]), fields);
    let offsets = flat(6, DType::Primitive(PType::U64, Nullability::NonNullable), 3);
    let validity = nullability
        .is_nullable()
        .then(|| flat(5, DType::Bool(Nullability::NonNullable), 4));
    ListPackPlan::try_new(nullability, 5, elements, offsets, validity)
        .expect("valid test list")
        .into_plan()
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

fn optimized(expr: &Expression, plan: PlanRef) -> VortexResult<String> {
    let plan = EvalPlan::try_new(expr.bind(plan.dtype())?, plan)?.into_plan();
    Ok(optimize(plan, &SESSION)?.display_tree().to_string())
}

#[test]
fn validity_reads_evaluate_over_the_validity() -> VortexResult<()> {
    insta::assert_snapshot!(optimized(&is_null(root()), items_plan(Nullability::Nullable))?, @r"
    root: vortex.plan.eval(bool, rows=5) expr=vortex.not($)
      child: vortex.plan.filter(bool, rows=5)
        child: vortex.test.source(bool, rows=5)
    ");
    insta::assert_snapshot!(
        optimized(&is_not_null(root()), items_plan(Nullability::NonNullable))?,
        @r"
    root: vortex.plan.eval(bool, rows=5) expr=true
      child: vortex.plan.pack({}, rows=5)
    "
    );
    Ok(())
}

#[test]
fn length_reads_assemble_lists_of_empty_structs() -> VortexResult<()> {
    let expr = pack(
        [("len", list_length(root())), ("valid", is_not_null(root()))],
        Nullability::NonNullable,
    );
    insta::assert_snapshot!(optimized(&expr, items_plan(Nullability::Nullable))?, @r"
    root: vortex.plan.eval({len=u64?, valid=bool}, rows=5) expr=pack(len: vortex.list.length($), valid: is_not_null($))
      child: vortex.plan.list_pack(list({})?, rows=5)
        elements: vortex.plan.pack({}, rows=6)
        offsets: vortex.plan.filter(u64, rows=6)
          child: vortex.test.source(u64, rows=6)
        validity: vortex.plan.filter(bool, rows=5)
          child: vortex.test.source(bool, rows=5)
    ");
    Ok(())
}

#[test]
fn lambda_bodies_push_into_the_elements() -> VortexResult<()> {
    let expr = list_contains(sku()?, lit(3i32));
    insta::assert_snapshot!(optimized(&expr, items_plan(Nullability::NonNullable))?, @r"
    root: vortex.plan.eval(bool, rows=5) expr=vortex.list.contains($, 3i32)
      child: vortex.plan.list_pack(list(i32), rows=5)
        elements: vortex.plan.filter(i32, rows=6)
          child: vortex.test.source(i32, rows=6)
        offsets: vortex.plan.filter(u64, rows=6)
          child: vortex.test.source(u64, rows=6)
    ");
    Ok(())
}

/// Two lambdas share one read of the offsets: their bodies are packed into one struct of results,
/// whose fields the transforms take, so `price` is never read.
#[test]
fn lambda_bodies_share_the_offsets() -> VortexResult<()> {
    let expr = pack(
        [("skus", sku()?), ("qtys", qty()?)],
        Nullability::NonNullable,
    );
    insta::assert_snapshot!(optimized(&expr, items_plan(Nullability::NonNullable))?, @r"
    root: vortex.plan.eval({skus=list(i32), qtys=list(i32)}, rows=5) expr=pack(skus: vortex.list.transform($, (element) -> $.element.0), qtys: vortex.list.transform($, (element) -> $.element.1))
      child: vortex.plan.list_pack(list({0=i32, 1=i32}), rows=5)
        elements: vortex.plan.eval({0=i32, 1=i32}, rows=6) expr=pack(0: $.0, 1: $.1)
          child: vortex.plan.pack({0=i32, 1=i32}, rows=6)
            0: vortex.plan.filter(i32, rows=6)
              child: vortex.test.source(i32, rows=6)
            1: vortex.plan.filter(i32, rows=6)
              child: vortex.test.source(i32, rows=6)
        offsets: vortex.plan.filter(u64, rows=6)
          child: vortex.test.source(u64, rows=6)
    ");
    Ok(())
}

/// Lambdas that read every field of the elements in order read the elements as stored, as packing
/// their bodies would only rename the fields.
#[test]
fn lambda_bodies_reading_every_field_keep_the_elements() -> VortexResult<()> {
    let elements = struct_plan(
        6,
        struct_of(&["sku", "qty"]),
        vec![flat(6, i32(), 0), flat(6, i32(), 1)],
    );
    let offsets = flat(6, DType::Primitive(PType::U64, Nullability::NonNullable), 2);
    let plan = ListPackPlan::try_new(Nullability::NonNullable, 5, elements, offsets, None)?;
    let expr = pack(
        [("skus", sku()?), ("qtys", qty()?)],
        Nullability::NonNullable,
    );
    insta::assert_snapshot!(optimized(&expr, plan.into_plan())?, @r"
    root: vortex.plan.eval({skus=list(i32), qtys=list(i32)}, rows=5) expr=pack(skus: vortex.list.transform($, (x) -> $.x.sku), qtys: vortex.list.transform($, (x) -> $.x.qty))
      child: vortex.plan.list_pack(list({sku=i32, qty=i32}), rows=5)
        elements: vortex.plan.pack({sku=i32, qty=i32}, rows=6)
          sku: vortex.plan.filter(i32, rows=6)
            child: vortex.test.source(i32, rows=6)
          qty: vortex.plan.filter(i32, rows=6)
            child: vortex.test.source(i32, rows=6)
        offsets: vortex.plan.filter(u64, rows=6)
          child: vortex.test.source(u64, rows=6)
    ");
    Ok(())
}

/// An expression that reads the list itself reads every child.
#[test]
fn list_reads_keep_the_list() -> VortexResult<()> {
    let expr = list_contains(root(), lit(3i32));
    let plan = ListPackPlan::try_new(
        Nullability::NonNullable,
        5,
        flat(6, i32(), 0),
        flat(6, DType::Primitive(PType::U64, Nullability::NonNullable), 1),
        None,
    )?
    .into_plan();
    insta::assert_snapshot!(optimized(&expr, plan)?, @r"
    root: vortex.plan.eval(bool, rows=5) expr=vortex.list.contains($, 3i32)
      child: vortex.plan.list_pack(list(i32), rows=5)
        elements: vortex.plan.filter(i32, rows=6)
          child: vortex.test.source(i32, rows=6)
        offsets: vortex.plan.filter(u64, rows=6)
          child: vortex.test.source(u64, rows=6)
    ");
    Ok(())
}

/// Lists `[{1, 10}, {2, 20}]`, null (hiding `[{3, i32::MAX}]`), `[]`, `[{4, 40}, {5, 50}]` and
/// `[{6, i32::MAX}]`, of `{sku: i32, qty: i32}` elements, stored with their element fields in two
/// chunks each, and a plan reading them.
fn stored_items(store: &Store) -> VortexResult<(ArrayRef, PlanRef)> {
    let mut ctx = SESSION.create_execution_ctx();
    let skus = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let qtys = buffer![10i32, 20, i32::MAX, 40, 50, i32::MAX].into_array();
    let offsets = buffer![0u32, 2, 3, 3, 5, 6].into_array();
    let validity = Validity::from_iter([true, false, true, true, true]);
    let elements = StructArray::from_fields(&[("sku", skus.clone()), ("qty", qtys.clone())])?;
    let list = ListArray::try_new(
        elements.into_array(),
        offsets.clone(),
        validity.clone(),
        &mut ctx,
    )?
    .into_array();

    let elements = struct_plan(
        6,
        struct_of(&["sku", "qty"]),
        vec![
            store.chunked(&skus, &[3, 3])?,
            store.chunked(&qtys, &[3, 3])?,
        ],
    );
    let validity = validity.to_array(5);
    let plan = ListPackPlan::try_new(
        Nullability::Nullable,
        5,
        elements,
        store.flat(&offsets)?,
        Some(store.flat(&validity)?),
    )?
    .into_plan();
    Ok((list, plan))
}

/// Runs `plan` over the rows of `rows` that `mask` selects, delivering the bytes of `store`, and
/// counts the segments it reads.
fn run(
    store: &Store,
    plan: PlanRef,
    rows: Range<u64>,
    mask: Mask,
) -> VortexResult<(ArrayRef, usize)> {
    let dtype = plan.dtype().clone();
    let mut scan = Scan::try_new(SESSION.clone(), plan, vec![Split { rows, mask }])?;
    let mut outputs = Vec::new();
    let mut reads = 0;
    loop {
        match scan.step()? {
            Turn::Read(read) => {
                reads += 1;
                scan.deliver(read.id, store.read(read.segment_id))?;
            }
            Turn::Output(_, array) => outputs.push(array),
            Turn::Waiting => unreachable!("every read is delivered at once"),
            Turn::Done => break,
        }
    }
    let array = ChunkedArray::try_new(outputs, dtype)?.into_array();
    let array = array
        .execute::<RecursiveCanonical>(&mut SESSION.create_execution_ctx())?
        .0
        .into_array();
    Ok((array, reads))
}

/// A list reads the offsets of its selected rows, the validity of the selected lists and, of each
/// element field its expression reads, the chunks that hold the elements of the selected valid
/// lists: `chunks` of the two. A null or empty list reads no element.
#[rstest]
#[case::full(0..5, Mask::new_true(5), 2)]
#[case::first_chunk(0..1, Mask::new_true(1), 1)]
#[case::second_chunk(3..5, Mask::from_iter([true, false]), 1)]
#[case::spanning(0..4, Mask::from_iter([true, false, true, true]), 2)]
#[case::null_list(1..2, Mask::new_true(1), 0)]
#[case::empty_list(2..3, Mask::new_true(1), 0)]
fn lists_read_elements_of_selected_valid_lists(
    #[case] rows: Range<u64>,
    #[case] mask: Mask,
    #[case] chunks: usize,
    #[values((root(), 2), (sku()?, 1), (qty()?, 1))] read: (Expression, usize),
) -> VortexResult<()> {
    let (expr, fields) = read;
    let store = Store::default();
    let (list, plan) = stored_items(&store)?;
    let plan = optimize(
        EvalPlan::try_new(expr.bind(plan.dtype())?, plan)?.into_plan(),
        &SESSION,
    )?;
    let expected = list
        .slice(usize::try_from(rows.start)?..usize::try_from(rows.end)?)?
        .filter(mask.clone())?
        .apply(&expr)?;
    let (actual, reads) = run(&store, plan, rows, mask)?;
    assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    assert_eq!(reads, 2 + chunks * fields);
    Ok(())
}

/// A lambda that can fail runs only on the elements of the selected valid lists.
#[rstest]
#[case::hidden_by_null(0..4, Mask::new_true(4), true)]
#[case::hidden_by_selection(3..5, Mask::from_iter([true, false]), true)]
#[case::selected(4..5, Mask::new_true(1), false)]
fn fallible_lambdas_run_on_selected_valid_lists(
    #[case] rows: Range<u64>,
    #[case] mask: Mask,
    #[case] succeeds: bool,
) -> VortexResult<()> {
    let expr = list_transform(
        root(),
        Lambda::new(["x"], checked_add(get_item("qty", var("x")), lit(1i32)))?,
    );
    let store = Store::default();
    let (list, plan) = stored_items(&store)?;
    let plan = optimize(
        EvalPlan::try_new(expr.bind(plan.dtype())?, plan)?.into_plan(),
        &SESSION,
    )?;
    let result = run(&store, plan, rows.clone(), mask.clone());
    assert_eq!(result.is_ok(), succeeds);
    if let Ok((actual, _)) = result {
        let expected = list
            .slice(usize::try_from(rows.start)?..usize::try_from(rows.end)?)?
            .filter(mask)?
            .apply(&expr)?;
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }
    Ok(())
}

/// Offsets that decrease or run past the elements fail the read rather than build invalid lists.
#[rstest]
#[case::decreasing(buffer![0u32, 2, 1].into_array())]
#[case::past_the_elements(buffer![0u32, 2, 4].into_array())]
fn corrupt_offsets_fail(#[case] offsets: ArrayRef) -> VortexResult<()> {
    let store = Store::default();
    let plan = ListPackPlan::try_new(
        Nullability::NonNullable,
        2,
        store.flat(&buffer![1i32, 2, 3].into_array())?,
        store.flat(&offsets)?,
        None,
    )?
    .into_plan();
    assert!(run(&store, plan, 0..2, Mask::new_true(2)).is_err());
    Ok(())
}
