// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use hegel::TestCase;
use hegel::generators as gs;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Patched;
use crate::arrays::PatchedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::filter::FilterKernel;
use crate::arrays::slice::SliceKernel;
use crate::assert_arrays_eq;
use crate::patches::Patches;
use crate::scalar_fn::builtin::compare::CompareKernel;
use crate::scalar_fn::builtin::operators::CompareOperator;

struct Fixture {
    array: PatchedArray,
    expected: Vec<Option<i32>>,
    edge: usize,
}

fn fixture(tc: &TestCase, ctx: &mut ExecutionCtx) -> VortexResult<Fixture> {
    let boundary = if tc.draw(gs::booleans()) { 65536 } else { 1024 };
    let len = boundary + 1025;
    let pivot = tc.draw(gs::integers::<usize>().max_value(len - 1));
    let mut positions = tc.draw(gs::vecs(gs::integers::<usize>().max_value(len - 1)).max_size(30));
    positions.extend([
        0,
        1,
        1023,
        1024,
        boundary - 1,
        boundary,
        boundary + 1,
        pivot,
        len - 1,
    ]);
    positions.sort_unstable();
    positions.dedup();
    let mut rows: Vec<_> = (0..len)
        .map(|i| (i % 5 != 0).then_some(i32::try_from(i % 11).unwrap()))
        .collect();
    let base = PrimitiveArray::from_option_iter(rows.iter().copied()).into_array();
    let values: Vec<_> = positions
        .iter()
        .enumerate()
        .map(|(i, &row)| {
            let value = (i % 3 != 0).then_some(17);
            rows[row] = value;
            value
        })
        .collect();
    let patches = Patches::new(
        len,
        0,
        PrimitiveArray::from_iter(positions.iter().map(|&i| u32::try_from(i).unwrap()))
            .into_array(),
        PrimitiveArray::from_option_iter(values).into_array(),
        None,
        ctx,
    )?;
    let a = Patched::from_array_and_patches(base, &patches, ctx)?;
    let start = tc.draw(gs::integers::<usize>().max_value(boundary - 1));
    let end = tc.draw(
        gs::integers::<usize>()
            .min_value(boundary + 2)
            .max_value(len),
    );
    let sliced = <Patched as SliceKernel>::slice(a.as_view(), start..end, ctx)?.unwrap();
    assert_arrays_eq!(
        sliced,
        PrimitiveArray::from_option_iter(rows[start..end].iter().copied()),
        ctx
    );
    let nested_start = tc.draw(gs::integers::<usize>().min_value(start).max_value(boundary));
    let nested = <Patched as SliceKernel>::slice(
        sliced.as_::<Patched>(),
        nested_start - start..end - start,
        ctx,
    )?
    .unwrap();
    let expected = &rows[nested_start..end];
    Ok(Fixture {
        array: nested.downcast::<Patched>(),
        expected: expected.to_vec(),
        edge: boundary - nested_start,
    })
}

#[hegel::test]
fn nested_slice(tc: TestCase) {
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let f = fixture(&tc, &mut ctx).unwrap();
    assert_arrays_eq!(
        f.array,
        PrimitiveArray::from_option_iter(f.expected.iter().copied()),
        &mut ctx
    );
    let lo = tc.draw(gs::integers::<usize>().max_value(f.expected.len()));
    let hi = tc.draw(
        gs::integers::<usize>()
            .min_value(lo)
            .max_value(f.expected.len()),
    );
    let sub = <Patched as SliceKernel>::slice(f.array.as_view(), lo..hi, &mut ctx)
        .unwrap()
        .unwrap();
    assert_arrays_eq!(
        sub,
        PrimitiveArray::from_option_iter(f.expected[lo..hi].iter().copied()),
        &mut ctx
    );
}

#[hegel::test]
fn scalar_lookup(tc: TestCase) {
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let f = fixture(&tc, &mut ctx).unwrap();
    let dense = PrimitiveArray::from_option_iter(f.expected.iter().copied());
    for row in [
        tc.draw(gs::integers::<usize>().max_value(f.expected.len() - 1)),
        f.edge,
        f.edge + 1,
    ] {
        assert_eq!(
            f.array.execute_scalar(row, &mut ctx).unwrap(),
            dense.execute_scalar(row, &mut ctx).unwrap()
        );
    }
}

#[hegel::test]
fn replacement_validity(tc: TestCase) {
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let f = fixture(&tc, &mut ctx).unwrap();
    assert_eq!(
        f.array
            .validity()
            .unwrap()
            .execute_mask(f.array.len(), &mut ctx)
            .unwrap(),
        Mask::from_iter(f.expected.iter().map(Option::is_some))
    );
}

#[hegel::test]
fn take(tc: TestCase) {
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let Fixture {
        array,
        expected,
        edge,
    } = fixture(&tc, &mut ctx).unwrap();
    let mut take = tc
        .draw(gs::vecs(gs::integers::<usize>().max_value(expected.len() - 1)).max_size(24))
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    take.extend([
        Some(edge + 1),
        None,
        Some(edge),
        Some(0),
        Some(edge),
        Some(expected.len() - 1),
    ]);
    let indices =
        PrimitiveArray::from_option_iter(take.iter().map(|i| i.map(|x| u32::try_from(x).unwrap())))
            .into_array();
    let taken = <Patched as TakeExecute>::take(array.as_view(), &indices, &mut ctx)
        .unwrap()
        .unwrap();
    assert_arrays_eq!(
        taken,
        PrimitiveArray::from_option_iter(take.iter().map(|i| i.and_then(|i| expected[i]))),
        &mut ctx
    );
}

#[hegel::test]
fn filter(tc: TestCase) {
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let Fixture {
        array, expected, ..
    } = fixture(&tc, &mut ctx).unwrap();
    let stride = tc.draw(gs::integers::<usize>().min_value(1).max_value(17));
    let mask = Mask::from_iter((0..expected.len()).map(|i| i % stride == 0));
    let filtered = <Patched as FilterKernel>::filter(array.as_view(), &mask, &mut ctx)
        .unwrap()
        .unwrap();
    assert_arrays_eq!(
        filtered,
        PrimitiveArray::from_option_iter(expected.iter().step_by(stride).copied()),
        &mut ctx
    );
}

#[hegel::test]
fn compare(tc: TestCase) {
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let Fixture {
        array, expected, ..
    } = fixture(&tc, &mut ctx).unwrap();
    let compared = <Patched as CompareKernel>::compare(
        array.as_view(),
        &ConstantArray::new(17i32, expected.len()).into_array(),
        CompareOperator::Eq,
        &mut ctx,
    )
    .unwrap()
    .unwrap();
    assert_arrays_eq!(
        compared,
        BoolArray::from_iter(expected.iter().map(|v| v.map(|v| v == 17))),
        &mut ctx
    );
}
