// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Repeated;
use crate::arrays::RepeatedArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::repeated::RepeatedArraySlotsExt;
use crate::assert_arrays_eq;
use crate::builders::builder_with_capacity_in;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn repeated(child: impl IntoArray, count: usize) -> VortexResult<ArrayRef> {
    Ok(RepeatedArray::try_new(child.into_array(), count)?.into_array())
}

#[test]
fn repeats_primitive() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(buffer![1i32, 2, 3], 3)?;

    assert_eq!(array.len(), 9);
    assert_arrays_eq!(
        array,
        PrimitiveArray::from_iter([1i32, 2, 3, 1, 2, 3, 1, 2, 3]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn repeats_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(PrimitiveArray::from_option_iter([Some(1i64), None]), 2)?;

    assert_arrays_eq!(
        array,
        PrimitiveArray::from_option_iter([Some(1i64), None, Some(1), None]),
        &mut ctx
    );
    assert_eq!(array.invalid_count(&mut ctx)?, 2);
    Ok(())
}

#[test]
fn repeats_strings() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child = VarBinViewArray::from_iter_nullable_str([Some("a"), None, Some("a long string")]);
    let array = repeated(child, 2)?;

    assert_arrays_eq!(
        array,
        VarBinViewArray::from_iter_nullable_str([
            Some("a"),
            None,
            Some("a long string"),
            Some("a"),
            None,
            Some("a long string"),
        ]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn repeats_struct() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child = StructArray::try_from_iter([
        ("a", buffer![1u8, 2].into_array()),
        ("b", VarBinViewArray::from_iter_str(["x", "y"]).into_array()),
    ])?;
    let expected = StructArray::try_from_iter([
        ("a", buffer![1u8, 2, 1, 2, 1, 2].into_array()),
        (
            "b",
            VarBinViewArray::from_iter_str(["x", "y", "x", "y", "x", "y"]).into_array(),
        ),
    ])?;

    assert_arrays_eq!(repeated(child, 3)?, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::zero_count(buffer![1i32, 2, 3].into_array(), 0)]
#[case::empty_child(PrimitiveArray::from_iter(Vec::<i32>::new()).into_array(), 4)]
fn repeats_to_empty(#[case] child: ArrayRef, #[case] count: usize) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(child, count)?;

    assert!(array.is_empty());
    assert_arrays_eq!(
        array,
        PrimitiveArray::from_iter(Vec::<i32>::new()),
        &mut ctx
    );
    Ok(())
}

#[test]
fn repeats_single_row() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(PrimitiveArray::from_option_iter([Some(7u16)]), 4)?;

    assert_arrays_eq!(
        array,
        PrimitiveArray::from_option_iter([Some(7u16); 4]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn scalar_at_wraps() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(buffer![10u32, 20, 30], 5)?;

    for (index, expected) in [(0, 10u32), (2, 30), (3, 10), (7, 20), (14, 30)] {
        assert_eq!(
            array.execute_scalar(index, &mut ctx)?,
            Scalar::from(expected)
        );
    }
    Ok(())
}

#[test]
fn append_to_builder() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(buffer![1i8, 2], 3)?;
    let mut builder = builder_with_capacity_in(array.dtype(), array.len() + 1, ctx.allocator());
    builder.append_scalar(&Scalar::from(0i8))?;
    array.append_to_builder(builder.as_mut(), &mut ctx)?;

    assert_arrays_eq!(
        builder.finish(),
        PrimitiveArray::from_iter([0i8, 1, 2, 1, 2, 1, 2]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::within_first_copy(0..2, vec![0, 1])]
#[case::within_later_copy(5..8, vec![1, 2, 3])]
#[case::exactly_one_copy(4..8, vec![0, 1, 2, 3])]
#[case::whole_copies(4..12, vec![0, 1, 2, 3, 0, 1, 2, 3])]
#[case::across_copies(3..6, vec![3, 0, 1])]
#[case::across_many_copies(1..11, vec![1, 2, 3, 0, 1, 2, 3, 0, 1, 2])]
fn slice(#[case] range: Range<usize>, #[case] expected: Vec<i32>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(buffer![0i32, 1, 2, 3], 3)?;

    assert_arrays_eq!(
        array.slice(range)?,
        PrimitiveArray::from_iter(expected),
        &mut ctx
    );
    Ok(())
}

#[test]
fn slice_of_whole_copies_stays_repeated() -> VortexResult<()> {
    let array = repeated(buffer![0i32, 1, 2, 3], 3)?;
    let sliced = array.slice(4..12)?;

    let repeated = sliced.as_::<Repeated>();
    assert_eq!(repeated.count(), 2);
    assert_eq!(repeated.child().len(), 4);
    Ok(())
}

#[test]
fn reduce_single_count_to_child() -> VortexResult<()> {
    let child = buffer![1i32, 2].into_array();
    let reduced = repeated(child.clone(), 1)?.optimize()?;

    assert!(reduced.is::<Primitive>());
    assert_eq!(reduced.len(), child.len());
    Ok(())
}

#[test]
fn reduce_constant_child() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let reduced = repeated(ConstantArray::new(5i32, 3), 4)?.optimize()?;

    assert!(reduced.is::<Constant>());
    assert_arrays_eq!(reduced, PrimitiveArray::from_iter([5i32; 12]), &mut ctx);
    Ok(())
}

#[test]
fn reduce_nested_repeats() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let reduced = repeated(repeated(buffer![1i32, 2], 2)?, 3)?.optimize()?;

    let flattened = reduced.as_::<Repeated>();
    assert_eq!(flattened.count(), 6);
    assert!(!flattened.child().is::<Repeated>());
    assert_arrays_eq!(
        reduced,
        PrimitiveArray::from_iter([1i32, 2].repeat(6)),
        &mut ctx
    );
    Ok(())
}

#[test]
fn rejects_overflowing_length() {
    let result = RepeatedArray::try_new(buffer![1u8, 2].into_array(), usize::MAX);
    assert!(result.is_err());
}
