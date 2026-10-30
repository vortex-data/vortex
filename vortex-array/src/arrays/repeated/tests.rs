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

fn repeated(child: impl IntoArray, len: usize) -> VortexResult<ArrayRef> {
    Ok(RepeatedArray::try_new(child.into_array(), len)?.into_array())
}

#[rstest]
#[case::whole_copies(9, vec![1, 2, 3, 1, 2, 3, 1, 2, 3])]
#[case::partial_last_copy(7, vec![1, 2, 3, 1, 2, 3, 1])]
#[case::shorter_than_child(2, vec![1, 2])]
fn repeats_primitive(#[case] len: usize, #[case] expected: Vec<i32>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(buffer![1i32, 2, 3], len)?;

    assert_eq!(array.len(), len);
    assert_arrays_eq!(array, PrimitiveArray::from_iter(expected), &mut ctx);
    Ok(())
}

#[test]
fn repeats_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(PrimitiveArray::from_option_iter([Some(1i64), None]), 5)?;

    assert_arrays_eq!(
        array,
        PrimitiveArray::from_option_iter([Some(1i64), None, Some(1), None, Some(1)]),
        &mut ctx
    );
    assert_eq!(array.invalid_count(&mut ctx)?, 2);
    Ok(())
}

#[test]
fn repeats_strings() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child = VarBinViewArray::from_iter_nullable_str([Some("a"), None, Some("a long string")]);
    let array = repeated(child, 5)?;

    assert_arrays_eq!(
        array,
        VarBinViewArray::from_iter_nullable_str([
            Some("a"),
            None,
            Some("a long string"),
            Some("a"),
            None,
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
        ("a", buffer![1u8, 2, 1, 2, 1].into_array()),
        (
            "b",
            VarBinViewArray::from_iter_str(["x", "y", "x", "y", "x"]).into_array(),
        ),
    ])?;

    assert_arrays_eq!(repeated(child, 5)?, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::zero_len(buffer![1i32, 2, 3].into_array())]
#[case::empty_child(PrimitiveArray::from_iter(Vec::<i32>::new()).into_array())]
fn repeats_to_empty(#[case] child: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(child, 0)?;

    assert!(array.is_empty());
    assert_arrays_eq!(
        array,
        PrimitiveArray::from_iter(Vec::<i32>::new()),
        &mut ctx
    );
    Ok(())
}

#[test]
fn rejects_empty_child_with_rows() {
    let child = PrimitiveArray::from_iter(Vec::<i32>::new()).into_array();
    assert!(RepeatedArray::try_new(child, 4).is_err());
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
    let array = repeated(buffer![10u32, 20, 30], 16)?;

    for (index, expected) in [(0, 10u32), (2, 30), (3, 10), (7, 20), (14, 30), (15, 10)] {
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
    let array = repeated(buffer![1i8, 2], 5)?;
    let mut builder = builder_with_capacity_in(array.dtype(), array.len() + 1, ctx.allocator());
    builder.append_scalar(&Scalar::from(0i8))?;
    array.append_to_builder(builder.as_mut(), &mut ctx)?;

    assert_arrays_eq!(
        builder.finish(),
        PrimitiveArray::from_iter([0i8, 1, 2, 1, 2, 1]),
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
#[case::one_unaligned_copy(3..7, vec![3, 0, 1, 2])]
#[case::unaligned_whole_copies(1..9, vec![1, 2, 3, 0, 1, 2, 3, 0])]
#[case::into_partial_last_copy(6..14, vec![2, 3, 0, 1, 2, 3, 0, 1])]
fn slice(#[case] range: Range<usize>, #[case] expected: Vec<i32>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = repeated(buffer![0i32, 1, 2, 3], 14)?;

    assert_arrays_eq!(
        array.slice(range)?,
        PrimitiveArray::from_iter(expected),
        &mut ctx
    );
    Ok(())
}

/// Any slice that spans more than one copy stays repeated, at exactly the slice's length.
#[rstest]
#[case::aligned_whole_copies(4..12)]
#[case::unaligned_whole_copies(1..9)]
#[case::partial_copies(1..11)]
#[case::into_partial_last_copy(6..14)]
fn slice_stays_repeated(#[case] range: Range<usize>) -> VortexResult<()> {
    let array = repeated(buffer![0i32, 1, 2, 3], 14)?;
    let sliced = array.slice(range.clone())?;

    let repeated = sliced.as_::<Repeated>();
    assert_eq!(repeated.len(), range.len());
    assert_eq!(repeated.child().len(), 4);
    Ok(())
}

#[rstest]
#[case::whole_child(2)]
#[case::prefix_of_child(1)]
fn reduce_to_child_prefix(#[case] len: usize) -> VortexResult<()> {
    let reduced = repeated(buffer![1i32, 2], len)?.optimize()?;

    assert!(reduced.is::<Primitive>());
    assert_eq!(reduced.len(), len);
    Ok(())
}

#[test]
fn reduce_constant_child() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let reduced = repeated(ConstantArray::new(5i32, 3), 10)?.optimize()?;

    assert!(reduced.is::<Constant>());
    assert_arrays_eq!(reduced, PrimitiveArray::from_iter([5i32; 10]), &mut ctx);
    Ok(())
}

#[test]
fn reduce_nested_whole_copies() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let reduced = repeated(repeated(buffer![1i32, 2], 4)?, 11)?.optimize()?;

    let flattened = reduced.as_::<Repeated>();
    assert_eq!(flattened.len(), 11);
    assert!(!flattened.child().is::<Repeated>());
    assert_arrays_eq!(
        reduced,
        PrimitiveArray::from_iter([1i32, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]),
        &mut ctx
    );
    Ok(())
}

/// An inner repeat that cuts its child short is not the child repeated, so it stays nested.
#[test]
fn nested_partial_copies_stay_nested() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let reduced = repeated(repeated(buffer![1i32, 2, 3], 4)?, 8)?.optimize()?;

    assert!(reduced.as_::<Repeated>().child().is::<Repeated>());
    assert_arrays_eq!(
        reduced,
        PrimitiveArray::from_iter([1i32, 2, 3, 1, 1, 2, 3, 1]),
        &mut ctx
    );
    Ok(())
}
