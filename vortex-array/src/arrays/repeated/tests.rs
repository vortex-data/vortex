// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Repeated;
use crate::arrays::RepeatedArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::extension::ExtensionArraySlotsExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::repeated::RepeatedArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::assert_arrays_eq;
use crate::builders::VariantBuilder;
use crate::builders::builder_with_capacity_in;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::validity::Validity;

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
    let sliced = array.slice(range.clone())?.optimize_ctx(&SESSION)?;

    let repeated = sliced.as_::<Repeated>();
    assert_eq!(repeated.len(), range.len());
    assert_eq!(repeated.child().len(), 4);
    Ok(())
}

#[rstest]
#[case::whole_child(2)]
#[case::prefix_of_child(1)]
fn reduce_to_child_prefix(#[case] len: usize) -> VortexResult<()> {
    let reduced = repeated(buffer![1i32, 2], len)?.optimize_ctx(&SESSION)?;

    assert!(reduced.is::<Primitive>());
    assert_eq!(reduced.len(), len);
    Ok(())
}

#[test]
fn reduce_constant_child() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let reduced = repeated(ConstantArray::new(5i32, 3), 10)?.optimize_ctx(&SESSION)?;

    assert!(reduced.is::<Constant>());
    assert_arrays_eq!(reduced, PrimitiveArray::from_iter([5i32; 10]), &mut ctx);
    Ok(())
}

#[test]
fn reduce_nested_whole_copies() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let reduced = repeated(repeated(buffer![1i32, 2], 4)?, 11)?.optimize_ctx(&SESSION)?;

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
    let reduced = repeated(repeated(buffer![1i32, 2, 3], 4)?, 8)?.optimize_ctx(&SESSION)?;

    assert!(reduced.as_::<Repeated>().child().is::<Repeated>());
    assert_arrays_eq!(
        reduced,
        PrimitiveArray::from_iter([1i32, 2, 3, 1, 1, 2, 3, 1]),
        &mut ctx
    );
    Ok(())
}

/// Three rows of a nullable struct with a nullable field.
fn struct_child() -> ArrayRef {
    StructArray::try_from_iter_with_validity(
        [
            (
                "a",
                PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
            ),
            (
                "b",
                VarBinViewArray::from_iter_str(["x", "y", "z"]).into_array(),
            ),
        ],
        Validity::Array(BoolArray::from_iter([true, true, false]).into_array()),
    )
    .expect("valid struct fixture")
    .into_array()
}

/// Three nullable lists of two elements.
fn fixed_size_list_child() -> ArrayRef {
    FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6].into_array(),
        2,
        Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        3,
    )
    .into_array()
}

/// The lists `[1, 2]`, `[3]` and `[4, 5, 6]`.
fn list_view_child() -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    ListViewArray::new(
        buffer![1i32, 2, 3, 4, 5, 6].into_array(),
        buffer![0u32, 2, 3].into_array(),
        buffer![2u32, 1, 3].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )
    .into_array()
}

/// Three timestamps.
fn extension_child() -> ArrayRef {
    ExtensionArray::try_new(
        Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        buffer![1i64, 2, 3].into_array(),
        &mut SESSION.create_execution_ctx(),
    )
    .expect("test storage is valid")
    .into_array()
}

/// Three Variants holding `1`, `2` and `3`, shredded as `i32`.
fn variant_child() -> ArrayRef {
    let core = ChunkedArray::try_new(
        (1..=3).map(|value| {
            let scalar = Scalar::variant(Scalar::primitive(value, Nullability::NonNullable));
            ConstantArray::new(scalar, 1).into_array()
        }),
        DType::Variant(Nullability::NonNullable),
    )
    .expect("valid Variant core fixture");
    VariantArray::try_new(core.into_array(), Some(buffer![1i32, 2, 3].into_array()))
        .expect("valid Variant fixture")
        .into_array()
}

/// Executing repeats the children of a nested child, rather than copying its rows.
#[rstest]
#[case::struct_(struct_child(), |array: &ArrayRef| array.as_::<Struct>().unmasked_field(0).clone())]
#[case::fixed_size_list(
    fixed_size_list_child(),
    |array: &ArrayRef| array.as_::<FixedSizeList>().elements().clone()
)]
#[case::list_view(list_view_child(), |array: &ArrayRef| array.as_::<ListView>().offsets().clone())]
#[case::extension(extension_child(), |array: &ArrayRef| array.as_::<Extension>().storage().clone())]
#[case::variant(
    variant_child(),
    |array: &ArrayRef| array.as_::<Variant>().core_storage().clone()
)]
fn executes_nested_children_repeated(
    #[case] child: ArrayRef,
    #[case] repeated_child: fn(&ArrayRef) -> ArrayRef,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 2 * child.len() + 1;
    let executed = repeated(child.clone(), len)?
        .execute::<Canonical>(&mut ctx)?
        .into_array();

    assert!(repeated_child(&executed).is::<Repeated>());
    let copies = ChunkedArray::try_new(
        [child.clone(), child.clone(), child.slice(0..1)?],
        child.dtype().clone(),
    )?;
    assert_arrays_eq!(executed, copies, &mut ctx);
    Ok(())
}

/// A Variant builder keeps the repeated core storage and shredded child as one array each.
#[test]
fn appends_variant_as_one_repeated_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child = variant_child();
    let mut builder = VariantBuilder::new(Nullability::NonNullable);
    repeated(child.clone(), 7)?.append_to_builder(&mut builder, &mut ctx)?;
    let built = builder.finish_into_variant()?;

    assert!(built.core_storage().is::<Repeated>());
    assert!(
        built
            .shredded()
            .is_some_and(|shredded| shredded.is::<Repeated>())
    );
    let copies = ChunkedArray::try_new(
        [child.clone(), child.clone(), child.slice(0..1)?],
        child.dtype().clone(),
    )?;
    assert_arrays_eq!(built, copies, &mut ctx);
    Ok(())
}

/// A struct builder keeps each repeated field as one chunk.
#[test]
fn appends_struct_fields_as_one_repeated_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let child = struct_child();
    let mut builder = builder_with_capacity_in(child.dtype(), 7, ctx.allocator());
    repeated(child.clone(), 7)?.append_to_builder(builder.as_mut(), &mut ctx)?;
    let built = builder.finish();

    let fields = built.as_::<Struct>();
    assert!(
        fields
            .iter_unmasked_fields()
            .all(|field| field.is::<Repeated>())
    );
    let copies = ChunkedArray::try_new(
        [child.clone(), child.clone(), child.slice(0..1)?],
        child.dtype().clone(),
    )?;
    assert_arrays_eq!(built, copies, &mut ctx);
    Ok(())
}
