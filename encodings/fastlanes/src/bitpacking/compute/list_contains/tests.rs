// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::list_contains;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementKernel;
#[cfg(not(codspeed))]
use vortex_array::test_harness::trace::trace_op;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::BitPackedArrayExt;
use crate::BitPackedData;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn member_list<T>(
    values: impl IntoIterator<Item = Option<T>>,
    member_nullability: Nullability,
) -> Scalar
where
    T: NativePType + Into<PValue>,
{
    let member_dtype = DType::Primitive(T::PTYPE, member_nullability);
    let members = values
        .into_iter()
        .map(|value| {
            value
                .map(|value| Scalar::primitive(value, member_nullability))
                .unwrap_or_else(|| Scalar::null(member_dtype.clone()))
        })
        .collect();
    Scalar::list(Arc::new(member_dtype), members, Nullability::NonNullable)
}

fn list_array(list: Scalar, len: usize) -> ArrayRef {
    ConstantArray::new(list, len).into_array()
}

fn execute_direct(
    list: &ArrayRef,
    element: &BitPackedArray,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<BoolArray> {
    <BitPacked as ListContainsElementKernel>::list_contains(list, element.as_view(), ctx)?
        .ok_or_else(|| vortex_err!("BitPacked list_contains kernel declined a supported input"))?
        .execute::<BoolArray>(ctx)
}

macro_rules! integer_type_test {
    ($name:ident, $T:ty, $bit_width:expr) => {
        #[test]
        fn $name() -> VortexResult<()> {
            let mut ctx = SESSION.create_execution_ctx();
            let values = (0..2_048)
                .map(|value| (value % 64) as $T)
                .collect::<Vec<_>>();
            let members = [1 as $T, 3 as $T, 63 as $T];
            let primitive = PrimitiveArray::from_iter(values.iter().copied());
            let packed = BitPackedData::encode(&primitive.into_array(), $bit_width, &mut ctx)?;
            let list = list_array(
                member_list(members.into_iter().map(Some), Nullability::NonNullable),
                packed.len(),
            );

            let actual = execute_direct(&list, &packed, &mut ctx)?;
            let expected =
                BoolArray::from_iter(values.into_iter().map(|value| members.contains(&value)));
            assert_arrays_eq!(actual, expected, &mut ctx);
            Ok(())
        }
    };
}

integer_type_test!(test_integer_type_u8, u8, 6);
integer_type_test!(test_integer_type_u16, u16, 6);
integer_type_test!(test_integer_type_u32, u32, 6);
integer_type_test!(test_integer_type_u64, u64, 6);
integer_type_test!(test_integer_type_i8, i8, 6);
integer_type_test!(test_integer_type_i16, i16, 6);
integer_type_test!(test_integer_type_i32, i32, 6);
integer_type_test!(test_integer_type_i64, i64, 6);

#[rstest]
#[case::one(vec![3])]
#[case::two(vec![3, 7])]
#[case::three(vec![3, 7, 11])]
#[case::four(vec![3, 7, 11, 15])]
#[case::duplicate_source(vec![3, 3, 7, 7, 11, 11, 15, 15, 15])]
fn test_member_cardinalities(#[case] members: Vec<i32>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0..2_048).map(|value| value % 128).collect::<Vec<_>>();
    let primitive = PrimitiveArray::from_iter(values.iter().copied());
    let packed = BitPackedData::encode(&primitive.into_array(), 7, &mut ctx)?;
    let list = list_array(
        member_list(members.iter().copied().map(Some), Nullability::NonNullable),
        packed.len(),
    );

    let actual = execute_direct(&list, &packed, &mut ctx)?;
    let expected = BoolArray::from_iter(values.into_iter().map(|value| members.contains(&value)));
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::generic_five((0..5).map(|value| value * 2).collect())]
#[case::decoded_many((0..32).map(|value| value * 2).collect())]
fn test_many_member_public_expression_paths(#[case] members: Vec<i32>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0..4_096).map(|value| value % 128).collect::<Vec<i32>>();
    let primitive = PrimitiveArray::from_iter(values.iter().copied());
    let packed = BitPackedData::encode(&primitive.into_array(), 7, &mut ctx)?;
    let expression = list_contains(
        lit(member_list(
            members.iter().copied().map(Some),
            Nullability::NonNullable,
        )),
        root(),
    );

    let actual = packed
        .into_array()
        .apply(&expression)?
        .execute::<BoolArray>(&mut ctx)?;
    let expected = BoolArray::from_iter(values.into_iter().map(|value| members.contains(&value)));
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_many_member_kernel_policy() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = [0i32, 7, 99];
    let primitive = PrimitiveArray::from_iter(values);
    let packed = BitPackedData::encode(&primitive.into_array(), 7, &mut ctx)?;
    let decode_threshold = super::min_decode_source_members(PType::I32, packed.len());

    for (member_count, expected_supported) in
        [(decode_threshold - 1, false), (decode_threshold, true)]
    {
        let member_count = i32::try_from(member_count).vortex_expect("member count fits in an i32");
        let list = list_array(
            member_list((0..member_count).map(Some), Nullability::NonNullable),
            packed.len(),
        );
        let actual = <BitPacked as ListContainsElementKernel>::list_contains(
            &list,
            packed.as_view(),
            &mut ctx,
        )?;

        assert_eq!(actual.is_some(), expected_supported);
        if let Some(actual) = actual {
            let expected =
                BoolArray::from_iter(values.map(|value| (0..member_count).contains(&value)));
            assert_arrays_eq!(actual, expected, &mut ctx);
        }
    }
    Ok(())
}

#[rstest]
#[case::present([true; 128], vec![0])]
#[case::absent([false; 128], vec![1])]
fn test_zero_bit_width(
    #[case] expected: [bool; 128],
    #[case] members: Vec<i32>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let primitive = PrimitiveArray::from_iter([0i32; 128]);
    let packed = BitPackedData::encode(&primitive.into_array(), 0, &mut ctx)?;
    let list = list_array(
        member_list(members.into_iter().map(Some), Nullability::NonNullable),
        packed.len(),
    );

    let actual = execute_direct(&list, &packed, &mut ctx)?;
    let expected = BoolArray::from_iter(expected);
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::fused(vec![3, 100_388])]
#[case::decoded({
    let mut members = (0..31).collect::<Vec<_>>();
    members.push(100_388);
    members
})]
fn test_sliced_patched_array(#[case] members: Vec<i32>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0..5_000)
        .map(|index| {
            if index % 97 == 0 {
                100_000 + index
            } else {
                index % 100
            }
        })
        .collect::<Vec<i32>>();
    let primitive = PrimitiveArray::from_iter(values.iter().copied());
    let packed = BitPackedData::encode(&primitive.into_array(), 7, &mut ctx)?;
    assert!(packed.patches().is_some(), "test setup requires patches");
    let range = 333..4_333;
    let sliced = <BitPacked as SliceKernel>::slice(packed.as_view(), range.clone(), &mut ctx)?
        .ok_or_else(|| vortex_err!("BitPacked slice kernel declined a supported input"))?;
    let list = list_array(
        member_list(members.iter().copied().map(Some), Nullability::NonNullable),
        sliced.len(),
    );

    let actual = <BitPacked as ListContainsElementKernel>::list_contains(
        &list,
        sliced.as_::<BitPacked>(),
        &mut ctx,
    )?
    .ok_or_else(|| vortex_err!("BitPacked list_contains kernel declined a sliced input"))?
    .execute::<BoolArray>(&mut ctx)?;
    let expected = BoolArray::from_iter(values[range].iter().map(|value| members.contains(value)));
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::nullable_needles(
    vec![Some(1), Some(3)],
    Nullability::NonNullable,
    vec![Some(1), None, Some(2)],
    vec![Some(true), None, Some(false)],
)]
#[case::nullable_members(
    vec![Some(1), None, Some(3)],
    Nullability::Nullable,
    vec![Some(1), Some(2), Some(3)],
    vec![Some(true), Some(false), Some(true)],
)]
#[case::all_null_members(
    vec![None, None],
    Nullability::Nullable,
    vec![Some(1), None, Some(2)],
    vec![Some(false), None, Some(false)],
)]
fn test_null_semantics(
    #[case] members: Vec<Option<i32>>,
    #[case] member_nullability: Nullability,
    #[case] values: Vec<Option<i32>>,
    #[case] expected: Vec<Option<bool>>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let primitive = PrimitiveArray::from_option_iter(values);
    let packed = BitPackedData::encode(&primitive.into_array(), 3, &mut ctx)?;
    let list = list_array(member_list(members, member_nullability), packed.len());

    let actual = execute_direct(&list, &packed, &mut ctx)?;
    let expected = BoolArray::from_iter(expected);
    assert_arrays_eq!(actual, expected, &mut ctx);
    Ok(())
}

#[test]
fn test_wrong_integer_type_declines_without_panic() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let primitive = PrimitiveArray::from_iter([1i32, 2, 3]);
    let packed = BitPackedData::encode(&primitive.into_array(), 2, &mut ctx)?;
    let list = list_array(
        member_list([Some(1i64), Some(3)], Nullability::NonNullable),
        packed.len(),
    );

    let result =
        <BitPacked as ListContainsElementKernel>::list_contains(&list, packed.as_view(), &mut ctx)?;
    assert!(result.is_none());
    Ok(())
}

#[test]
fn test_nonconstant_list_declines() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let primitive = PrimitiveArray::from_iter([1i32, 2, 3]);
    let packed = BitPackedData::encode(&primitive.into_array(), 2, &mut ctx)?;
    let list = ListArray::from_iter_slow::<u32, _>(
        vec![vec![1i32], vec![2], vec![3]],
        Arc::new(DType::Primitive(i32::PTYPE, Nullability::NonNullable)),
    )?
    .into_array();

    let result =
        <BitPacked as ListContainsElementKernel>::list_contains(&list, packed.as_view(), &mut ctx)?;
    assert!(result.is_none());
    Ok(())
}

#[test]
#[cfg(not(codspeed))]
fn test_registered_kernel_executes_through_expression() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0..2_048).map(|value| value % 128).collect::<Vec<i32>>();
    let primitive = PrimitiveArray::from_iter(values.iter().copied());
    let packed = BitPackedData::encode(&primitive.into_array(), 7, &mut ctx)?;
    let members = [0, 99];
    let expression = list_contains(
        lit(member_list(
            members.into_iter().map(Some),
            Nullability::NonNullable,
        )),
        root(),
    );
    let contains = packed.into_array().apply(&expression)?;

    let traced = trace_op(|| contains.execute::<BoolArray>(&mut ctx))?;
    let trace = traced.trace.to_string();
    let applied = trace
        .lines()
        .filter(|line| {
            line.contains("child_execute_parent session[")
                && line.contains("slot=1")
                && line.contains("parent=vortex.list.contains")
                && line.contains("child=fastlanes.bitpacked")
        })
        .collect::<Vec<_>>();
    // A silent fallback preserves values but loses compressed-domain execution.
    assert_eq!(applied.len(), 1, "{trace}");

    let expected = BoolArray::from_iter(values.into_iter().map(|value| members.contains(&value)));
    assert_arrays_eq!(traced.output, expected, &mut ctx);
    Ok(())
}
