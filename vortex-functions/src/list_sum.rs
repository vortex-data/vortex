// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynGroupedAccumulator;
use vortex_array::aggregate_fn::GroupedAccumulator;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// Sum of the elements in each list of a `List` or `FixedSizeList` typed array.
///
/// Follows SQL `SUM` semantics per list, matching DuckDB's `list_sum`: null lists, empty
/// lists, and lists whose elements are all null yield a null sum; null elements are skipped.
/// Integer and decimal overflow yields a null sum value, matching [`SumV2`]. The result dtype
/// follows [`SumV2`]'s widening rules and is always nullable.
///
/// NaN handling for float elements is controlled by [`NumericalAggregateOpts`]: with
/// `skip_nans` (the default) NaN values contribute nothing, otherwise any NaN poisons the
/// list's sum to NaN.
#[derive(Clone)]
pub struct ListSum;

impl ScalarFnVTable for ListSum {
    type Options = NumericalAggregateOpts;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.list.sum");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.serialize()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        NumericalAggregateOpts::deserialize(metadata)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for list_sum()"),
        }
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let elem_dtype = match &arg_dtypes[0] {
            DType::List(elem, _) | DType::FixedSizeList(elem, ..) => elem.as_ref(),
            other => vortex_bail!("list_sum() requires List or FixedSizeList, got {other}"),
        };
        SumV2
            .return_dtype(options, elem_dtype)
            .ok_or_else(|| vortex_err!("list_sum() cannot sum elements of type {elem_dtype}"))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;

        let elem_dtype = match input.dtype() {
            DType::List(elem, _) | DType::FixedSizeList(elem, ..) => elem.as_ref().clone(),
            other => vortex_bail!("list_sum() requires List or FixedSizeList, got {other}"),
        };

        let columnar = input.execute::<Columnar>(ctx)?;

        match columnar {
            Columnar::Constant(constant) => {
                // Canonicalize one row of the constant and broadcast its sum.
                let one_row = ConstantArray::new(constant.scalar().clone(), 1)
                    .into_array()
                    .execute::<Canonical>(ctx)?
                    .into_array();
                let sum =
                    list_sum_impl(one_row, elem_dtype, options, ctx)?.execute_scalar(0, ctx)?;
                Ok(ConstantArray::new(sum, constant.len()).into_array())
            }
            Columnar::Canonical(canonical) => {
                list_sum_impl(canonical.into_array(), elem_dtype, options, ctx)
            }
        }
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        // A null list sums to null, which is all that strictness requires. Element nulls are part
        // of the list value rather than row-level nulls, and a valid list may still sum to null
        // (empty, all-null, or overflow) since strictness is only one-directional.
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

/// Sum each list of a canonical `array` into one value per list.
fn list_sum_impl(
    canonical: ArrayRef,
    elem_dtype: DType,
    options: &NumericalAggregateOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut acc = GroupedAccumulator::try_new(SumV2, *options, elem_dtype)?;
    acc.accumulate_list(&canonical, ctx)?;
    acc.finish()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use prost::Message;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::proto::expr as pb;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::Expression;
    use vortex_expr::proto::ExprSerializeProtoExt;
    use vortex_expr::root;
    use vortex_session::VortexSession;

    use crate::expr::list_sum;
    use crate::expr::list_sum_opts;
    use crate::list_sum::ListSum;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();

        crate::initialize(&session);

        session
    });

    fn create_list_elements() -> ArrayRef {
        PrimitiveArray::from_option_iter::<i32, _>([
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            None,
        ])
        .into_array()
    }

    #[rstest]
    #[case(buffer![0u32, 2, 5, 5, 7].into_array())]
    #[case(buffer![0u64, 2, 5, 5, 7].into_array())]
    fn test_list_sum(#[case] offsets: ArrayRef) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = create_list_elements();
        let list =
            ListArray::try_new(elements, offsets, Validity::NonNullable, &mut ctx)?.into_array();
        let result = list.apply(&list_sum(root()))?;
        let mut ctx = SESSION.create_execution_ctx();
        // [1, 2] = 3; [3, 4, 5] = 12; [] = null; [6, null] = 6 (nulls skipped).
        let expected =
            PrimitiveArray::from_option_iter::<i64, _>([Some(3), Some(12), None, Some(6)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nullable_list_sum() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = create_list_elements();
        let list = ListArray::try_new(
            elements,
            buffer![0u32, 2, 5, 5, 7].into_array(),
            Validity::Array(BoolArray::from_iter([true, false, true, false]).into_array()),
            &mut ctx,
        )?
        .into_array();
        let result = list.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        // Row 1 and 3 are null lists; row 2 is a valid but empty list.
        let expected = PrimitiveArray::from_option_iter::<i64, _>([Some(3), None, None, None]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_all_null_elements_sum_to_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = PrimitiveArray::from_option_iter::<i32, _>([None, None, Some(1)]);
        let list = ListArray::try_new(
            elements.into_array(),
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let result = list.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter::<i64, _>([None, Some(1)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_overflow_sums_to_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = PrimitiveArray::from_iter([i64::MAX, 1, 1, 2]);
        let list = ListArray::try_new(
            elements.into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let result = list.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter::<i64, _>([None, Some(3)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_unsigned_widening() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = PrimitiveArray::from_iter([1u8, 2, 3]);
        let list = ListArray::try_new(
            elements.into_array(),
            buffer![0u32, 3].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let result = list.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter::<u64, _>([Some(6)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_bool_elements() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = BoolArray::from_iter([true, true, false, true]);
        let list = ListArray::try_new(
            elements.into_array(),
            buffer![0u32, 3, 4].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let result = list.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter::<u64, _>([Some(2), Some(1)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nan_skipped_by_default() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = PrimitiveArray::from_iter([1.0f64, f64::NAN, 2.0]);
        let list = ListArray::try_new(
            elements.into_array(),
            buffer![0u32, 3].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let result = list.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter::<f64, _>([Some(3.0)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_all_nan_list_sums_to_zero() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // NaN elements are *valid*: with the default `skip_nans` they contribute nothing, but
        // the list still has valid elements, so the sum is `0.0` rather than null (unlike an
        // all-null list).
        let elements = PrimitiveArray::from_iter([f64::NAN, f64::NAN]);
        let list = ListArray::try_new(
            elements.into_array(),
            buffer![0u32, 2].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let result = list.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter::<f64, _>([Some(0.0)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_nan_poisons_with_include_nans() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = PrimitiveArray::from_iter([1.0f64, f64::NAN, 2.0]);
        let list = ListArray::try_new(
            elements.into_array(),
            buffer![0u32, 3].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let result = list.apply(&list_sum_opts(
            root(),
            NumericalAggregateOpts::include_nans(),
        ))?;

        let mut ctx = SESSION.create_execution_ctx();
        assert!(result.is_valid(0, &mut ctx)?);
        let result = result.execute::<PrimitiveArray>(&mut ctx)?;
        assert!(result.as_slice::<f64>()[0].is_nan());
        Ok(())
    }

    #[test]
    fn test_listview_sum() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = create_list_elements();
        // Overlapping and out-of-order views over the shared elements.
        let lv = ListViewArray::new(
            elements,
            buffer![5u32, 0, 4, 1].into_array(),
            buffer![2u32, 3, 0, 2].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )
        .into_array();
        let result = lv.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        // [6, null] = 6; [1, 2, 3] = 6; [] = null; [2, 3] = 5.
        let expected =
            PrimitiveArray::from_option_iter::<i64, _>([Some(6), Some(6), None, Some(5)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    fn create_fixed_size_list(validity: Validity) -> ArrayRef {
        // 4 lists of size 2 over 8 primitive elements.
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8]).into_array();
        FixedSizeListArray::new(elements, 2, validity, 4).into_array()
    }

    #[test]
    fn test_fixed_size_list_sum() -> VortexResult<()> {
        let fsl = create_fixed_size_list(Validity::NonNullable);
        let result = fsl.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected =
            PrimitiveArray::from_option_iter::<i64, _>([Some(3), Some(7), Some(11), Some(15)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_fixed_size_list_sum_nullable() -> VortexResult<()> {
        let fsl = create_fixed_size_list(Validity::Array(
            BoolArray::from_iter([true, false, true, false]).into_array(),
        ));
        let result = fsl.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter::<i64, _>([Some(3), None, Some(11), None]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_list_sum_take() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = create_list_elements();
        let list = ListArray::try_new(
            elements,
            buffer![0u32, 2, 5, 5, 7].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let taken = list.take(buffer![3u64, 0, 2].into_array())?;

        let result = taken.apply(&list_sum(root()))?;
        let mut ctx = SESSION.create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter::<i64, _>([Some(6), Some(3), None]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_list_sum_slice() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = create_list_elements();
        let list = ListArray::try_new(
            elements,
            buffer![0u32, 2, 5, 5, 7].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let sliced = list.slice(1..4)?;

        let result = sliced.apply(&list_sum(root()))?;
        let mut ctx = SESSION.create_execution_ctx();
        let expected = PrimitiveArray::from_option_iter::<i64, _>([Some(12), None, Some(6)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_empty_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = PrimitiveArray::from_iter([0i32; 0]);
        let list = ListArray::try_new(
            elements.into_array(),
            buffer![0u32].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let result = list.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter::<i64, _>([]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_constant_list_sum() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements = create_list_elements();
        let list = ListArray::try_new(
            elements,
            buffer![0u32, 2, 5, 5, 7].into_array(),
            Validity::NonNullable,
            &mut ctx,
        )?
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let scalar = list.execute_scalar(1, &mut ctx)?;

        let constant = ConstantArray::new(scalar, 3).into_array();
        let result = constant.apply(&list_sum(root()))?;
        let expected = PrimitiveArray::from_option_iter::<i64, _>([Some(12), Some(12), Some(12)]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_null_scalar_list_sum() -> VortexResult<()> {
        let null_scalar = Scalar::null(DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        ));
        let array = ConstantArray::new(null_scalar, 2).into_array();
        let result = array.apply(&list_sum(root()))?;

        let mut ctx = SESSION.create_execution_ctx();
        assert!(!result.is_valid(0, &mut ctx)?);
        assert!(!result.is_valid(1, &mut ctx)?);
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([None::<i64>, None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn test_unsupported_dtypes() {
        let opts = NumericalAggregateOpts::default();
        // Non-list inputs are rejected.
        assert!(
            ListSum
                .return_dtype(
                    &opts,
                    &[DType::Primitive(PType::I32, Nullability::NonNullable)]
                )
                .is_err()
        );
        // Non-numeric element types are rejected.
        assert!(
            ListSum
                .return_dtype(
                    &opts,
                    &[DType::List(
                        Arc::new(DType::Utf8(Nullability::NonNullable)),
                        Nullability::NonNullable,
                    )],
                )
                .is_err()
        );
    }

    #[test]
    fn test_display() {
        assert_eq!(list_sum(root()).to_string(), "vortex.list.sum($)");
        assert_eq!(
            list_sum_opts(root(), NumericalAggregateOpts::include_nans()).to_string(),
            "vortex.list.sum($, opts=skip_nans=false)"
        );
    }

    #[test]
    fn test_proto_round_trip() -> VortexResult<()> {
        for expr in [
            list_sum(root()),
            list_sum_opts(root(), NumericalAggregateOpts::include_nans()),
        ] {
            let proto = expr.serialize_proto()?;
            let buf = proto.encode_to_vec();
            let decoded = pb::Expr::decode(buf.as_slice())?;
            let session = VortexSession::empty();
            crate::initialize(&session);
            let deser = Expression::from_proto(&decoded, &session)?;
            assert_eq!(expr, deser);
        }
        Ok(())
    }
}
