// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow-backed implementations of the compute hooks declared in
//! [`vortex_array::arrow_hooks`].
//!
//! Linking this crate registers these implementations as the default fallback
//! compute kernels used when `vortex-array` does not have a specialised native path.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::BooleanArray;
use arrow_buffer::NullBuffer;
use arrow_ord::cmp;
use arrow_ord::ord::make_comparator;
use arrow_schema::DataType;
use arrow_schema::SortOptions;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::aliases::inventory;
use vortex_array::array::Array;
use vortex_array::arrow_hooks::ArrowCompute;
use vortex_array::arrow_hooks::ArrowComputeRegistration;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::NumericOperator;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::BitBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::BitBufferIntoArrow;
use crate::Datum;
use crate::FromArrowArray;
use crate::IntoArrowArray;
use crate::from_arrow_array_with_len;

fn arrow_compare(
    left: &ArrayRef,
    right: &ArrayRef,
    operator: CompareOperator,
) -> VortexResult<ArrayRef> {
    assert_eq!(left.len(), right.len());

    let nullable = left.dtype().is_nullable() || right.dtype().is_nullable();

    // Arrow's vectorized comparison kernels don't support nested types.
    let arrow_array: BooleanArray = if left.dtype().is_nested() || right.dtype().is_nested() {
        let rhs = right.clone().into_arrow_preferred()?;
        let lhs = left.clone().into_arrow(rhs.data_type())?;

        assert!(
            lhs.data_type().equals_datatype(rhs.data_type()),
            "lhs data_type: {}, rhs data_type: {}",
            lhs.data_type(),
            rhs.data_type()
        );

        compare_nested_arrow_arrays(lhs.as_ref(), rhs.as_ref(), operator)?
    } else {
        let lhs = Datum::try_new(left)?;
        let rhs = Datum::try_new_with_target_datatype(right, lhs.data_type())?;

        match operator {
            CompareOperator::Eq => cmp::eq(&lhs, &rhs)?,
            CompareOperator::NotEq => cmp::neq(&lhs, &rhs)?,
            CompareOperator::Gt => cmp::gt(&lhs, &rhs)?,
            CompareOperator::Gte => cmp::gt_eq(&lhs, &rhs)?,
            CompareOperator::Lt => cmp::lt(&lhs, &rhs)?,
            CompareOperator::Lte => cmp::lt_eq(&lhs, &rhs)?,
        }
    };

    from_arrow_array_with_len(&arrow_array, left.len(), nullable)
}

/// Compare two Arrow arrays element-wise using [`make_comparator`].
fn compare_nested_arrow_arrays(
    lhs: &dyn arrow_array::Array,
    rhs: &dyn arrow_array::Array,
    operator: CompareOperator,
) -> VortexResult<BooleanArray> {
    let compare_arrays_at = make_comparator(lhs, rhs, SortOptions::default())?;

    let cmp_fn = match operator {
        CompareOperator::Eq => Ordering::is_eq,
        CompareOperator::NotEq => Ordering::is_ne,
        CompareOperator::Gt => Ordering::is_gt,
        CompareOperator::Gte => Ordering::is_ge,
        CompareOperator::Lt => Ordering::is_lt,
        CompareOperator::Lte => Ordering::is_le,
    };

    let values = (0..lhs.len())
        .map(|i| cmp_fn(compare_arrays_at(i, i)))
        .collect();
    let nulls = NullBuffer::union(lhs.nulls(), rhs.nulls());

    Ok(BooleanArray::new(values, nulls))
}

fn arrow_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    operator: NumericOperator,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let len = lhs.len();

    let left = Datum::try_new(lhs)?;
    let right = Datum::try_new_with_target_datatype(rhs, left.data_type())?;

    let array = match operator {
        NumericOperator::Add => arrow_arith::numeric::add(&left, &right)?,
        NumericOperator::Sub => arrow_arith::numeric::sub(&left, &right)?,
        NumericOperator::Mul => arrow_arith::numeric::mul(&left, &right)?,
        NumericOperator::Div => arrow_arith::numeric::div(&left, &right)?,
    };

    from_arrow_array_with_len(array.as_ref(), len, nullable)
}

fn arrow_boolean(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: Operator,
) -> VortexResult<ArrayRef> {
    use arrow_array::cast::AsArray;

    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    let lhs = lhs.clone().into_arrow(&DataType::Boolean)?.as_boolean().clone();
    let rhs = rhs.clone().into_arrow(&DataType::Boolean)?.as_boolean().clone();

    let array = match op {
        Operator::And => arrow_arith::boolean::and_kleene(&lhs, &rhs)?,
        Operator::Or => arrow_arith::boolean::or_kleene(&lhs, &rhs)?,
        other => return Err(vortex_err!("Not a boolean operator: {other}")),
    };

    ArrayRef::from_arrow(&array, nullable)
}

fn arrow_like(
    array: &ArrayRef,
    pattern: &ArrayRef,
    options: LikeOptions,
) -> VortexResult<ArrayRef> {
    let nullable = array.dtype().is_nullable() | pattern.dtype().is_nullable();
    let len = array.len();

    let lhs = Datum::try_new(array)?;
    let rhs = Datum::try_new_with_target_datatype(pattern, lhs.data_type())?;

    let result = match (options.negated, options.case_insensitive) {
        (false, false) => arrow_string::like::like(&lhs, &rhs)?,
        (true, false) => arrow_string::like::nlike(&lhs, &rhs)?,
        (false, true) => arrow_string::like::ilike(&lhs, &rhs)?,
        (true, true) => arrow_string::like::nilike(&lhs, &rhs)?,
    };

    from_arrow_array_with_len(&result, len, nullable)
}

fn arrow_zip(
    condition: &ArrayRef,
    lhs: &ArrayRef,
    rhs: &ArrayRef,
) -> VortexResult<ArrayRef> {
    use arrow_array::cast::AsArray;

    let cond_arrow = condition.clone().into_arrow(&DataType::Boolean)?;
    let cond_arrow = cond_arrow.as_boolean();
    let lhs_arrow = lhs.clone().into_arrow_preferred()?;
    let rhs_arrow = rhs.clone().into_arrow(lhs_arrow.data_type())?;

    let zipped = arrow_select::zip::zip(cond_arrow, &lhs_arrow, &rhs_arrow)?;
    ArrayRef::from_arrow(zipped.as_ref(), lhs.dtype().is_nullable() || rhs.dtype().is_nullable())
}

fn arrow_filter_varbinview(
    array: &VarBinViewArray,
    mask: &Arc<MaskValues>,
) -> VortexResult<VarBinViewArray> {
    let array_ref = array.clone().into_array().into_arrow_preferred()?;
    let mask_array =
        BooleanArray::new(mask.bit_buffer().clone().into_arrow_boolean_buffer(), None);
    let filtered = arrow_select::filter::filter(array_ref.as_ref(), &mask_array)?;
    use vortex_array::array::TypedArrayRef;
    use vortex_array::arrays::VarBinView;

    let vortex_array =
        ArrayRef::from_arrow(filtered.as_ref(), array.as_ref().dtype().is_nullable())?;
    Ok(vortex_array.as_::<VarBinView>().into_owned())
}

fn arrow_varbin_compare_with_const(
    lhs: &ArrayRef,
    rhs_const: &Scalar,
    op: CompareOperator,
) -> VortexResult<ArrayRef> {
    use arrow_array::BinaryArray;
    use arrow_array::StringArray;

    let nullable = lhs.dtype().is_nullable() || rhs_const.dtype().is_nullable();
    let len = lhs.len();

    let lhs_datum = Datum::try_new(lhs)?;

    let array = match rhs_const.dtype() {
        DType::Utf8(_) => {
            let rhs_datum = rhs_const
                .as_utf8()
                .value()
                .map(StringArray::new_scalar)
                .unwrap_or_else(|| arrow_array::Scalar::new(StringArray::new_null(1)));
            apply_cmp_op(&lhs_datum, &rhs_datum, op)?
        }
        DType::Binary(_) => {
            let rhs_datum = rhs_const
                .as_binary()
                .value()
                .map(BinaryArray::new_scalar)
                .unwrap_or_else(|| arrow_array::Scalar::new(BinaryArray::new_null(1)));
            apply_cmp_op(&lhs_datum, &rhs_datum, op)?
        }
        _ => {
            return Err(vortex_err!(
                "Unsupported dtype for varbin compare: {:?}",
                rhs_const.dtype()
            ));
        }
    };

    from_arrow_array_with_len(&array, len, nullable)
}

fn apply_cmp_op(
    lhs: &dyn arrow_array::Datum,
    rhs: &dyn arrow_array::Datum,
    op: CompareOperator,
) -> VortexResult<BooleanArray> {
    Ok(match op {
        CompareOperator::Eq => cmp::eq(lhs, rhs)?,
        CompareOperator::NotEq => cmp::neq(lhs, rhs)?,
        CompareOperator::Gt => cmp::gt(lhs, rhs)?,
        CompareOperator::Gte => cmp::gt_eq(lhs, rhs)?,
        CompareOperator::Lt => cmp::lt(lhs, rhs)?,
        CompareOperator::Lte => cmp::lt_eq(lhs, rhs)?,
    })
}

inventory::submit! {
    ArrowComputeRegistration(ArrowCompute {
        compare: arrow_compare,
        numeric: arrow_numeric,
        boolean: arrow_boolean,
        like: arrow_like,
        zip: arrow_zip,
        filter_varbinview: arrow_filter_varbinview,
        varbin_compare_with_const: arrow_varbin_compare_with_const,
    })
}
