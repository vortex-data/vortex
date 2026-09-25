// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;

use arbitrary::Result as AResult;
use arbitrary::Unstructured;

use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::expr::BoundExpression;
use crate::expr::bound;
use crate::scalar::arbitrary::random_scalar;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;

pub fn projection_expr(
    u: &mut Unstructured<'_>,
    dtype: &DType,
) -> AResult<Option<BoundExpression>> {
    let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
        return Ok(None);
    };

    let column_count = u.int_in_range::<usize>(0..=max(struct_dtype.nfields(), 10))?;

    let cols = (0..column_count)
        .map(|_| {
            let get_item = u.choose_iter(struct_dtype.names().iter())?;
            Ok((
                get_item.clone(),
                bound::col(get_item.clone(), dtype.clone()),
            ))
        })
        .collect::<AResult<Vec<_>>>()?;

    Ok(Some(bound::pack(cols, u.arbitrary()?)))
}

pub fn filter_expr(u: &mut Unstructured<'_>, dtype: &DType) -> AResult<Option<BoundExpression>> {
    let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
        return Ok(None);
    };

    let filter_count = u.int_in_range::<usize>(0..=max(struct_dtype.nfields(), 10))?;

    let filters = (0..filter_count)
        .map(|_| {
            let (col, field_dtype) =
                u.choose_iter(struct_dtype.names().iter().zip(struct_dtype.fields()))?;
            random_comparison(u, col, &field_dtype, dtype)
        })
        .collect::<AResult<Vec<_>>>()?;

    Ok(bound::and_collect(filters))
}

fn random_comparison(
    u: &mut Unstructured<'_>,
    name: &FieldName,
    field_dtype: &DType,
    scope: &DType,
) -> AResult<BoundExpression> {
    let scalar = random_scalar(u, field_dtype)?;
    Binary
        .try_new_bound_expr(
            arbitrary_comparison_operator(u)?,
            [bound::col(name.clone(), scope.clone()), bound::lit(scalar)],
        )
        .map_err(|_| arbitrary::Error::IncorrectFormat)
}

fn arbitrary_comparison_operator(u: &mut Unstructured<'_>) -> AResult<Operator> {
    Ok(match u.int_in_range(0..=5)? {
        0 => Operator::Eq,
        1 => Operator::NotEq,
        2 => Operator::Gt,
        3 => Operator::Gte,
        4 => Operator::Lt,
        5 => Operator::Lte,
        _ => unreachable!("range 0..=5"),
    })
}
