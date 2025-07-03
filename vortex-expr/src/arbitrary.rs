// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;

use arbitrary::{Result as AResult, Unstructured};
use vortex_dtype::{DType, FieldName};
use vortex_scalar::arbitrary::random_scalar;

use crate::{BinaryExpr, ExprRef, Operator, and_collect, get_item_scope, lit, pack};

pub fn projection_expr(u: &mut Unstructured<'_>, dtype: &DType) -> AResult<Option<ExprRef>> {
    let Some(struct_dtype) = dtype.as_struct() else {
        return Ok(None);
    };

    let column_count = u.int_in_range::<usize>(0..=max(struct_dtype.nfields(), 10))?;

    let cols = (0..column_count)
        .map(|_| {
            let get_item = u.choose_iter(struct_dtype.names().iter())?;
            Ok((get_item.clone(), get_item_scope(get_item.clone())))
        })
        .collect::<AResult<Vec<_>>>()?;

    Ok(Some(pack(cols, u.arbitrary()?)))
}

pub fn filter_expr(u: &mut Unstructured<'_>, dtype: &DType) -> AResult<Option<ExprRef>> {
    let Some(struct_dtype) = dtype.as_struct() else {
        return Ok(None);
    };

    let filter_count = u.int_in_range::<usize>(0..=max(struct_dtype.nfields(), 10))?;

    let filters = (0..filter_count)
        .map(|_| {
            let (col, dtype) =
                u.choose_iter(struct_dtype.names().iter().zip(struct_dtype.fields()))?;
            random_comparison(u, col, &dtype)
        })
        .collect::<AResult<Vec<_>>>()?;

    Ok(and_collect(filters))
}

fn random_comparison(u: &mut Unstructured<'_>, col: &FieldName, dtype: &DType) -> AResult<ExprRef> {
    let scalar = random_scalar(u, dtype)?;
    Ok(BinaryExpr::new_expr(
        get_item_scope(col.clone()),
        arbitrary_comparison_operator(u)?,
        lit(scalar),
    ))
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
