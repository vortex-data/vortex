// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use anyhow::Result;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::expr::BoundExpr;
use vortex::expr::lit;
use vortex::expr::root as bound_root;
use vortex::scalar::Scalar as VortexScalar;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::is_null::IsNull;
use vortex::scalar_fn::fns::not::Not;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::select::FieldSelection;
use vortex::scalar_fn::fns::select::Select;

use crate::scalar::Scalar;

#[derive(Clone)]
enum ExprKind {
    Root,
    Literal(VortexScalar),
    GetItem {
        field: FieldName,
        child: Box<ExprKind>,
    },
    Not(Box<ExprKind>),
    IsNull(Box<ExprKind>),
    Binary {
        operator: Operator,
        lhs: Box<ExprKind>,
        rhs: Box<ExprKind>,
    },
    Select {
        fields: FieldNames,
        child: Box<ExprKind>,
    },
}

pub(crate) struct Expr {
    inner: ExprKind,
}

impl Expr {
    pub(crate) fn bind(&self, scope: &DType) -> Result<BoundExpr> {
        self.inner.bind(scope)
    }
}

impl ExprKind {
    fn bind(&self, scope: &DType) -> Result<BoundExpr> {
        Ok(match self {
            Self::Root => bound_root(scope.clone()),
            Self::Literal(scalar) => lit(scalar.clone()),
            Self::GetItem { field, child } => {
                GetItem.try_new_expr(field.clone(), [child.bind(scope)?])?
            }
            Self::Not(child) => Not.try_new_expr(EmptyOptions, [child.bind(scope)?])?,
            Self::IsNull(child) => IsNull.try_new_expr(EmptyOptions, [child.bind(scope)?])?,
            Self::Binary { operator, lhs, rhs } => {
                Binary.try_new_expr(*operator, [lhs.bind(scope)?, rhs.bind(scope)?])?
            }
            Self::Select { fields, child } => Select.try_new_expr(
                FieldSelection::Include(fields.clone()),
                [child.bind(scope)?],
            )?,
        })
    }
}

pub(crate) fn literal(scalar: Box<Scalar>) -> Box<Expr> {
    Box::new(Expr {
        inner: ExprKind::Literal(scalar.inner),
    })
}

pub(crate) fn root() -> Box<Expr> {
    Box::new(Expr {
        inner: ExprKind::Root,
    })
}

pub(crate) fn column(name: String) -> Box<Expr> {
    Box::new(Expr {
        inner: ExprKind::GetItem {
            field: name.into(),
            child: Box::new(ExprKind::Root),
        },
    })
}

pub(crate) fn get_item(field: String, child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: ExprKind::GetItem {
            field: field.into(),
            child: Box::new(child.inner),
        },
    })
}

pub(crate) fn not_(child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: ExprKind::Not(Box::new(child.inner)),
    })
}

pub(crate) fn is_null(child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: ExprKind::IsNull(Box::new(child.inner)),
    })
}

macro_rules! binary_op {
    ($fn_name:ident, $operator:expr $(, $suffix:tt)?) => {
        paste::paste! {
            pub(crate) fn [<$fn_name $($suffix)?>](
                lhs: Box<Expr>,
                rhs: Box<Expr>,
            ) -> Box<Expr> {
                Box::new(Expr {
                    inner: ExprKind::Binary {
                        operator: $operator,
                        lhs: Box::new(lhs.inner),
                        rhs: Box::new(rhs.inner),
                    },
                })
            }
        }
    };
}

binary_op!(eq, Operator::Eq);
binary_op!(not_eq, Operator::NotEq, _);
binary_op!(gt, Operator::Gt);
binary_op!(gt_eq, Operator::Gte);
binary_op!(lt, Operator::Lt);
binary_op!(lt_eq, Operator::Lte);
binary_op!(and, Operator::And, _);
binary_op!(or, Operator::Or, _);
binary_op!(checked_add, Operator::Add);

pub(crate) fn select(fields: Vec<String>, child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: ExprKind::Select {
            fields: fields
                .into_iter()
                .map(FieldName::from)
                .collect::<Vec<_>>()
                .into(),
            child: Box::new(child.inner),
        },
    })
}
