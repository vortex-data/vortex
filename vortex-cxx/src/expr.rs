// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::dtype::FieldName;
use vortex::expr::ExprRef;

use crate::scalar::Scalar;

pub(crate) struct Expr {
    pub(crate) inner: ExprRef,
}

pub(crate) fn literal(scalar: Box<Scalar>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::lit(scalar.inner),
    })
}

pub(crate) fn root() -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::root(),
    })
}

pub(crate) fn column(name: String) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::get_item(name, vortex::expr::root()),
    })
}

pub(crate) fn get_item(field: String, child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::get_item(field, child.inner),
    })
}

pub(crate) fn not_(child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::not(child.inner),
    })
}

pub(crate) fn is_null(child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::is_null(child.inner),
    })
}

macro_rules! binary_op {
    ($fn_name:ident $(, $suffix:tt)?) => {
        paste::paste! {
            pub(crate) fn [<$fn_name $($suffix)?>](
                lhs: Box<Expr>,
                rhs: Box<Expr>,
            ) -> Box<Expr> {
                Box::new(Expr {
                    inner: vortex::expr::$fn_name(lhs.inner, rhs.inner),
                })
            }
        }
    };
}

binary_op!(eq);
binary_op!(not_eq, _);
binary_op!(gt);
binary_op!(gt_eq);
binary_op!(lt);
binary_op!(lt_eq);
binary_op!(and, _);
binary_op!(or, _);
binary_op!(checked_add);

pub(crate) fn select(fields: Vec<String>, child: Box<Expr>) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex::expr::select(
            fields.into_iter().map(FieldName::from).collect::<Vec<_>>(),
            child.inner,
        ),
    })
}
