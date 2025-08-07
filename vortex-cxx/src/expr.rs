// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_expr::*;

#[cxx::bridge(namespace = "vortex::ffi")]
mod ffi {
    extern "Rust" {
        type Expr;
        // fn literal(value: String, dtype: String) -> Result<Box<Expr>>;
        fn root() -> Box<Expr>;
        fn column(name: String) -> Box<Expr>;
        fn get_item(field: String, child: Box<Expr>) -> Result<Box<Expr>>;
        fn not_(child: Box<Expr>) -> Result<Box<Expr>>;
        fn is_null(child: Box<Expr>) -> Result<Box<Expr>>;
        // binary op
        fn eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Result<Box<Expr>>;
        fn not_eq_(lhs: Box<Expr>, rhs: Box<Expr>) -> Result<Box<Expr>>;
        fn gt(lhs: Box<Expr>, rhs: Box<Expr>) -> Result<Box<Expr>>;
        fn gt_eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Result<Box<Expr>>;
        fn lt(lhs: Box<Expr>, rhs: Box<Expr>) -> Result<Box<Expr>>;
        fn lt_eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Result<Box<Expr>>;
        fn and_(lhs: Box<Expr>, rhs: Box<Expr>) -> Result<Box<Expr>>;
        fn or_(lhs: Box<Expr>, rhs: Box<Expr>) -> Result<Box<Expr>>;
        fn checked_add(lhs: Box<Expr>, rhs: Box<Expr>) -> Result<Box<Expr>>;
    }
}

struct Expr {
    inner: ExprRef,
}

// fn literal(
//     value: String,
//     dtype: String,
// ) -> Result<Box<Expr>, Box<dyn std::error::Error + Send + Sync>> {
//     let dtype: DType = serde_json::from_str(&dtype)?;
//     let scalar: Scalar = serde_json::from_str(&value)?;
//     Ok(Box::new(Expr {
//         inner: LiteralExpr::new(scalar).into_expr(),
//     }))
// }

fn root() -> Box<Expr> {
    Box::new(Expr {
        inner: vortex_expr::root(),
    })
}

fn column(name: String) -> Box<Expr> {
    Box::new(Expr {
        inner: vortex_expr::get_item(name, vortex_expr::root()),
    })
}

fn get_item(
    field: String,
    child: Box<Expr>,
) -> Result<Box<Expr>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(Box::new(Expr {
        inner: vortex_expr::get_item(field, child.inner),
    }))
}

fn not_(child: Box<Expr>) -> Result<Box<Expr>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(Box::new(Expr {
        inner: NotExpr::new(child.inner).into_expr(),
    }))
}

fn is_null(child: Box<Expr>) -> Result<Box<Expr>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(Box::new(Expr {
        inner: IsNullExpr::new(child.inner).into_expr(),
    }))
}

macro_rules! binary_op {
    ($fn_name:ident $(, $suffix:tt)?) => {
        paste::paste! {
            fn [<$fn_name $($suffix)?>](
                lhs: Box<Expr>,
                rhs: Box<Expr>,
            ) -> Result<Box<Expr>, Box<dyn std::error::Error + Send + Sync>> {
                Ok(Box::new(Expr {
                    inner: vortex_expr::$fn_name(lhs.inner, rhs.inner),
                }))
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
