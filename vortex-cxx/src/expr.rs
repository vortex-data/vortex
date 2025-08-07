// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_expr::{ExprRef, GetItemExpr, IntoExpr};

#[cxx::bridge(namespace = "vortex::ffi")]
mod ffi {
    extern "Rust" {
        type Expr;
        fn get_item(field: String, child: Box<Expr>) -> Result<Box<Expr>>;
    }
}

struct Expr {
    inner: ExprRef,
}

fn get_item(
    field: String,
    child: Box<Expr>,
) -> Result<Box<Expr>, Box<dyn std::error::Error + Send + Sync>> {
    Ok(Box::new(Expr {
        inner: GetItemExpr::new(field, child.inner).into_expr(),
    }))
}
