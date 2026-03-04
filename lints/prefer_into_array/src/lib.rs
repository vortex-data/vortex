#![feature(rustc_private)]

extern crate rustc_hir;
extern crate rustc_middle;

use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::TyKind;

dylint_linting::declare_late_lint! {
    /// ### What it does
    /// Detects calls to `.to_array()` on owned values that implement `IntoArray`,
    /// where `.into_array()` would avoid an unnecessary clone.
    ///
    /// ### Why is this bad?
    /// When you have an owned concrete array (e.g. `ConstantArray::new(42, 10)`),
    /// calling `.to_array()` auto-refs via `Deref` to `&dyn DynArray` and internally
    /// clones. Using `.into_array()` consumes self directly with no clone.
    ///
    /// ### Example
    /// ```rust,ignore
    /// // Bad: clones internally via auto-ref + Deref
    /// let arr: ArrayRef = ConstantArray::new(42, 10).to_array();
    ///
    /// // Good: consumes self, no clone
    /// let arr: ArrayRef = ConstantArray::new(42, 10).into_array();
    /// ```
    pub PREFER_INTO_ARRAY,
    Warn,
    "prefer `.into_array()` over `.to_array()` on owned values to avoid unnecessary cloning"
}

impl<'tcx> LateLintPass<'tcx> for PreferIntoArray {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // Match method calls named `to_array` with no arguments (just the receiver).
        if let ExprKind::MethodCall(method, receiver, args, _span) = &expr.kind {
            if method.ident.as_str() != "to_array" || !args.is_empty() {
                return;
            }

            // Get the type of the receiver *before* any autoref/autoderef adjustments.
            let recv_ty = cx.typeck_results().expr_ty(receiver);

            // If the receiver is already a reference, `.to_array()` is fine — the caller
            // doesn't own the value so `.into_array()` isn't available.
            if matches!(recv_ty.kind(), TyKind::Ref(..)) {
                return;
            }

            cx.lint(PREFER_INTO_ARRAY, |diag| {
                diag.primary_message(
                    "calling `.to_array()` on an owned value clones unnecessarily",
                );
                diag.span(expr.span);
                diag.help(
                    "use `.into_array()` instead, which consumes the owned value without cloning",
                );
            });
        }
    }
}

#[test]
fn ui() {
    dylint_testing::ui_test(env!("CARGO_PKG_NAME"), &std::path::Path::new("ui"));
}
