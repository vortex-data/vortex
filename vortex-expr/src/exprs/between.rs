// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Display};

use vortex_array::compute::{BetweenOptions, StrictComparison, between as between_compute};
use vortex_array::{ArrayRef, DeserializeMetadata, ProstMetadata};
use vortex_dtype::DType;
use vortex_dtype::DType::Bool;
use vortex_error::{VortexResult, vortex_bail};
use vortex_proto::expr as pb;

use crate::{
    AnalysisExpr, BinaryExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, VTable, vtable,
};

vtable!(Between);

#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Debug, Hash, Eq)]
pub struct BetweenExpr {
    arr: ExprRef,
    lower: ExprRef,
    upper: ExprRef,
    options: BetweenOptions,
}

impl PartialEq for BetweenExpr {
    fn eq(&self, other: &Self) -> bool {
        self.arr.eq(&other.arr)
            && self.lower.eq(&other.lower)
            && self.upper.eq(&other.upper)
            && self.options == other.options
    }
}

pub struct BetweenExprEncoding;

impl VTable for BetweenVTable {
    type Expr = BetweenExpr;
    type Encoding = BetweenExprEncoding;
    type Metadata = ProstMetadata<pb::BetweenOpts>;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("between")
    }

    fn encoding(_expr: &Self::Expr) -> ExprEncodingRef {
        ExprEncodingRef::new_ref(BetweenExprEncoding.as_ref())
    }

    fn metadata(expr: &Self::Expr) -> Option<Self::Metadata> {
        Some(ProstMetadata(pb::BetweenOpts {
            lower_strict: expr.options.lower_strict == StrictComparison::Strict,
            upper_strict: expr.options.upper_strict == StrictComparison::Strict,
        }))
    }

    fn children(expr: &Self::Expr) -> Vec<&ExprRef> {
        vec![&expr.arr, &expr.lower, &expr.upper]
    }

    fn with_children(expr: &Self::Expr, children: Vec<ExprRef>) -> VortexResult<Self::Expr> {
        Ok(BetweenExpr::new(
            children[0].clone(),
            children[1].clone(),
            children[2].clone(),
            expr.options.clone(),
        ))
    }

    fn build(
        _encoding: &Self::Encoding,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        children: Vec<ExprRef>,
    ) -> VortexResult<Self::Expr> {
        Ok(BetweenExpr::new(
            children[0].clone(),
            children[1].clone(),
            children[2].clone(),
            BetweenOptions {
                lower_strict: if metadata.lower_strict {
                    StrictComparison::Strict
                } else {
                    StrictComparison::NonStrict
                },
                upper_strict: if metadata.upper_strict {
                    StrictComparison::Strict
                } else {
                    StrictComparison::NonStrict
                },
            },
        ))
    }

    fn evaluate(expr: &Self::Expr, scope: &Scope) -> VortexResult<ArrayRef> {
        let arr_val = expr.arr.unchecked_evaluate(scope)?;
        let lower_arr_val = expr.lower.unchecked_evaluate(scope)?;
        let upper_arr_val = expr.upper.unchecked_evaluate(scope)?;

        between_compute(&arr_val, &lower_arr_val, &upper_arr_val, &expr.options)
    }

    fn return_dtype(expr: &Self::Expr, scope: &DType) -> VortexResult<DType> {
        let arr_dt = expr.arr.return_dtype(scope)?;
        let lower_dt = expr.lower.return_dtype(scope)?;
        let upper_dt = expr.upper.return_dtype(scope)?;

        if !arr_dt.eq_ignore_nullability(&lower_dt) {
            vortex_bail!(
                "Array dtype {} does not match lower dtype {}",
                arr_dt,
                lower_dt
            );
        }
        if !arr_dt.eq_ignore_nullability(&upper_dt) {
            vortex_bail!(
                "Array dtype {} does not match upper dtype {}",
                arr_dt,
                upper_dt
            );
        }

        Ok(Bool(
            arr_dt.nullability() | lower_dt.nullability() | upper_dt.nullability(),
        ))
    }
}

impl BetweenExpr {
    pub fn new(arr: ExprRef, lower: ExprRef, upper: ExprRef, options: BetweenOptions) -> Self {
        Self {
            arr,
            lower,
            upper,
            options,
        }
    }

    pub fn new_expr(
        arr: ExprRef,
        lower: ExprRef,
        upper: ExprRef,
        options: BetweenOptions,
    ) -> ExprRef {
        Self::new(arr, lower, upper, options).into_expr()
    }

    pub fn to_binary_expr(&self) -> ExprRef {
        let lhs = BinaryExpr::new(
            self.lower.clone(),
            self.options.lower_strict.to_operator().into(),
            self.arr.clone(),
        );
        let rhs = BinaryExpr::new(
            self.arr.clone(),
            self.options.upper_strict.to_operator().into(),
            self.upper.clone(),
        );
        BinaryExpr::new(lhs.into_expr(), crate::Operator::And, rhs.into_expr()).into_expr()
    }
}

impl Display for BetweenExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({} {} {} {} {})",
            self.lower,
            self.options.lower_strict.to_operator(),
            self.arr,
            self.options.upper_strict.to_operator(),
            self.upper
        )
    }
}

impl AnalysisExpr for BetweenExpr {}

/// Creates an expression that checks if values are between two bounds.
///
/// Returns a boolean array indicating which values fall within the specified range.
/// The comparison strictness is controlled by the options parameter.
///
/// ```rust
/// # use vortex_array::compute::BetweenOptions;
/// # use vortex_array::compute::StrictComparison;
/// # use vortex_expr::{between, lit, root};
/// let opts = BetweenOptions {
///     lower_strict: StrictComparison::NonStrict,
///     upper_strict: StrictComparison::NonStrict,
/// };
/// let expr = between(root(), lit(10), lit(20), opts);
/// ```
pub fn between(arr: ExprRef, lower: ExprRef, upper: ExprRef, options: BetweenOptions) -> ExprRef {
    BetweenExpr::new(arr, lower, upper, options).into_expr()
}
