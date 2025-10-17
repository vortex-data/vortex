// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use futures::try_join;
use itertools::Itertools;
use vortex_array::compute::{BetweenOptions, StrictComparison, between as between_compute};
use vortex_array::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, LengthBounds, Operator,
    OperatorEq, OperatorHash, OperatorId, OperatorRef,
};
use vortex_array::{Array, ArrayRef, Canonical, DeserializeMetadata, IntoArray, ProstMetadata};
use vortex_dtype::DType;
use vortex_dtype::DType::Bool;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_proto::expr as pb;

use crate::display::{DisplayAs, DisplayFormat};
use crate::{
    AnalysisExpr, BinaryExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, Scope, StatsCatalog,
    VTable, vtable,
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

    fn operator(expr: &Self::Expr, scope: &OperatorRef) -> VortexResult<Option<OperatorRef>> {
        let Some(arr) = expr.arr.operator(scope)? else {
            return Ok(None);
        };
        let Some(lower) = expr.lower.operator(scope)? else {
            return Ok(None);
        };
        let Some(upper) = expr.upper.operator(scope)? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(BetweenOperator {
            children: [arr, lower, upper],
            dtype: expr.return_dtype(scope.dtype())?,
            options: expr.options.clone(),
        })))
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

impl DisplayAs for BetweenExpr {
    fn fmt_as(&self, df: DisplayFormat, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact => {
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
            DisplayFormat::Tree => {
                write!(f, "Between")
            }
        }
    }

    fn child_names(&self) -> Option<Vec<String>> {
        // Children are: arr, lower, upper (based on the order in the children() method)
        Some(vec![
            "array".to_string(),
            format!("lower ({:?})", self.options.lower_strict),
            format!("upper ({:?})", self.options.upper_strict),
        ])
    }
}

impl AnalysisExpr for BetweenExpr {
    fn stat_falsification(&self, catalog: &mut dyn StatsCatalog) -> Option<ExprRef> {
        self.to_binary_expr().stat_falsification(catalog)
    }
}

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

#[derive(Debug)]
pub struct BetweenOperator {
    children: [OperatorRef; 3],
    dtype: DType,
    options: BetweenOptions,
}

impl OperatorHash for BetweenOperator {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        for child in &self.children {
            child.operator_hash(state);
        }
        self.dtype.hash(state);
        self.options.hash(state);
    }
}

impl OperatorEq for BetweenOperator {
    fn operator_eq(&self, other: &Self) -> bool {
        self.children.len() == other.children.len()
            && self
                .children
                .iter()
                .zip(other.children.iter())
                .all(|(a, b)| a.operator_eq(b))
            && self.dtype == other.dtype
            && self.options == other.options
    }
}

impl Operator for BetweenOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.between")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn bounds(&self) -> LengthBounds {
        self.children[0].bounds()
    }

    fn children(&self) -> &[OperatorRef] {
        &self.children
    }

    fn with_children(self: Arc<Self>, children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        let (arr, lower, upper) = children
            .into_iter()
            .tuples()
            .next()
            .vortex_expect("expected 3 children");

        Ok(Arc::new(BetweenOperator {
            children: [arr, lower, upper],
            dtype: self.dtype.clone(),
            options: self.options.clone(),
        }))
    }

    fn is_selection_target(&self, _child_idx: usize) -> Option<bool> {
        // All children are position preserving.
        Some(true)
    }
}

impl BatchOperator for BetweenOperator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        let arr = ctx.child(0)?;
        let lower = ctx.child(1)?;
        let upper = ctx.child(2)?;
        Ok(Box::new(BetweenExecution {
            arr,
            lower,
            upper,
            options: self.options.clone(),
        }))
    }
}

struct BetweenExecution {
    arr: BatchExecutionRef,
    lower: BatchExecutionRef,
    upper: BatchExecutionRef,
    options: BetweenOptions,
}

#[async_trait]
impl BatchExecution for BetweenExecution {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        let (arr, lower, upper) = try_join!(
            self.arr.execute(),
            self.lower.execute(),
            self.upper.execute()
        )?;
        let result = between_compute(
            arr.into_array().as_ref(),
            lower.into_array().as_ref(),
            upper.into_array().as_ref(),
            &self.options,
        )?;
        Ok(result.to_canonical())
    }
}

// TODO(ngates): we need scalar variants for batch execution. Although really it should be
//  pipelined?

#[cfg(test)]
mod tests {
    use vortex_array::compute::{BetweenOptions, StrictComparison};

    use crate::{between, get_item, lit, root};

    #[test]
    fn test_display() {
        let expr = between(
            get_item("score", root()),
            lit(10),
            lit(50),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::Strict,
            },
        );
        assert_eq!(expr.to_string(), "(10i32 <= $.score < 50i32)");

        let expr2 = between(
            root(),
            lit(0),
            lit(100),
            BetweenOptions {
                lower_strict: StrictComparison::Strict,
                upper_strict: StrictComparison::NonStrict,
            },
        );
        assert_eq!(expr2.to_string(), "(0i32 < $ <= 100i32)");
    }
}
