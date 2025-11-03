// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::{Debug, Formatter};
use vortex_array::compute::{between as between_compute, BetweenOptions, StrictComparison};
use vortex_array::{ArrayRef, ArraySessionExt, DeserializeMetadata, ProstMetadata};
use vortex_dtype::DType;
use vortex_dtype::DType::Bool;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_proto::expr as pb;

use crate::display::{DisplayAs, DisplayFormat};
use crate::metadata::{EmptyMetadata, ExprMetadata};
use crate::v2::{Expression, ExpressionView};
use crate::{
    vtable, AnalysisExpr, BinaryExpr, ExprEncodingRef, ExprId, ExprRef, IntoExpr, StatsCatalog,
    VTable,
};

vtable!(Between);

pub struct BetweenExpr;
pub struct BetweenExprEncoding;

impl VTable for BetweenVTable {
    type Expr = ();
    type Encoding = BetweenExprEncoding;
    type Metadata = ProstMetadata<pb::BetweenOpts>;
    type Metadata2 = BetweenOptions;

    fn id(_encoding: &Self::Encoding) -> ExprId {
        ExprId::new_ref("between")
    }

    fn validate(expr: ExpressionView<Self>) -> VortexResult<()> {
        if expr.children().len() != 3 {
            vortex_bail!(
                "Between expression requires exactly 3 children, got {}",
                expr.children().len()
            );
        }
        Ok(())
    }

    fn deserialize_metadata(
        encoding: &Self::Encoding,
        metadata: &[u8],
    ) -> VortexResult<Self::Metadata2> {
        let meta = ProstMetadata::<pb::BetweenOpts>::deserialize(metadata)?;
        Ok(BetweenOptions {
            lower_strict: if meta.lower_strict {
                StrictComparison::Strict
            } else {
                StrictComparison::NonStrict
            },
            upper_strict: if meta.upper_strict {
                StrictComparison::Strict
            } else {
                StrictComparison::NonStrict
            },
        })
    }

    fn return_dtype2(expr: ExpressionView<Self>, scope: &DType) -> VortexResult<DType> {
        let arr_dt = expr.child().return_dtype(scope)?;
        let lower_dt = expr.lower().return_dtype(scope)?;
        let upper_dt = expr.upper().return_dtype(scope)?;

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

    fn evaluate2(expr: ExpressionView<Self>, scope: &ArrayRef) -> VortexResult<ArrayRef> {
        let arr = expr.child().evaluate(scope)?;
        let lower = expr.lower().evaluate(scope)?;
        let upper = expr.upper().evaluate(scope)?;
        between_compute(&arr, &lower, &upper, expr.metadata())
    }
}

impl DisplayAs for BetweenOptions {
    fn fmt_as(&self, df: DisplayFormat, f: &mut Formatter) -> std::fmt::Result {
        match df {
            DisplayFormat::Compact | DisplayFormat::Tree => {
                write!(
                    f,
                    "BetweenOptions(lower_strict: {:?}, upper_strict: {:?})",
                    self.lower_strict, self.upper_strict
                )
            }
        }
    }

    fn child_names(&self) -> Option<Vec<String>> {
        Some(vec![
            "array".to_string(),
            "lower".to_string(),
            "upper".to_string(),
        ])
    }
}

impl ExprMetadata for BetweenOptions {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl ExpressionView<'_, BetweenVTable> {
    pub fn child(&self) -> &Expression {
        &self.children()[0]
    }

    pub fn lower(&self) -> &Expression {
        &self.children()[1]
    }

    pub fn upper(&self) -> &Expression {
        &self.children()[2]
    }

    pub fn to_binary_expr(&self) -> Expression {
        let options = self.metadata();
        let arr = self.children()[0].clone();
        let lower = self.children()[1].clone();
        let upper = self.children()[2].clone();

        let lhs = BinaryExpr::new(
            lower,
            options.lower_strict.to_operator().into(),
            arr.clone(),
        );
        let rhs = BinaryExpr::new(arr, options.upper_strict.to_operator().into(), upper);
        BinaryExpr::new(lhs.into_expr(), crate::Operator::And, rhs.into_expr()).into_expr()
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
pub fn between(
    arr: Expression,
    lower: Expression,
    upper: Expression,
    options: BetweenOptions,
) -> Expression {
    static BETWEEN: ExprEncodingRef = ExprEncodingRef::new_ref(BetweenExprEncoding.as_static_ref());

    Expression::try_new(
        BETWEEN.clone(),
        EmptyMetadata::new(),
        vec![arr.clone(), lower.clone(), upper.clone()].into(),
    )
    .vortex_expect("Failed to create Between expression")
}

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
