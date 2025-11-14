// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::reducer::apply_child_rules;

/// Unlike `simplify`, this function simplifies an expression under the assumption that scope is
/// a known DType. Simplification is applied first and then additional dtype-aware rules.
///
/// NOTE: After typed simplification, returned expressions is "bound" to the scope DType.
///     Applying the returned expression to a different DType may produce wrong results.
pub fn simplify_typed(e: Expression, ctx: &DType) -> VortexResult<Expression> {
    // Apply all registered rules (PackGetItemRule, RemoveSelectRule, RemoveMergeRule)
    let session = ExprSession::default();
    let e = apply_child_rules(e, ctx, &session)?;

    Ok(e)
}
