// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::transform::remove_merge::remove_merge;
use crate::expr::transform::remove_select::remove_select;
use crate::expr::transform::simplify::simplify;

/// Unlike `simplify`, this function simplifies an expression under the assumption that scope is
/// a known DType. Simplified is applied first and then additional rules.
///
/// NOTE: After typed simplification, returned expressions is "bound" to the scope DType.
///     Applying the returned expression to a different DType may produce wrong results.
pub fn simplify_typed(e: Expression, ctx: &DType) -> VortexResult<Expression> {
    let e = simplify(e)?;

    let e = remove_select(e, ctx)?;
    let e = remove_merge(e, ctx)?;
    let e = simplify(e)?;

    Ok(e)
}
