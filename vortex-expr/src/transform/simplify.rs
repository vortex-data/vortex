// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::transform::match_between::find_between;
// use crate::transform::match_between::find_between;
use crate::traversal::{NodeExt, Transformed};
use crate::{ExprRef, GetItemVTable, PackVTable};

/// Simplifies an expression into an equivalent expression which is faster and easier to analyze.
///
/// If the scope dtype is known, see `simplify_typed` for a simplifier which uses dtype.
pub fn simplify(e: ExprRef) -> VortexResult<ExprRef> {
    let e = e
        .transform_up(simplify_transformer)
        .map(|e| e.into_inner())?;
    Ok(find_between(e.clone()))
}

fn simplify_transformer(node: ExprRef) -> VortexResult<Transformed<ExprRef>> {
    // pack(l_1: e_1, ..., l_i: e_i, ..., l_n: e_n).get_item(l_i) = e_i where 0 <= i <= n
    if let Some(get_item) = node.as_opt::<GetItemVTable>()
        && let Some(pack) = get_item.child().as_opt::<PackVTable>()
    {
        let expr = pack.field(get_item.field())?;
        return Ok(Transformed::yes(expr));
    }
    Ok(Transformed::no(node))
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::NonNullable;

    use crate::transform::simplify::simplify;
    use crate::{get_item, lit, pack};

    #[test]
    fn test_simplify() {
        let e = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
        let e = simplify(e).unwrap();
        assert_eq!(&e, &lit(2));
    }
}
