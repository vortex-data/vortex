// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::pack::Pack;
use crate::expr::transform::match_between::find_between;
use crate::expr::traversal::{NodeExt, Transformed};

/// Simplifies an expression into an equivalent expression which is faster and easier to analyze.
///
/// If the scope dtype is known, see `simplify_typed` for a simplifier which uses dtype.
pub fn simplify(e: Expression) -> VortexResult<Expression> {
    let e = e
        .transform_up(simplify_transformer)
        .map(|e| e.into_inner())?;
    Ok(find_between(e))
}

fn simplify_transformer(node: Expression) -> VortexResult<Transformed<Expression>> {
    // pack(l_1: e_1, ..., l_i: e_i, ..., l_n: e_n).get_item(l_i) = e_i where 0 <= i <= n
    if let Some(get_item) = node.as_opt::<GetItem>()
        && let Some(pack) = get_item.child(0).as_opt::<Pack>()
    {
        let expr = pack.field(get_item.data())?;
        return Ok(Transformed::yes(expr));
    }
    Ok(Transformed::no(node))
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::NonNullable;

    use super::simplify;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::pack::pack;

    #[test]
    fn test_simplify() {
        let e = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
        let e = simplify(e).unwrap();
        assert_eq!(&e, &lit(2));
    }
}
