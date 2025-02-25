use vortex_error::VortexResult;

use crate::transform::match_between::find_between;
// use crate::transform::match_between::find_between;
use crate::traversal::{MutNodeVisitor, Node, TransformResult};
use crate::{ExprRef, GetItem, Pack};

/// Simplifies an expression into an equivalent expression which is faster and easier to analyze.
///
/// If the scope dtype is known, see `simplify_typed` for a simplifier which uses dtype.
pub fn simplify(e: ExprRef) -> VortexResult<ExprRef> {
    let mut folder = Simplify;
    let e = e.transform(&mut folder).map(|e| e.result)?;
    Ok(find_between(e.clone()))
}

struct Simplify;

impl MutNodeVisitor for Simplify {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<ExprRef>> {
        // pack(l_1: e_1, ..., l_i: e_i, ..., l_n: e_n).get_item(l_i) = e_i where 0 <= i <= n
        if let Some(get_item) = node.as_any().downcast_ref::<GetItem>() {
            if let Some(pack) = get_item.child().as_any().downcast_ref::<Pack>() {
                let expr = pack.field(get_item.field())?;
                return Ok(TransformResult::yes(expr));
            }
        }
        Ok(TransformResult::no(node))
    }
}

#[cfg(test)]
mod tests {
    use crate::transform::simplify::simplify;
    use crate::{get_item, lit, pack};

    #[test]
    fn test_simplify() {
        let e = get_item("b", pack([("a", lit(1)), ("b", lit(2))]));
        let e = simplify(e).unwrap();
        assert_eq!(&e, &lit(2));
    }
}
