use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;
use vortex_array::compute::invert;
use vortex_array::ArrayData;
use vortex_dtype::Field;
use vortex_error::VortexResult;

use crate::{ExprRef, VortexExpr};

#[derive(Debug)]
pub struct Not {
    child: ExprRef,
}

impl Not {
    pub fn new_expr(child: ExprRef) -> ExprRef {
        Arc::new(Self { child })
    }

    pub fn child(&self) -> &ExprRef {
        &self.child
    }
}

impl Display for Not {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "!")?;
        self.child.fmt(f)
    }
}

impl VortexExpr for Not {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        let child_result = self.child.evaluate(batch)?;
        invert(&child_result)
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        self.child.collect_references(references)
    }
}

impl PartialEq for Not {
    fn eq(&self, other: &Not) -> bool {
        other.child.eq(&self.child)
    }
}

impl Eq for Not {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::array::BoolArray;
    use vortex_array::IntoArrayVariant;

    use crate::{Identity, Not};

    #[test]
    fn invert_booleans() {
        let not_expr = Not::new_expr(Arc::new(Identity));
        let bools = BoolArray::from_iter([false, true, false, false, true, true]);
        assert_eq!(
            not_expr
                .evaluate(bools.as_ref())
                .unwrap()
                .into_bool()
                .unwrap()
                .boolean_buffer()
                .iter()
                .collect::<Vec<_>>(),
            vec![true, false, true, true, false, false]
        );
    }
}
