use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::aliases::hash_set::HashSet;
use vortex_array::ArrayData;
use vortex_dtype::field::Field;
use vortex_error::{vortex_err, VortexResult};

use crate::{unbox_any, ExprRef, VortexExpr};

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
        child_result.with_dyn(|a| {
            a.as_bool_array()
                .ok_or_else(|| vortex_err!("Child was not a bool array"))
                .and_then(|b| b.invert())
        })
    }

    fn collect_references<'a>(&'a self, references: &mut HashSet<&'a Field>) {
        self.child.collect_references(references)
    }
}

impl PartialEq<dyn Any> for Not {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x.child.eq(&self.child))
            .unwrap_or(false)
    }
}

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
