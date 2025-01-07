use std::any::Any;
use std::fmt::Display;
use std::sync::Arc;

use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::{ExprRef, VortexExpr};

#[derive(Debug, Eq, PartialEq)]
pub struct Identity;

impl Identity {
    pub fn new_expr() -> ExprRef {
        Arc::new(Identity)
    }
}

impl Display for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[]")
    }
}

impl VortexExpr for Identity {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
        Ok(batch.clone())
    }

    fn children(&self) -> Vec<&ExprRef> {
        vec![]
    }

    fn replacing_children(self: Arc<Self>, children: Vec<ExprRef>) -> ExprRef {
        assert_eq!(children.len(), 0);
        self
    }
}

// Return a global pointer to the identity token.
pub fn ident() -> ExprRef {
    Identity::new_expr()
}
