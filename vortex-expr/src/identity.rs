use std::any::Any;
use std::fmt::Display;
use std::sync::{Arc, LazyLock};

use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::{ExprRef, VortexExpr};

static IDENTITY: LazyLock<ExprRef> = LazyLock::new(|| Arc::new(Identity));

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Identity;

impl Identity {
    pub fn new_expr() -> ExprRef {
        IDENTITY.clone()
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

    fn unchecked_evaluate(&self, batch: &ArrayData) -> VortexResult<ArrayData> {
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

#[cfg(test)]
mod tests {
    use crate::{ident, test_harness};

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        assert_eq!(ident().return_dtype(&dtype).unwrap(), dtype);
        assert_eq!(ident().return_dtype(&dtype).unwrap(), dtype);
    }
}
