use std::any::Any;

use vortex::Array;
use vortex_error::VortexResult;

use crate::{unbox_any, VortexExpr};

#[derive(Debug, Eq, PartialEq)]
pub struct Identity;

impl VortexExpr for Identity {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, batch: &Array) -> VortexResult<Array> {
        Ok(batch.clone())
    }
}

impl PartialEq<dyn Any> for Identity {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x == self)
            .unwrap_or(false)
    }
}
