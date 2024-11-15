use std::any::Any;
use std::fmt::Display;

use vortex_array::ArrayData;
use vortex_error::VortexResult;

use crate::{unbox_any, VortexExpr};

#[derive(Debug, Eq, PartialEq)]
pub struct Identity;

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
}

impl PartialEq<dyn Any> for Identity {
    fn eq(&self, other: &dyn Any) -> bool {
        unbox_any(other)
            .downcast_ref::<Self>()
            .map(|x| x == self)
            .unwrap_or(false)
    }
}
