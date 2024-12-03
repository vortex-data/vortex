mod bool;
mod null;
mod primitive;

use std::any::Any;

pub use bool::*;
pub use null::*;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::Scalar;

use crate::ArrayData;

pub trait ArrayBuilder {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn finish(&mut self) -> VortexResult<ArrayData>;
}

impl dyn ArrayBuilder + '_ {
    /// A generic function to append a scalar to the builder.
    pub fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        match scalar.dtype() {
            DType::Null => self
                .as_any_mut()
                .downcast_mut::<NullBuilder>()
                .ok_or_else(|| vortex_err!("Cannot append null scalar to non-null builder"))?
                .append_null(),
            DType::Bool(_) => {}
            DType::Primitive(..) => {}
            DType::Utf8(_) => {}
            DType::Binary(_) => {}
            DType::Struct(..) => {}
            DType::List(..) => {}
            DType::Extension(_) => {}
        }
        Ok(())
    }
}
