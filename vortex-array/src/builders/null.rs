use std::any::Any;

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::array::NullArray;
use crate::builders::ArrayBuilder;
use crate::{Array, IntoArray};

pub struct NullBuilder {
    length: usize,
}

impl Default for NullBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NullBuilder {
    pub fn new() -> Self {
        Self { length: 0 }
    }
}

impl ArrayBuilder for NullBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &DType::Null
    }

    fn len(&self) -> usize {
        self.length
    }

    fn append_zeros(&mut self, n: usize) {
        self.length += n;
    }

    fn append_nulls(&mut self, n: usize) {
        self.length += n;
    }

    fn finish(&mut self) -> VortexResult<Array> {
        Ok(NullArray::new(self.length).into_array())
    }
}
