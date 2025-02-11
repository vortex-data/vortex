use std::any::Any;

use vortex_dtype::{DType, Nullability};
use vortex_error::VortexResult;

use crate::builders::varbinview_builder::VarBinViewBuilder;
use crate::builders::ArrayBuilder;
use crate::Array;

pub struct BinaryBuilder {
    varbinview_builder: VarBinViewBuilder,
}

impl BinaryBuilder {
    pub fn with_capacity(nullability: Nullability, capacity: usize) -> Self {
        Self {
            varbinview_builder: VarBinViewBuilder::with_capacity(
                DType::Binary(nullability),
                capacity,
            ),
        }
    }

    #[inline]
    pub fn append_value<S: AsRef<[u8]>>(&mut self, value: S) {
        self.varbinview_builder.append_value(value.as_ref());
    }

    #[inline]
    pub fn append_option<S: AsRef<[u8]>>(&mut self, value: Option<S>) {
        self.varbinview_builder.append_option(value.as_ref());
    }
}

impl ArrayBuilder for BinaryBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.varbinview_builder.dtype()
    }

    #[inline]
    fn len(&self) -> usize {
        self.varbinview_builder.len()
    }

    #[inline]
    fn append_zeros(&mut self, n: usize) {
        self.varbinview_builder.append_zeros(n);
    }

    #[inline]
    fn append_nulls(&mut self, n: usize) {
        self.varbinview_builder.append_nulls(n);
    }

    #[inline]
    fn extend_from_array(&mut self, array: Array) -> VortexResult<()> {
        self.varbinview_builder.extend_from_array(array)
    }

    fn finish(&mut self) -> VortexResult<Array> {
        self.varbinview_builder.finish()
    }
}
