mod binary;
mod bool;
mod extension;
mod lazy_validity_builder;
mod list;
mod null;
mod primitive;
mod struct_;
mod utf8;
mod varbinview_builder;

use std::any::Any;

pub use binary::*;
pub use bool::*;
pub use extension::*;
pub use list::*;
pub use null::*;
pub use primitive::*;
pub use utf8::*;
use vortex_dtype::{match_each_native_ptype, DType};
use vortex_error::{vortex_bail, vortex_err, VortexResult};
use vortex_scalar::{
    BinaryScalar, BoolScalar, ExtScalar, ListScalar, PrimitiveScalar, Scalar, StructScalar,
    Utf8Scalar,
};

use crate::builders::struct_::StructBuilder;
use crate::Array;

pub trait ArrayBuilder: Send {
    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn dtype(&self) -> &DType;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a "zero" value to the array.
    fn append_zero(&mut self) {
        self.append_zeros(1)
    }

    /// Appends n "zero" values to the array.
    fn append_zeros(&mut self, n: usize);

    /// Append a "null" value to the array.
    fn append_null(&mut self) {
        self.append_nulls(1)
    }

    /// Appends n "null" values to the array.
    fn append_nulls(&mut self, n: usize);

    /// Extends the array with the provided array, canonicalizing if necessary.
    fn extend_from_array(&mut self, array: Array) -> VortexResult<()>;

    fn finish(&mut self) -> VortexResult<Array>;
}

pub fn builder_with_capacity(dtype: &DType, capacity: usize) -> Box<dyn ArrayBuilder> {
    match dtype {
        DType::Null => Box::new(NullBuilder::new()),
        DType::Bool(n) => Box::new(BoolBuilder::with_capacity(*n, capacity)),
        DType::Primitive(ptype, n) => {
            match_each_native_ptype!(ptype, |$P| {
                Box::new(PrimitiveBuilder::<$P>::with_capacity(*n, capacity))
            })
        }
        DType::Utf8(n) => Box::new(Utf8Builder::with_capacity(*n, capacity)),
        DType::Binary(n) => Box::new(BinaryBuilder::with_capacity(*n, capacity)),
        DType::Struct(struct_dtype, n) => Box::new(StructBuilder::with_capacity(
            struct_dtype.clone(),
            *n,
            capacity,
        )),
        DType::List(dtype, n) => Box::new(ListBuilder::<u64>::with_capacity(
            dtype.clone(),
            *n,
            capacity,
        )),
        DType::Extension(ext_dtype) => {
            Box::new(ExtensionBuilder::with_capacity(ext_dtype.clone(), capacity))
        }
    }
}

pub trait ArrayBuilderExt: ArrayBuilder {
    /// A generic function to append a scalar to the builder.
    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        if !scalar.dtype().eq_ignore_nullability(self.dtype()) {
            vortex_bail!(
                "Builder has dtype {:?}, scalar has {:?}",
                self.dtype(),
                scalar.dtype()
            )
        }
        match scalar.dtype() {
            DType::Null => self
                .as_any_mut()
                .downcast_mut::<NullBuilder>()
                .ok_or_else(|| vortex_err!("Cannot append null scalar to non-null builder"))?
                .append_null(),
            DType::Bool(_) => self
                .as_any_mut()
                .downcast_mut::<BoolBuilder>()
                .ok_or_else(|| vortex_err!("Cannot append bool scalar to non-bool builder"))?
                .append_option(BoolScalar::try_from(scalar)?.value()),
            DType::Primitive(ptype, ..) => {
                match_each_native_ptype!(ptype, |$P| {
                    self
                    .as_any_mut()
                    .downcast_mut::<PrimitiveBuilder<$P>>()
                    .ok_or_else(|| {
                        vortex_err!("Cannot append primitive scalar to non-primitive builder")
                    })?
                    .append_option(PrimitiveScalar::try_from(scalar)?.typed_value::<$P>())
                })
            }
            DType::Utf8(_) => self
                .as_any_mut()
                .downcast_mut::<Utf8Builder>()
                .ok_or_else(|| vortex_err!("Cannot append utf8 scalar to non-utf8 builder"))?
                .append_option(Utf8Scalar::try_from(scalar)?.value()),
            DType::Binary(_) => self
                .as_any_mut()
                .downcast_mut::<BinaryBuilder>()
                .ok_or_else(|| vortex_err!("Cannot append binary scalar to non-binary builder"))?
                .append_option(BinaryScalar::try_from(scalar)?.value()),
            DType::Struct(..) => self
                .as_any_mut()
                .downcast_mut::<StructBuilder>()
                .ok_or_else(|| vortex_err!("Cannot append struct scalar to non-struct builder"))?
                .append_value(StructScalar::try_from(scalar)?)?,
            DType::List(..) => self
                .as_any_mut()
                .downcast_mut::<ListBuilder<u64>>()
                .ok_or_else(|| vortex_err!("Cannot append list scalar to non-list builder"))?
                .append_value(ListScalar::try_from(scalar)?)?,
            DType::Extension(..) => self
                .as_any_mut()
                .downcast_mut::<ExtensionBuilder>()
                .ok_or_else(|| {
                    vortex_err!("Cannot append extension scalar to non-extension builder")
                })?
                .append_value(ExtScalar::try_from(scalar)?)?,
        }
        Ok(())
    }
}

impl<T: ?Sized + ArrayBuilder> ArrayBuilderExt for T {}
