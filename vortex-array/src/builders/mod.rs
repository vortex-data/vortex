//! Builders for Vortex arrays.
//!
//! Every logical type in Vortex has a canonical (uncompressed) in-memory encoding. This module
//! provides pre-allocated builders to construct new canonical arrays.
//!
//! ## Example:
//!
//! ```
//! use vortex_array::builders::{builder_with_capacity, ArrayBuilderExt};
//! use vortex_dtype::{DType, Nullability};
//!
//! // Create a new builder for string data.
//! let mut builder = builder_with_capacity(&DType::Utf8(Nullability::NonNullable), 4);
//!
//! builder.append_scalar(&"a".into()).unwrap();
//! builder.append_scalar(&"b".into()).unwrap();
//! builder.append_scalar(&"c".into()).unwrap();
//! builder.append_scalar(&"d".into()).unwrap();
//!
//! let strings = builder.finish();
//!
//! assert_eq!(strings.scalar_at(0).unwrap(), "a".into());
//! assert_eq!(strings.scalar_at(1).unwrap(), "b".into());
//! assert_eq!(strings.scalar_at(2).unwrap(), "c".into());
//! assert_eq!(strings.scalar_at(3).unwrap(), "d".into());
//! ```

mod bool;
mod decimal;
mod extension;
mod lazy_validity_builder;
mod list;
mod null;
mod primitive;
mod struct_;
mod varbinview;

use std::any::Any;

pub use bool::*;
pub use decimal::*;
pub use extension::*;
pub use list::*;
pub use null::*;
pub use primitive::*;
pub use struct_::*;
pub use varbinview::*;
use vortex_dtype::{DType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;
use vortex_scalar::{
    BinaryScalar, BoolScalar, DecimalValue, ExtScalar, ListScalar, PrimitiveScalar, Scalar,
    ScalarValue, StructScalar, Utf8Scalar, match_each_decimal_value, match_each_decimal_value_type,
};

use crate::arrays::smallest_storage_type;
use crate::{Array, ArrayRef};

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
    fn extend_from_array(&mut self, array: &dyn Array) -> VortexResult<()>;

    /// Ensure that the builder can hold at least `capacity` number of items
    fn ensure_capacity(&mut self, capacity: usize);

    /// Override builders validity with the one provided
    fn set_validity(&mut self, validity: Mask);

    /// Constructs an Array from the builder components.
    ///
    /// # Panics
    ///
    /// This function may panic if the builder's methods are called with invalid arguments. If only
    /// the methods on this interface are used, the builder should not panic. However, specific
    /// builders have interfaces that may be misused. For example, if the number of values in a
    /// [PrimitiveBuilder]'s [vortex_buffer::BufferMut] does not match the number of validity bits,
    /// the PrimitiveBuilder's [Self::finish] will panic.
    fn finish(&mut self) -> ArrayRef;
}

/// Construct a new canonical builder for the given [`DType`].
///
///
/// # Example
///
/// ```
/// use vortex_array::builders::{builder_with_capacity, ArrayBuilderExt};
/// use vortex_dtype::{DType, Nullability};
///
/// // Create a new builder for string data.
/// let mut builder = builder_with_capacity(&DType::Utf8(Nullability::NonNullable), 4);
///
/// builder.append_scalar(&"a".into()).unwrap();
/// builder.append_scalar(&"b".into()).unwrap();
/// builder.append_scalar(&"c".into()).unwrap();
/// builder.append_scalar(&"d".into()).unwrap();
///
/// let strings = builder.finish();
///
/// assert_eq!(strings.scalar_at(0).unwrap(), "a".into());
/// assert_eq!(strings.scalar_at(1).unwrap(), "b".into());
/// assert_eq!(strings.scalar_at(2).unwrap(), "c".into());
/// assert_eq!(strings.scalar_at(3).unwrap(), "d".into());
/// ```
pub fn builder_with_capacity(dtype: &DType, capacity: usize) -> Box<dyn ArrayBuilder> {
    match dtype {
        DType::Null => Box::new(NullBuilder::new()),
        DType::Bool(n) => Box::new(BoolBuilder::with_capacity(*n, capacity)),
        DType::Primitive(ptype, n) => {
            match_each_native_ptype!(ptype, |$P| {
                Box::new(PrimitiveBuilder::<$P>::with_capacity(*n, capacity))
            })
        }
        DType::Decimal(decimal_type, n) => {
            match_each_decimal_value_type!(smallest_storage_type(decimal_type), |$D| {
                Box::new(DecimalBuilder::with_capacity::<$D>(capacity, decimal_type.clone(), *n))
            })
        }
        DType::Utf8(n) => Box::new(VarBinViewBuilder::with_capacity(DType::Utf8(*n), capacity)),
        DType::Binary(n) => Box::new(VarBinViewBuilder::with_capacity(
            DType::Binary(*n),
            capacity,
        )),
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
    /// A generic function to append a scalar value to the builder.
    fn append_scalar_value(&mut self, value: ScalarValue) -> VortexResult<()> {
        if value.is_null() {
            self.append_null();
            Ok(())
        } else {
            self.append_scalar(&Scalar::new(self.dtype().clone(), value))
        }
    }

    /// A generic function to append a scalar to the builder.
    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        if scalar.dtype() != self.dtype() {
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
            DType::Decimal(..) => match scalar.as_decimal().decimal_value() {
                None => self.append_null(),
                Some(v) => match_each_decimal_value!(v, |$value| {
                    self.as_any_mut()
                        .downcast_mut::<DecimalBuilder>()
                        .ok_or_else(|| {
                            ::vortex_error::vortex_err!(
                                "Cannot append decimal scalar of type {} to builder of type",
                                stringify!($ty),
                            )
                        })?
                        .append_value(*$value)
                }),
            },
            DType::Utf8(_) => self
                .as_any_mut()
                .downcast_mut::<VarBinViewBuilder>()
                .ok_or_else(|| vortex_err!("Cannot append utf8 scalar to non-utf8 builder"))?
                .append_option(Utf8Scalar::try_from(scalar)?.value()),
            DType::Binary(_) => self
                .as_any_mut()
                .downcast_mut::<VarBinViewBuilder>()
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
