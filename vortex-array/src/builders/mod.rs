// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builders for Vortex arrays.
//!
//! Every logical type in Vortex has a canonical (uncompressed) in-memory encoding. This module
//! provides pre-allocated builders to construct new canonical arrays.
//!
//! ## Example:
//!
//! ```
//! use vortex_array::builders::{builder_with_capacity, ArrayBuilder};
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
//! assert_eq!(strings.scalar_at(0), "a".into());
//! assert_eq!(strings.scalar_at(1), "b".into());
//! assert_eq!(strings.scalar_at(2), "c".into());
//! assert_eq!(strings.scalar_at(3), "d".into());
//! ```

use std::any::Any;

use vortex_dtype::{DType, match_each_native_ptype};
use vortex_error::{VortexResult, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::{Scalar, match_each_decimal_value_type};

use crate::arrays::smallest_storage_type;
use crate::canonical::Canonical;
use crate::{Array, ArrayRef};

mod lazy_null_builder;
use lazy_null_builder::LazyNullBufferBuilder;

mod bool;
mod decimal;
mod extension;
mod fixed_size_list;
mod list;
mod null;
mod primitive;
mod struct_;
mod varbinview;

pub use bool::*;
pub use decimal::*;
pub use extension::*;
pub use fixed_size_list::*;
pub use list::*;
pub use null::*;
pub use primitive::*;
pub use struct_::*;
pub use varbinview::*;

#[cfg(test)]
mod tests;

/// The default capacity for builders.
///
/// This is equal to the default capacity for Arrow Arrays.
pub const DEFAULT_BUILDER_CAPACITY: usize = 1024;

pub trait ArrayBuilder: Send {
    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn dtype(&self) -> &DType;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a "zero" value to the array.
    ///
    /// Zero values are generally determined by [`Scalar::default_value`].
    fn append_zero(&mut self) {
        self.append_zeros(1)
    }

    /// Appends n "zero" values to the array.
    ///
    /// Zero values are generally determined by [`Scalar::default_value`].
    fn append_zeros(&mut self, n: usize);

    /// Append a "null" value to the array.
    ///
    /// Implementors should panic if this method is called on a non-nullable [`ArrayBuilder`].
    fn append_null(&mut self) {
        self.append_nulls(1)
    }

    /// The inner part of `append_nulls`.
    ///
    /// # Safety
    ///
    /// The array builder must be nullable.
    unsafe fn append_nulls_unchecked(&mut self, n: usize);

    /// Appends n "null" values to the array.
    ///
    /// Implementors should panic if this method is called on a non-nullable [`ArrayBuilder`].
    fn append_nulls(&mut self, n: usize) {
        assert!(
            self.dtype().is_nullable(),
            "tried to append {n} nulls to a non-nullable array builder"
        );

        // SAFETY: We check above that the array builder is nullable.
        unsafe {
            self.append_nulls_unchecked(n);
        }
    }

    /// Appends a default value to the array.
    fn append_default(&mut self) {
        self.append_defaults(1)
    }

    /// Appends n default values to the array.
    ///
    /// If the array builder is nullable, then this has the behavior of `self.append_nulls(n)`.
    /// If the array builder is non-nullable, then it has the behavior of `self.append_zeros(n)`.
    fn append_defaults(&mut self, n: usize) {
        if self.dtype().is_nullable() {
            self.append_nulls(n);
        } else {
            self.append_zeros(n);
        }
    }

    /// A generic function to append a scalar to the builder.
    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()>;

    /// The inner part of `extend_from_array`.
    ///
    /// # Safety
    ///
    /// The array that must have an equal [`DType`] to the array builder's `DType` (with nullability
    /// superset semantics).
    unsafe fn extend_from_array_unchecked(&mut self, array: &dyn Array);

    /// Extends the array with the provided array, canonicalizing if necessary.
    ///
    /// Implementors must validate that the passed in [`Array`] has the correct [`DType`].
    fn extend_from_array(&mut self, array: &dyn Array) {
        if !self.dtype().eq_with_nullability_superset(array.dtype()) {
            vortex_panic!(
                "tried to extend a builder with `DType` {} with an array with `DType {}",
                self.dtype(),
                array.dtype()
            );
        }

        // SAFETY: We checked that the array had a valid `DType` above.
        unsafe { self.extend_from_array_unchecked(array) }
    }

    /// Ensure that the builder can hold at least `capacity` number of items
    fn ensure_capacity(&mut self, capacity: usize);

    /// Override builders validity with the one provided.
    ///
    /// Note that this will have no effect on the final array if the array builder is non-nullable.
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

    /// Constructs a canonical array directly from the builder.
    ///
    /// This method provides a default implementation that creates an [`ArrayRef`] via `finish` and
    /// then converts it to canonical form. Specific builders can override this with optimized
    /// implementations that avoid the intermediate [`Array`] creation.
    fn finish_into_canonical(&mut self) -> Canonical {
        self.finish().to_canonical()
    }
}

/// Construct a new canonical builder for the given [`DType`].
///
///
/// # Example
///
/// ```
/// use vortex_array::builders::{builder_with_capacity, ArrayBuilder};
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
/// assert_eq!(strings.scalar_at(0), "a".into());
/// assert_eq!(strings.scalar_at(1), "b".into());
/// assert_eq!(strings.scalar_at(2), "c".into());
/// assert_eq!(strings.scalar_at(3), "d".into());
/// ```
pub fn builder_with_capacity(dtype: &DType, capacity: usize) -> Box<dyn ArrayBuilder> {
    match dtype {
        DType::Null => Box::new(NullBuilder::new()),
        DType::Bool(n) => Box::new(BoolBuilder::with_capacity(*n, capacity)),
        DType::Primitive(ptype, n) => {
            match_each_native_ptype!(ptype, |P| {
                Box::new(PrimitiveBuilder::<P>::with_capacity(*n, capacity))
            })
        }
        DType::Decimal(decimal_type, n) => {
            match_each_decimal_value_type!(smallest_storage_type(decimal_type), |D| {
                Box::new(DecimalBuilder::with_capacity::<D>(
                    capacity,
                    *decimal_type,
                    *n,
                ))
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
        DType::FixedSizeList(elem_dtype, list_size, null) => Box::new(
            FixedSizeListBuilder::with_capacity(elem_dtype.clone(), *list_size, *null, capacity),
        ),
        DType::Extension(ext_dtype) => {
            Box::new(ExtensionBuilder::with_capacity(ext_dtype.clone(), capacity))
        }
    }
}
