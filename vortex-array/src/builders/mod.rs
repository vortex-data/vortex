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
//! use vortex_array::dtype::{DType, Nullability};
//! use vortex_array::{LEGACY_SESSION, VortexSessionExecute};
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
//! let mut ctx = LEGACY_SESSION.create_execution_ctx();
//!
//! assert_eq!(strings.execute_scalar(0, &mut ctx).unwrap(), "a".into());
//! assert_eq!(strings.execute_scalar(1, &mut ctx).unwrap(), "b".into());
//! assert_eq!(strings.execute_scalar(2, &mut ctx).unwrap(), "c".into());
//! assert_eq!(strings.execute_scalar(3, &mut ctx).unwrap(), "d".into());
//! ```

use std::any::Any;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::memory::HostAllocatorRef;
use crate::scalar::Scalar;

mod lazy_null_builder;
pub(crate) use lazy_null_builder::LazyBitBufferBuilder;

mod bool;
mod decimal;
pub mod dict;
mod extension;
mod fixed_size_list;
mod list;
mod listview;
mod null;
mod primitive;
mod struct_;
mod varbinview;

pub use bool::*;
pub use decimal::*;
pub use extension::*;
pub use fixed_size_list::*;
pub use list::*;
pub use listview::*;
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
    unsafe fn extend_from_array_unchecked(&mut self, array: &ArrayRef);

    /// Extends the array with the provided array, canonicalizing if necessary.
    ///
    /// Implementors must validate that the passed in [`ArrayRef`] has the correct [`DType`].
    fn extend_from_array(&mut self, array: &ArrayRef) {
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

    /// Allocate space for extra `additional` items
    fn reserve_exact(&mut self, additional: usize);

    /// Override builders validity with the one provided.
    ///
    /// Note that this will have no effect on the final array if the array builder is non-nullable.
    fn set_validity(&mut self, validity: Mask) {
        if !self.dtype().is_nullable() {
            return;
        }
        assert_eq!(self.len(), validity.len());
        unsafe { self.set_validity_unchecked(validity) }
    }

    /// override validity with the one provided, without checking lengths
    ///
    /// # Safety
    ///
    /// Given validity must have an equal length to [`self.len()`](Self::len).
    unsafe fn set_validity_unchecked(&mut self, validity: Mask);

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
    /// implementations that avoid the intermediate [`ArrayRef`] creation.
    fn finish_into_canonical(&mut self) -> Canonical;
}

/// Construct a new canonical builder for the given [`DType`].
///
///
/// # Example
///
/// ```
/// use vortex_array::builders::{builder_with_capacity, ArrayBuilder};
/// use vortex_array::dtype::{DType, Nullability};
/// use vortex_array::{LEGACY_SESSION, VortexSessionExecute};
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
/// let mut ctx = LEGACY_SESSION.create_execution_ctx();
///
/// assert_eq!(strings.execute_scalar(0, &mut ctx).unwrap(), "a".into());
/// assert_eq!(strings.execute_scalar(1, &mut ctx).unwrap(), "b".into());
/// assert_eq!(strings.execute_scalar(2, &mut ctx).unwrap(), "c".into());
/// assert_eq!(strings.execute_scalar(3, &mut ctx).unwrap(), "d".into());
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
            match_each_decimal_value_type!(
                DecimalType::smallest_decimal_value_type(decimal_type),
                |D| {
                    Box::new(DecimalBuilder::with_capacity::<D>(
                        capacity,
                        *decimal_type,
                        *n,
                    ))
                }
            )
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
        DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
        DType::List(dtype, n) => Box::new(ListViewBuilder::<u64, u64>::with_capacity(
            Arc::clone(dtype),
            *n,
            2 * capacity, // Arbitrarily choose 2 times the `offsets` capacity here.
            capacity,
        )),
        DType::FixedSizeList(elem_dtype, list_size, null) => {
            Box::new(FixedSizeListBuilder::with_capacity(
                Arc::clone(elem_dtype),
                *list_size,
                *null,
                capacity,
            ))
        }
        DType::Extension(ext_dtype) => {
            Box::new(ExtensionBuilder::with_capacity(ext_dtype.clone(), capacity))
        }
        DType::Variant(_) => {
            unimplemented!()
        }
    }
}

/// Construct a new canonical builder for the given [`DType`] using a host
/// [`crate::memory::HostAllocator`].
pub fn builder_with_capacity_in(
    allocator: HostAllocatorRef,
    dtype: &DType,
    capacity: usize,
) -> Box<dyn ArrayBuilder> {
    let _allocator = allocator;
    builder_with_capacity(dtype, capacity)
}
