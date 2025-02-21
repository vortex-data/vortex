mod canonical;
mod convert;
pub mod data;
mod implementation;
mod statistics;
mod validity;
mod variants;
mod visitor;

use std::any::{type_name, Any};
use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

pub use canonical::*;
pub use convert::*;
pub use implementation::*;
pub use statistics::*;
pub use validity::*;
pub use variants::*;
pub use visitor::*;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexError, VortexResult};
use vortex_mask::Mask;

use crate::arrays::{
    BoolEncoding, ExtensionEncoding, ListEncoding, NullEncoding, PrimitiveEncoding, StructEncoding,
    VarBinEncoding, VarBinViewEncoding,
};
use crate::builders::ArrayBuilder;
use crate::stats::Statistics;
use crate::visitor::ArrayVisitor;
use crate::vtable::{EncodingVTable, VTableRef};
use crate::{Canonical, EncodingId};

/// The base trait for all Vortex arrays.
///
/// Users should invoke functions on this trait. Implementations should implement the corresponding
/// function on the `_Impl` traits, e.g. [`ArrayValidityImpl`]. The functions here dispatch to the
/// implementations, while validating pre- and post-conditions.
pub trait Array: Send + Sync + Debug + ArrayStatistics + ArrayVariants {
    /// Returns the array as a reference to a generic [`Any`] trait object.
    fn as_any(&self) -> &dyn Any;

    /// Returns the array as an [`Arc`] reference to a generic [`Any`] trait object.
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Returns the array as an [`ArrayRef`].
    fn to_array(&self) -> ArrayRef;

    /// Converts the array into an [`ArrayRef`].
    fn into_array(self) -> ArrayRef
    where
        Self: Sized;

    /// Returns the length of the array.
    fn len(&self) -> usize;

    /// Returns whether the array is empty (has zero rows).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the logical Vortex [`DType`] of the array.
    fn dtype(&self) -> &DType;

    /// Returns the encoding of the array.
    fn encoding(&self) -> EncodingId;

    /// Returns the encoding VTable.
    fn vtable(&self) -> VTableRef;

    /// Returns whether the array is of the given encoding.
    fn is_encoding(&self, encoding: EncodingId) -> bool {
        self.encoding() == encoding
    }

    /// Returns whether this array is an arrow encoding.
    // TODO(ngates): this shouldn't live here.
    fn is_arrow(&self) -> bool {
        self.is_encoding(NullEncoding.id())
            || self.is_encoding(BoolEncoding.id())
            || self.is_encoding(PrimitiveEncoding.id())
            || self.is_encoding(VarBinEncoding.id())
            || self.is_encoding(VarBinViewEncoding.id())
    }

    /// Whether the array is of a canonical encoding.
    // TODO(ngates): this shouldn't live here.
    fn is_canonical(&self) -> bool {
        self.is_encoding(NullEncoding.id())
            || self.is_encoding(BoolEncoding.id())
            || self.is_encoding(PrimitiveEncoding.id())
            || self.is_encoding(StructEncoding.id())
            || self.is_encoding(ListEncoding.id())
            || self.is_encoding(VarBinViewEncoding.id())
            || self.is_encoding(ExtensionEncoding.id())
    }

    /// Returns whether the item at `index` is valid.
    fn is_valid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether the item at `index` is invalid.
    fn is_invalid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether all items in the array are valid.
    ///
    /// This is usually cheaper than computing a precise `valid_count`.
    fn all_valid(&self) -> VortexResult<bool>;

    /// Returns whether the array is all invalid.
    ///
    /// This is usually cheaper than computing a precise `invalid_count`.
    fn all_invalid(&self) -> VortexResult<bool>;

    /// Returns the number of valid elements in the array.
    fn valid_count(&self) -> VortexResult<usize>;

    /// Returns the number of invalid elements in the array.
    fn invalid_count(&self) -> VortexResult<usize>;

    /// Returns the canonical validity mask for the array.
    fn validity_mask(&self) -> VortexResult<Mask>;

    /// Returns the canonical representation of the array.
    fn to_canonical(&self) -> VortexResult<Canonical>;

    /// Writes the array into the canonical builder.
    ///
    /// The [`DType`] of the builder must match that of the array.
    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()>;

    /// Accepts a visitor to traverse the array.
    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()>;

    /// Returns the statistics of the array.
    // TODO(ngates): change how this works. It's weird.
    fn statistics(&self) -> &dyn Statistics;
}

impl Array for Arc<dyn Array> {
    fn as_any(&self) -> &dyn Any {
        self.as_ref().as_any()
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_array(&self) -> ArrayRef {
        self.clone()
    }

    fn into_array(self) -> ArrayRef {
        self
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn dtype(&self) -> &DType {
        self.as_ref().dtype()
    }

    fn encoding(&self) -> EncodingId {
        self.as_ref().encoding()
    }

    fn vtable(&self) -> VTableRef {
        self.as_ref().vtable()
    }

    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        self.as_ref().is_valid(index)
    }

    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        self.as_ref().is_invalid(index)
    }

    fn all_valid(&self) -> VortexResult<bool> {
        self.as_ref().all_valid()
    }

    fn all_invalid(&self) -> VortexResult<bool> {
        self.as_ref().all_invalid()
    }

    fn valid_count(&self) -> VortexResult<usize> {
        self.as_ref().valid_count()
    }

    fn invalid_count(&self) -> VortexResult<usize> {
        self.as_ref().invalid_count()
    }

    fn validity_mask(&self) -> VortexResult<Mask> {
        self.as_ref().validity_mask()
    }

    fn to_canonical(&self) -> VortexResult<Canonical> {
        self.as_ref().to_canonical()
    }

    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        self.as_ref().append_to_builder(builder)
    }

    fn accept(&self, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        self.as_ref().accept(visitor)
    }

    fn statistics(&self) -> &dyn Statistics {
        self.as_ref().statistics()
    }
}

/// A reference counted pointer to a dynamic [`Array`] trait object.
pub type ArrayRef = Arc<dyn Array>;

impl ToOwned for dyn Array {
    type Owned = ArrayRef;

    fn to_owned(&self) -> Self::Owned {
        self.to_array()
    }
}

impl<A: Array + Clone> TryFromArrayRef for A {
    fn try_from_array(array: ArrayRef) -> VortexResult<Self> {
        Ok(Arc::unwrap_or_clone(
            array
                .as_any_arc()
                .downcast::<A>()
                .map_err(|_| vortex_err!("Cannot downcast to {}", type_name::<A>()))?,
        ))
    }
}

impl<A: Array + Clone> TryFromArrayRef for Arc<A> {
    fn try_from_array(array: ArrayRef) -> VortexResult<Self> {
        array
            .as_any_arc()
            .downcast::<A>()
            .map_err(|_| vortex_err!("Cannot downcast to {}", type_name::<A>()))
    }
}
