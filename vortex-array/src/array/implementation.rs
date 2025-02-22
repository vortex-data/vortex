use std::any::Any;
use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;

use crate::array::canonical::ArrayCanonicalImpl;
use crate::array::validity::ArrayValidityImpl;
use crate::array::visitor::ArrayVisitorImpl;
use crate::builders::ArrayBuilder;
use crate::stats::Statistics;
use crate::vtable::{EncodingVTable, VTableRef};
use crate::{
    Array, ArrayRef, ArrayStatisticsImpl, ArrayVariantsImpl, Canonical, Encoding, EncodingId,
};

/// A trait used to encapsulate common implementation behaviour for a Vortex [`Array`].
pub trait ArrayImpl:
    'static
    + Send
    + Sync
    + Debug
    + Clone
    + ArrayCanonicalImpl
    + ArrayStatisticsImpl
    + ArrayValidityImpl
    + ArrayVariantsImpl
    + ArrayVisitorImpl<<Self::Encoding as Encoding>::Metadata>
{
    type Encoding: Encoding;

    fn _len(&self) -> usize;
    fn _dtype(&self) -> &DType;
    fn _vtable(&self) -> VTableRef;
}

impl<A: ArrayImpl + 'static> Array for A {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_array(&self) -> ArrayRef {
        Arc::new(self.clone())
    }

    fn into_array(self) -> ArrayRef
    where
        Self: Sized,
    {
        Arc::new(self)
    }

    fn len(&self) -> usize {
        ArrayImpl::_len(self)
    }

    fn dtype(&self) -> &DType {
        ArrayImpl::_dtype(self)
    }

    fn encoding(&self) -> EncodingId {
        <Self as ArrayImpl>::Encoding::ID
    }

    fn vtable(&self) -> VTableRef {
        ArrayImpl::_vtable(self)
    }

    /// Returns whether the item at `index` is valid.
    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        if index >= self.len() {
            vortex_bail!("Index out of bounds: {} >= {}", index, self.len());
        }
        ArrayValidityImpl::_is_valid(self, index)
    }

    /// Returns whether the item at `index` is invalid.
    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        self.is_valid(index).map(|valid| !valid)
    }

    /// Returns whether all items in the array are valid.
    ///
    /// This is usually cheaper than computing a precise `valid_count`.
    fn all_valid(&self) -> VortexResult<bool> {
        ArrayValidityImpl::_all_valid(self)
    }

    /// Returns whether the array is all invalid.
    ///
    /// This is usually cheaper than computing a precise `invalid_count`.
    fn all_invalid(&self) -> VortexResult<bool> {
        ArrayValidityImpl::_all_invalid(self)
    }

    /// Returns the number of valid elements in the array.
    fn valid_count(&self) -> VortexResult<usize> {
        let count = ArrayValidityImpl::_valid_count(self)?;
        assert!(count <= self.len(), "Valid count exceeds array length");
        Ok(count)
    }

    /// Returns the number of invalid elements in the array.
    fn invalid_count(&self) -> VortexResult<usize> {
        let count = ArrayValidityImpl::_invalid_count(self)?;
        assert!(count <= self.len(), "Invalid count exceeds array length");
        Ok(count)
    }

    /// Returns the canonical validity mask for the array.
    fn validity_mask(&self) -> VortexResult<Mask> {
        let mask = ArrayValidityImpl::_validity_mask(self)?;
        assert_eq!(mask.len(), self.len(), "Validity mask length mismatch");
        Ok(mask)
    }

    /// Returns the canonical representation of the array.
    fn to_canonical(&self) -> VortexResult<Canonical> {
        let canonical = ArrayCanonicalImpl::_to_canonical(self)?;
        assert_eq!(
            canonical.as_ref().len(),
            self.len(),
            "Canonical length mismatch"
        );
        assert_eq!(
            canonical.as_ref().dtype(),
            self.dtype(),
            "Canonical dtype mismatch"
        );
        canonical.as_ref().statistics().inherit(self.statistics());
        Ok(canonical)
    }

    /// Writes the array into the canonical builder.
    ///
    /// The [`DType`] of the builder must match that of the array.
    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        // TODO(ngates): add dtype function to ArrayBuilder
        // if builder.dtype() != self.dtype() {
        //     vortex_bail!(
        //         "Builder dtype mismatch: expected {:?}, got {:?}",
        //         self.dtype(),
        //         builder.dtype()
        //     );
        // }
        let len = builder.len();
        ArrayCanonicalImpl::_append_to_builder(self, builder)?;
        assert_eq!(
            len + self.len(),
            builder.len(),
            "Builder length mismatch after writing array"
        );
        Ok(())
    }

    fn statistics(&self) -> &dyn Statistics {
        self
    }
}
