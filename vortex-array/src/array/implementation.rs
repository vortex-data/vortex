use std::any::Any;
use std::sync::Arc;

use arrow_array::builder::ArrayBuilder;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;

use crate::{Array, ArrayCanonicalImpl, ArrayRef, ArrayValidityImpl, ArrayVariantsImpl, Canonical};

/// A trait used to encapsulate common implementation behaviour for a Vortex [`Array`].
pub trait ArrayImpl:
    Send + Sync + Clone + ArrayCanonicalImpl + ArrayValidityImpl + ArrayVariantsImpl
{
    fn _len(&self) -> usize;
    fn _dtype(&self) -> &DType;
}

impl<A: ArrayImpl> Array for A {
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
        assert_eq!(canonical.len(), self.len(), "Canonical length mismatch");
        // assert_eq!(canonical.dtype(), self.dtype(), "Canonical dtype mismatch");
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
        ArrayCanonicalImpl::_to_builder(self, builder)?;
        assert_eq!(
            len + self.len(),
            builder.len(),
            "Builder length mismatch after writing array"
        );
        Ok(())
    }
}
