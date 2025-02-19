use std::any::Any;
use std::sync::Arc;

use arrow_array::builder::ArrayBuilder;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};
use vortex_mask::Mask;

use crate::Canonical;

/// The base trait for all Vortex arrays.
///
/// Users should invoke functions on this trait. Implementations should implement the corresponding
/// function on the `_Impl` traits, e.g. [`ArrayValidityImpl`]. The functions here dispatch to the
/// implementations, while validating pre- and post-conditions.
pub trait Array: 'static + Send + Sync + ArrayValidityImpl {
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
        assert_eq!(canonical.dtype(), self.dtype(), "Canonical dtype mismatch");
        Ok(canonical)
    }

    /// Writes the array into the canonical builder.
    ///
    /// The [`DType`] of the builder must match that of the array.
    fn to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        if builder.dtype() != self.dtype() {
            vortex_bail!(
                "Builder dtype mismatch: expected {:?}, got {:?}",
                self.dtype(),
                builder.dtype()
            );
        }
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
}

/// A reference counted pointer to a dynamic [`Array`] trait object.
pub type ArrayRef = Arc<dyn Array>;

impl ToOwned for dyn Array {
    type Owned = ArrayRef;

    fn to_owned(&self) -> Self::Owned {
        self.to_array()
    }
}

/// Implementation trait for validity functions.
///
/// These functions should not be called directly, rather their equivalents on the base
/// [`Array`] trait should be used.
pub trait ArrayValidityImpl {
    /// Returns whether the `index` item is valid.
    ///
    /// ## Pre-conditions
    /// - `index` is less than the length of the array.
    fn _is_valid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether the array is all valid.
    fn _all_valid(&self) -> VortexResult<bool>;

    /// Returns whether the array is all invalid.
    fn _all_invalid(&self) -> VortexResult<bool>;

    /// Returns the number of valid elements in the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn _valid_count(&self) -> VortexResult<usize> {
        Ok(self._validity_mask()?.true_count())
    }

    /// Returns the number of invalid elements in the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn _invalid_count(&self) -> VortexResult<usize> {
        Ok(self._validity_mask()?.false_count())
    }

    /// Returns the canonical validity mask for the array.
    ///
    /// ## Post-conditions
    /// - The count is less than or equal to the length of the array.
    fn _validity_mask(&self) -> VortexResult<Mask>;
}

impl ArrayValidityImpl for Arc<dyn Array> {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.as_ref()._is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.as_ref()._all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.as_ref()._all_invalid()
    }

    fn _valid_count(&self) -> VortexResult<usize> {
        self.as_ref()._valid_count()
    }

    fn _invalid_count(&self) -> VortexResult<usize> {
        self.as_ref()._invalid_count()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.as_ref()._validity_mask()
    }
}

/// Implementation trait for canonicalization functions.
///
/// These functions should not be called directly, rather their equivalents on the base
/// [`Array`] trait should be used.
pub trait ArrayCanonicalImpl {
    /// Returns the canonical representation of the array.
    ///
    /// ## Post-conditions
    /// - The length is equal to that of the input array.
    /// - The [`DType`] is equal to that of the input array.
    fn _to_canonical(&self) -> VortexResult<Canonical>;

    /// Writes the array into the canonical builder.
    ///
    /// ## Post-conditions
    /// - The length of the builder is incremented by the length of the input array.
    fn _to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()>;
}

impl ArrayCanonicalImpl for Arc<dyn Array> {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        self.as_ref()._to_canonical()
    }

    fn _to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        self.as_ref()._to_builder(builder)
    }
}
