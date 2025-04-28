use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::array::canonical::ArrayCanonicalImpl;
use crate::array::validity::ArrayValidityImpl;
use crate::array::visitor::ArrayVisitorImpl;
use crate::builders::ArrayBuilder;
use crate::stats::{Precision, Stat, StatsSetRef};
use crate::vtable::VTableRef;
use crate::{
    Array, ArrayRef, ArrayStatisticsImpl, ArrayVariantsImpl, ArrayVisitor, Canonical, Encoding,
    EncodingId,
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

    /// Replace the children of this array with the given arrays.
    ///
    /// ## Pre-conditions
    ///
    /// - The number of given children matches the current number of children of the array.
    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self>;
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
        self.vtable().id()
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
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return Ok(self.len() - invalid_count);
        }

        let count = ArrayValidityImpl::_valid_count(self)?;
        assert!(count <= self.len(), "Valid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(self.len() - count));

        Ok(count)
    }

    /// Returns the number of invalid elements in the array.
    fn invalid_count(&self) -> VortexResult<usize> {
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return Ok(invalid_count);
        }

        let count = ArrayValidityImpl::_invalid_count(self)?;
        assert!(count <= self.len(), "Invalid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(count));

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
        if builder.dtype() != self.dtype() {
            vortex_bail!(
                "Builder dtype mismatch: expected {}, got {}",
                self.dtype(),
                builder.dtype(),
            );
        }
        let len = builder.len();

        ArrayCanonicalImpl::_append_to_builder(self, builder)?;
        assert_eq!(
            len + self.len(),
            builder.len(),
            "Builder length mismatch after writing array for encoding {}",
            self.encoding(),
        );
        Ok(())
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        self._stats_ref()
    }

    fn with_children(&self, children: &[ArrayRef]) -> VortexResult<ArrayRef> {
        if self.nchildren() != children.len() {
            vortex_bail!("Child count mismatch");
        }

        for (s, o) in self.children().iter().zip(children.iter()) {
            assert_eq!(s.len(), o.len());
        }

        Ok(self._with_children(children)?.into_array())
    }
}
