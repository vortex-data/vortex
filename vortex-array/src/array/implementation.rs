use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::array::canonical::ArrayCanonicalImpl;
use crate::array::convert::IntoArray;
use crate::array::operations::ArrayOperationsImpl;
use crate::array::validity::ArrayValidityImpl;
use crate::array::visitor::ArrayVisitorImpl;
use crate::builders::ArrayBuilder;
use crate::compute::{ComputeFn, InvocationArgs, Output};
use crate::stats::{Precision, Stat, StatsProviderExt, StatsSetRef};
use crate::vtable::VTableRef;
use crate::{
    Array, ArrayRef, ArrayStatistics, ArrayStatisticsImpl, ArrayVariantsImpl, ArrayVisitor,
    Canonical, Encoding, EncodingId,
};

/// A trait used to encapsulate common implementation behaviour for a Vortex [`Array`].
pub trait ArrayImpl:
    'static
    + Send
    + Sync
    + Debug
    + Clone
    + ArrayCanonicalImpl
    + ArrayOperationsImpl
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

    /// Dynamically invoke a kernel for the given compute function.
    fn _invoke(
        &self,
        _compute_fn: &ComputeFn,
        _args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        Ok(None)
    }
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

    /// Perform a constant-time slice of the array.
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        if start == 0 && stop == self.len() {
            return Ok(self.to_array());
        }

        if start > self.len() {
            vortex_bail!(OutOfBounds: start, 0, self.len());
        }
        if stop > self.len() {
            vortex_bail!(OutOfBounds: stop, 0, self.len());
        }
        if start > stop {
            vortex_bail!("start ({start}) must be <= stop ({stop})");
        }

        if start == stop {
            return Ok(Canonical::empty(self.dtype()).into_array());
        }

        // We know that constant array don't need stats propagation, so we can avoid the overhead of
        // computing derived stats and merging them in.
        // TODO(ngates): skip the is_constant check here, it can force an expensive compute.
        // TODO(ngates): provide a means to slice an array _without_ propagating stats.
        let derived_stats = (!self.is_constant()).then(|| {
            let stats = self.statistics().to_owned();

            // an array that is not constant can become constant after slicing
            let is_constant = stats.get_as::<bool>(Stat::IsConstant);
            let is_sorted = stats.get_as::<bool>(Stat::IsSorted);
            let is_strict_sorted = stats.get_as::<bool>(Stat::IsStrictSorted);

            let mut stats = stats.keep_inexact_stats(&[
                Stat::Max,
                Stat::Min,
                Stat::NullCount,
                Stat::UncompressedSizeInBytes,
            ]);

            if is_constant == Some(Precision::Exact(true)) {
                stats.set(Stat::IsConstant, Precision::exact(true));
            }
            if is_sorted == Some(Precision::Exact(true)) {
                stats.set(Stat::IsSorted, Precision::exact(true));
            }
            if is_strict_sorted == Some(Precision::Exact(true)) {
                stats.set(Stat::IsStrictSorted, Precision::exact(true));
            }

            stats
        });

        let sliced = ArrayOperationsImpl::_slice(self, start, stop)?;

        assert_eq!(
            sliced.len(),
            stop - start,
            "Slice length mismatch {}",
            self.encoding()
        );
        assert_eq!(
            sliced.dtype(),
            self.dtype(),
            "Slice dtype mismatch {}",
            self.encoding()
        );

        if let Some(derived_stats) = derived_stats {
            let mut stats = sliced.statistics().to_owned();
            stats.combine_sets(&derived_stats, self.dtype())?;
            for (stat, val) in stats.into_iter() {
                sliced.statistics().set(stat, val)
            }
        }

        Ok(sliced)
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
            "Canonical length mismatch {}",
            self.encoding(),
        );
        assert_eq!(
            canonical.as_ref().dtype(),
            self.dtype(),
            "Canonical dtype mismatch {}",
            self.encoding(),
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

    fn invoke(
        &self,
        compute_fn: &ComputeFn,
        args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        self._invoke(compute_fn, args)
    }
}
