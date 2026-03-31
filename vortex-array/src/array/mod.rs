// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;
use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::AnyCanonical;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::Canonical;
use crate::DynArrayEq;
use crate::DynArrayHash;
use crate::ExecutionCtx;
use crate::LEGACY_SESSION;
use crate::ToCanonical;
use crate::VortexSessionExecute;
use crate::aggregate_fn::fns::sum::sum;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::DictArray;
use crate::arrays::FilterArray;
use crate::arrays::Null;
use crate::arrays::Primitive;
use crate::arrays::SliceArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinView;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::hash;
use crate::matcher::Matcher;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable::ArrayId;
use crate::vtable::ArrayInner;
use crate::vtable::ArrayView;
use crate::vtable::DynVTable;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;

/// The public API trait for all Vortex arrays.
///
/// This trait is sealed and cannot be implemented outside of `vortex-array`.
/// Use [`ArrayRef`] as the primary handle for working with arrays.
#[doc(hidden)]
pub trait DynArray:
    'static + private::Sealed + Send + Sync + Debug + DynArrayEq + DynArrayHash
{
    /// Returns the array as a reference to a generic [`Any`] trait object.
    fn as_any(&self) -> &dyn Any;

    /// Returns the array as an `Arc<dyn Any + Send + Sync>`.
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    /// Returns the array as an [`ArrayRef`].
    fn to_array(&self) -> ArrayRef;

    /// Returns the length of the array.
    fn len(&self) -> usize;

    /// Returns whether the array is empty (has zero rows).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the logical Vortex [`DType`] of the array.
    fn dtype(&self) -> &DType;

    /// Returns the vtable of the array.
    fn vtable(&self) -> &dyn DynVTable;

    /// Returns the encoding ID of the array.
    fn encoding_id(&self) -> ArrayId;

    /// Performs a constant-time slice of the array.
    fn slice(&self, this: &ArrayRef, range: Range<usize>) -> VortexResult<ArrayRef>;

    /// Wraps the array in a [`FilterArray`] such that it is logically filtered by the given mask.
    fn filter(&self, this: &ArrayRef, mask: Mask) -> VortexResult<ArrayRef>;

    /// Wraps the array in a [`DictArray`] such that it is logically taken by the given indices.
    fn take(&self, this: &ArrayRef, indices: ArrayRef) -> VortexResult<ArrayRef>;

    /// Fetch the scalar at the given index.
    ///
    /// This method panics if the index is out of bounds for the array.
    fn scalar_at(&self, this: &ArrayRef, index: usize) -> VortexResult<Scalar>;

    /// Returns whether the item at `index` is valid.
    fn is_valid(&self, this: &ArrayRef, index: usize) -> VortexResult<bool>;

    /// Returns whether the item at `index` is invalid.
    fn is_invalid(&self, this: &ArrayRef, index: usize) -> VortexResult<bool>;

    /// Returns whether all items in the array are valid.
    ///
    /// This is usually cheaper than computing a precise `valid_count`, but may return false
    /// negatives.
    fn all_valid(&self, this: &ArrayRef) -> VortexResult<bool>;

    /// Returns whether the array is all invalid.
    ///
    /// This is usually cheaper than computing a precise `invalid_count`, but may return false
    /// negatives.
    fn all_invalid(&self, this: &ArrayRef) -> VortexResult<bool>;

    /// Returns the number of valid elements in the array.
    fn valid_count(&self, this: &ArrayRef) -> VortexResult<usize>;

    /// Returns the number of invalid elements in the array.
    fn invalid_count(&self, this: &ArrayRef) -> VortexResult<usize>;

    /// Returns the [`Validity`] of the array.
    fn validity(&self, this: &ArrayRef) -> VortexResult<Validity>;

    /// Returns the canonical validity mask for the array.
    fn validity_mask(&self, this: &ArrayRef) -> VortexResult<Mask>;

    /// Returns the canonical representation of the array.
    fn to_canonical(&self, this: &ArrayRef) -> VortexResult<Canonical>;

    /// Writes the array into the canonical builder.
    ///
    /// The [`DType`] of the builder must match that of the array.
    fn append_to_builder(
        &self,
        this: &ArrayRef,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Returns the statistics of the array.
    // TODO(ngates): change how this works. It's weird.
    fn statistics(&self) -> StatsSetRef<'_>;

    /// Replaces the children of the array with the given array references.
    fn with_children(&self, this: &ArrayRef, children: Vec<ArrayRef>) -> VortexResult<ArrayRef>;

    // --- Visitor methods (formerly in ArrayVisitor) ---

    /// Returns the children of the array.
    fn children(&self, this: &ArrayRef) -> Vec<ArrayRef>;

    /// Returns the number of children of the array.
    fn nchildren(&self, this: &ArrayRef) -> usize;

    /// Returns the nth child of the array without allocating a Vec.
    ///
    /// Returns `None` if the index is out of bounds.
    fn nth_child(&self, this: &ArrayRef, idx: usize) -> Option<ArrayRef>;

    /// Returns the names of the children of the array.
    fn children_names(&self, this: &ArrayRef) -> Vec<String>;

    /// Returns the array's children with their names.
    fn named_children(&self, this: &ArrayRef) -> Vec<(String, ArrayRef)>;

    /// Returns the buffers of the array.
    fn buffers(&self, this: &ArrayRef) -> Vec<ByteBuffer>;

    /// Returns the buffer handles of the array.
    fn buffer_handles(&self, this: &ArrayRef) -> Vec<BufferHandle>;

    /// Returns the names of the buffers of the array.
    fn buffer_names(&self, this: &ArrayRef) -> Vec<String>;

    /// Returns the array's buffers with their names.
    fn named_buffers(&self, this: &ArrayRef) -> Vec<(String, BufferHandle)>;

    /// Returns the number of buffers of the array.
    fn nbuffers(&self, this: &ArrayRef) -> usize;

    /// Returns the serialized metadata of the array, or `None` if the array does not
    /// support serialization.
    fn metadata(&self, this: &ArrayRef) -> VortexResult<Option<Vec<u8>>>;

    /// Formats a human-readable metadata description.
    fn metadata_fmt(&self, this: &ArrayRef, f: &mut Formatter<'_>) -> std::fmt::Result;

    /// Checks if all buffers in the array tree are host-resident.
    fn is_host(&self, this: &ArrayRef) -> bool;

    /// Count the number of buffers encoded by self and all child arrays.
    fn nbuffers_recursive(&self) -> usize {
        let this = self.to_array();
        this.children()
            .iter()
            .map(|c| c.nbuffers_recursive())
            .sum::<usize>()
            + this.nbuffers()
    }

    /// Depth-first traversal of the array and its children.
    fn depth_first_traversal(&self) -> DepthFirstArrayIterator {
        DepthFirstArrayIterator {
            stack: vec![self.to_array()],
        }
    }
}

/// A depth-first pre-order iterator over an Array.
pub struct DepthFirstArrayIterator {
    stack: Vec<ArrayRef>,
}

impl Iterator for DepthFirstArrayIterator {
    type Item = ArrayRef;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.stack.pop()?;
        for child in next.children().into_iter().rev() {
            self.stack.push(child);
        }
        Some(next)
    }
}

/// A reference-counted pointer to a type-erased array.
#[derive(Clone)]
pub struct ArrayRef(Arc<dyn DynArray>);

impl ArrayRef {
    /// Create from an `Arc<dyn DynArray>`.
    pub fn from_inner(inner: Arc<dyn DynArray>) -> Self {
        Self(inner)
    }

    /// Returns a reference to the inner Arc.
    pub fn inner(&self) -> &Arc<dyn DynArray> {
        &self.0
    }

    /// Returns a reference to the inner dyn DynArray.
    pub fn as_dyn(&self) -> &dyn DynArray {
        self.0.as_ref()
    }

    /// Returns true if the two ArrayRefs point to the same allocation.
    pub fn ptr_eq(this: &ArrayRef, other: &ArrayRef) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }
}

impl Deref for ArrayRef {
    type Target = dyn DynArray;
    fn deref(&self) -> &dyn DynArray {
        self.0.as_ref()
    }
}

impl AsRef<dyn DynArray> for ArrayRef {
    fn as_ref(&self) -> &dyn DynArray {
        self.0.as_ref()
    }
}

impl Debug for ArrayRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&*self.0, f)
    }
}

impl std::fmt::Display for ArrayRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&*self.0, f)
    }
}

#[allow(clippy::same_name_method)]
impl ArrayRef {
    /// Returns the array as a reference to a generic [`Any`] trait object.
    pub fn as_any(&self) -> &dyn Any {
        self.0.as_any()
    }

    /// Returns the array as an `Arc<dyn Any + Send + Sync>`.
    pub fn as_any_arc(self) -> Arc<dyn Any + Send + Sync> {
        self.0.as_any_arc()
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the array is empty (has zero rows).
    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    /// Returns the logical Vortex [`DType`] of the array.
    pub fn dtype(&self) -> &DType {
        self.0.dtype()
    }

    /// Returns the vtable of the array.
    pub fn vtable(&self) -> &dyn DynVTable {
        self.0.vtable()
    }

    /// Returns the encoding ID of the array.
    pub fn encoding_id(&self) -> ArrayId {
        self.0.encoding_id()
    }

    /// Performs a constant-time slice of the array.
    pub fn slice(&self, range: Range<usize>) -> VortexResult<ArrayRef> {
        self.0.slice(self, range)
    }

    /// Wraps the array in a [`FilterArray`] such that it is logically filtered by the given mask.
    pub fn filter(&self, mask: Mask) -> VortexResult<ArrayRef> {
        self.0.filter(self, mask)
    }

    /// Wraps the array in a [`DictArray`] such that it is logically taken by the given indices.
    pub fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        self.0.take(self, indices)
    }

    /// Fetch the scalar at the given index.
    pub fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        self.0.scalar_at(self, index)
    }

    /// Returns whether the item at `index` is valid.
    pub fn is_valid(&self, index: usize) -> VortexResult<bool> {
        self.0.is_valid(self, index)
    }

    /// Returns whether the item at `index` is invalid.
    pub fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        self.0.is_invalid(self, index)
    }

    /// Returns whether all items in the array are valid.
    pub fn all_valid(&self) -> VortexResult<bool> {
        self.0.all_valid(self)
    }

    /// Returns whether the array is all invalid.
    pub fn all_invalid(&self) -> VortexResult<bool> {
        self.0.all_invalid(self)
    }

    /// Returns the number of valid elements in the array.
    pub fn valid_count(&self) -> VortexResult<usize> {
        self.0.valid_count(self)
    }

    /// Returns the number of invalid elements in the array.
    pub fn invalid_count(&self) -> VortexResult<usize> {
        self.0.invalid_count(self)
    }

    /// Returns the [`Validity`] of the array.
    pub fn validity(&self) -> VortexResult<Validity> {
        self.0.validity(self)
    }

    /// Returns the canonical validity mask for the array.
    pub fn validity_mask(&self) -> VortexResult<Mask> {
        self.0.validity_mask(self)
    }

    /// Returns the canonical representation of the array.
    pub fn to_canonical(&self) -> VortexResult<Canonical> {
        self.0.to_canonical(self)
    }

    /// Writes the array into the canonical builder.
    pub fn append_to_builder(
        &self,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.0.append_to_builder(self, builder, ctx)
    }

    /// Returns the statistics of the array.
    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.0.statistics()
    }

    /// Replaces the children of the array with the given array references.
    pub fn with_children(&self, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        self.0.with_children(self, children)
    }

    /// Does the array match the given matcher.
    pub fn is<M: Matcher>(&self) -> bool {
        M::matches(&*self.0)
    }

    /// Returns the array downcast by the given matcher.
    pub fn as_<M: Matcher>(&self) -> M::Match<'_> {
        self.as_opt::<M>().vortex_expect("Failed to downcast")
    }

    /// Returns the array downcast by the given matcher.
    pub fn as_opt<M: Matcher>(&self) -> Option<M::Match<'_>> {
        M::try_match(&*self.0)
    }

    /// Returns the array downcast to the given `ArrayInner<V>` as an owned object.
    pub fn try_into<V: VTable>(self) -> Result<ArrayInner<V>, ArrayRef> {
        if !self.is::<V>() {
            return Err(self);
        }
        let arc = self.0.as_any_arc();
        let typed: Arc<ArrayInner<V>> = arc
            .downcast::<ArrayInner<V>>()
            .map_err(|_| vortex_err!("failed to downcast"))
            .vortex_expect("Failed to downcast");
        Ok(match Arc::try_unwrap(typed) {
            Ok(inner) => inner,
            Err(arc) => arc.deref().clone(),
        })
    }

    /// Returns a reference to the typed `ArrayInner<V>` if this array matches the given vtable type.
    pub fn as_typed<V: VTable>(&self) -> Option<&ArrayInner<V>> {
        self.0.as_any().downcast_ref::<ArrayInner<V>>()
    }

    /// Returns the constant scalar if this is a constant array.
    pub fn as_constant(&self) -> Option<Scalar> {
        self.as_opt::<Constant>().map(|a| a.scalar().clone())
    }

    /// Total size of the array in bytes, including all children and buffers.
    pub fn nbytes(&self) -> u64 {
        let mut nbytes = 0;
        for array in self.0.depth_first_traversal() {
            for buffer in array.buffers() {
                nbytes += buffer.len() as u64;
            }
        }
        nbytes
    }

    /// Returns whether this array is an arrow encoding.
    pub fn is_arrow(&self) -> bool {
        self.is::<Null>()
            || self.is::<Bool>()
            || self.is::<Primitive>()
            || self.is::<VarBin>()
            || self.is::<VarBinView>()
    }

    /// Whether the array is of a canonical encoding.
    pub fn is_canonical(&self) -> bool {
        self.is::<AnyCanonical>()
    }

    /// Returns a new array with the child at `child_idx` replaced by `replacement`.
    pub fn with_child(&self, child_idx: usize, replacement: ArrayRef) -> VortexResult<ArrayRef> {
        let mut children: Vec<ArrayRef> = self.children();
        vortex_ensure!(
            child_idx < children.len(),
            "child index {} out of bounds for array with {} children",
            child_idx,
            children.len()
        );
        children[child_idx] = replacement;
        self.with_children(children)
    }

    // ArrayVisitor delegation methods

    /// Returns the children of the array.
    pub fn children(&self) -> Vec<ArrayRef> {
        self.0.children(self)
    }

    /// Returns the number of children of the array.
    pub fn nchildren(&self) -> usize {
        self.0.nchildren(self)
    }

    /// Returns the nth child of the array without allocating a Vec.
    pub fn nth_child(&self, idx: usize) -> Option<ArrayRef> {
        self.0.nth_child(self, idx)
    }

    /// Returns the names of the children of the array.
    pub fn children_names(&self) -> Vec<String> {
        self.0.children_names(self)
    }

    /// Returns the array's children with their names.
    pub fn named_children(&self) -> Vec<(String, ArrayRef)> {
        self.0.named_children(self)
    }

    /// Returns the data buffers of the array.
    pub fn buffers(&self) -> Vec<ByteBuffer> {
        self.0.buffers(self)
    }

    /// Returns the buffer handles of the array.
    pub fn buffer_handles(&self) -> Vec<BufferHandle> {
        self.0.buffer_handles(self)
    }

    /// Returns the names of the buffers of the array.
    pub fn buffer_names(&self) -> Vec<String> {
        self.0.buffer_names(self)
    }

    /// Returns the array's buffers with their names.
    pub fn named_buffers(&self) -> Vec<(String, BufferHandle)> {
        self.0.named_buffers(self)
    }

    /// Returns the number of data buffers of the array.
    pub fn nbuffers(&self) -> usize {
        self.0.nbuffers(self)
    }

    /// Returns the serialized metadata of the array.
    pub fn metadata(&self) -> VortexResult<Option<Vec<u8>>> {
        self.0.metadata(self)
    }

    /// Formats a human-readable metadata description.
    pub fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.metadata_fmt(self, f)
    }

    /// Returns whether all buffers are host-resident.
    pub fn is_host(&self) -> bool {
        self.0.is_host(self)
    }

    // ArrayVisitorExt delegation methods

    /// Count the number of buffers encoded by self and all child arrays.
    pub fn nbuffers_recursive(&self) -> usize {
        self.0.nbuffers_recursive()
    }

    /// Depth-first traversal of the array and its children.
    pub fn depth_first_traversal(&self) -> impl Iterator<Item = ArrayRef> {
        self.0.depth_first_traversal()
    }

    /// Returns a clone of this ArrayRef as an ArrayRef (for compatibility).
    pub fn to_array(&self) -> ArrayRef {
        self.clone()
    }
}

// Internal-only methods on dyn DynArray for use within the crate.
#[allow(dead_code)]
impl dyn DynArray + '_ {
    /// Does the array match the given matcher.
    pub(crate) fn is<M: Matcher>(&self) -> bool {
        M::matches(self)
    }

    /// Returns the array downcast by the given matcher.
    pub(crate) fn as_<M: Matcher>(&self) -> M::Match<'_> {
        self.as_opt::<M>().vortex_expect("Failed to downcast")
    }

    /// Returns the array downcast by the given matcher.
    pub(crate) fn as_opt<M: Matcher>(&self) -> Option<M::Match<'_>> {
        M::try_match(self)
    }

    /// Returns a reference to the typed `ArrayInner<V>` if this array matches the given vtable type.
    pub(crate) fn as_typed<V: VTable>(&self) -> Option<&ArrayInner<V>> {
        DynArray::as_any(self).downcast_ref::<ArrayInner<V>>()
    }
}

/// Trait for converting a type into a Vortex [`ArrayRef`].
pub trait IntoArray {
    fn into_array(self) -> ArrayRef;
}

impl IntoArray for ArrayRef {
    fn into_array(self) -> ArrayRef {
        self
    }
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for ArrayInner<V> {}
}

// =============================================================================
// New path: DynArray and supporting trait impls for ArrayInner<V>
// =============================================================================

/// DynArray implementation for [`ArrayInner<V>`].
///
/// This is self-contained: identity methods use `ArrayInner<V>`'s own fields (dtype, len, stats),
/// while data-access methods delegate to VTable methods on the inner `V::ArrayData`.
impl<V: VTable> DynArray for ArrayInner<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_array(&self) -> ArrayRef {
        self.to_array_ref()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn vtable(&self) -> &dyn DynVTable {
        &self.vtable
    }

    fn encoding_id(&self) -> ArrayId {
        self.vtable.id()
    }

    fn slice(&self, this: &ArrayRef, range: Range<usize>) -> VortexResult<ArrayRef> {
        let start = range.start;
        let stop = range.end;

        if start == 0 && stop == self.len {
            return Ok(this.clone());
        }

        vortex_ensure!(
            start <= self.len,
            "OutOfBounds: start {start} > length {}",
            self.len
        );
        vortex_ensure!(
            stop <= self.len,
            "OutOfBounds: stop {stop} > length {}",
            self.len
        );

        vortex_ensure!(start <= stop, "start ({start}) must be <= stop ({stop})");

        if start == stop {
            return Ok(Canonical::empty(&self.dtype).into_array());
        }

        let sliced = SliceArray::try_new(this.clone(), range)?
            .into_array()
            .optimize()?;

        // Propagate some stats from the original array to the sliced array.
        if !sliced.is::<Constant>() {
            self.statistics().with_iter(|iter| {
                sliced.statistics().inherit(iter.filter(|(stat, value)| {
                    matches!(
                        stat,
                        Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted
                    ) && value.as_ref().as_exact().is_some_and(|v| {
                        Scalar::try_new(DType::Bool(Nullability::NonNullable), Some(v.clone()))
                            .vortex_expect("A stat that was expected to be a boolean stat was not")
                            .as_bool()
                            .value()
                            .unwrap_or_default()
                    })
                }));
            });
        }

        Ok(sliced)
    }

    fn filter(&self, this: &ArrayRef, mask: Mask) -> VortexResult<ArrayRef> {
        FilterArray::try_new(this.clone(), mask)?
            .into_array()
            .optimize()
    }

    fn take(&self, this: &ArrayRef, indices: ArrayRef) -> VortexResult<ArrayRef> {
        DictArray::try_new(indices, this.clone())?
            .into_array()
            .optimize()
    }

    fn scalar_at(&self, this: &ArrayRef, index: usize) -> VortexResult<Scalar> {
        vortex_ensure!(index < self.len, OutOfBounds: index, 0, self.len);
        if DynArray::is_invalid(self, this, index)? {
            return Ok(Scalar::null(self.dtype.clone()));
        }
        let view = unsafe { ArrayView::new(this, &self.data) };
        let scalar = <V::OperationsVTable as OperationsVTable<V>>::scalar_at(
            view,
            index,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
        vortex_ensure!(&self.dtype == scalar.dtype(), "Scalar dtype mismatch");
        Ok(scalar)
    }

    fn is_valid(&self, this: &ArrayRef, index: usize) -> VortexResult<bool> {
        vortex_ensure!(index < self.len, OutOfBounds: index, 0, self.len);
        match DynArray::validity(self, this)? {
            Validity::NonNullable | Validity::AllValid => Ok(true),
            Validity::AllInvalid => Ok(false),
            Validity::Array(a) => a
                .scalar_at(index)?
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("validity value at index {} is null", index)),
        }
    }

    fn is_invalid(&self, this: &ArrayRef, index: usize) -> VortexResult<bool> {
        Ok(!DynArray::is_valid(self, this, index)?)
    }

    fn all_valid(&self, this: &ArrayRef) -> VortexResult<bool> {
        match DynArray::validity(self, this)? {
            Validity::NonNullable | Validity::AllValid => Ok(true),
            Validity::AllInvalid => Ok(false),
            Validity::Array(a) => Ok(a.statistics().compute_min::<bool>().unwrap_or(false)),
        }
    }

    fn all_invalid(&self, this: &ArrayRef) -> VortexResult<bool> {
        match DynArray::validity(self, this)? {
            Validity::NonNullable | Validity::AllValid => Ok(false),
            Validity::AllInvalid => Ok(true),
            Validity::Array(a) => Ok(!a.statistics().compute_max::<bool>().unwrap_or(true)),
        }
    }

    fn valid_count(&self, this: &ArrayRef) -> VortexResult<usize> {
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return Ok(self.len - invalid_count);
        }

        let count = match DynArray::validity(self, this)? {
            Validity::NonNullable | Validity::AllValid => self.len,
            Validity::AllInvalid => 0,
            Validity::Array(a) => {
                let mut ctx = LEGACY_SESSION.create_execution_ctx();
                let array_sum = sum(&a, &mut ctx)?;
                array_sum
                    .as_primitive()
                    .as_::<usize>()
                    .ok_or_else(|| vortex_err!("sum of validity array is null"))?
            }
        };
        vortex_ensure!(count <= self.len, "Valid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(self.len - count));

        Ok(count)
    }

    fn invalid_count(&self, this: &ArrayRef) -> VortexResult<usize> {
        Ok(self.len - DynArray::valid_count(self, this)?)
    }

    fn validity(&self, this: &ArrayRef) -> VortexResult<Validity> {
        if self.dtype.is_nullable() {
            let view = unsafe { ArrayView::new(this, &self.data) };
            let validity = <V::ValidityVTable as ValidityVTable<V>>::validity(view)?;
            if let Validity::Array(array) = &validity {
                vortex_ensure!(array.len() == self.len, "Validity array length mismatch");
                vortex_ensure!(
                    matches!(array.dtype(), DType::Bool(Nullability::NonNullable)),
                    "Validity array is not non-nullable boolean: {}",
                    self.vtable.id(),
                );
            }
            Ok(validity)
        } else {
            Ok(Validity::NonNullable)
        }
    }

    fn validity_mask(&self, this: &ArrayRef) -> VortexResult<Mask> {
        match DynArray::validity(self, this)? {
            Validity::NonNullable | Validity::AllValid => Ok(Mask::new_true(self.len)),
            Validity::AllInvalid => Ok(Mask::new_false(self.len)),
            Validity::Array(a) => Ok(a.to_bool().to_mask()),
        }
    }

    fn to_canonical(&self, this: &ArrayRef) -> VortexResult<Canonical> {
        this.clone()
            .execute(&mut LEGACY_SESSION.create_execution_ctx())
    }

    fn append_to_builder(
        &self,
        this: &ArrayRef,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if builder.dtype() != &self.dtype {
            vortex_panic!(
                "Builder dtype mismatch: expected {}, got {}",
                self.dtype,
                builder.dtype(),
            );
        }
        let len = builder.len();

        let view = unsafe { ArrayView::new(this, &self.data) };
        V::append_to_builder(view, builder, ctx)?;

        assert_eq!(
            len + self.len,
            builder.len(),
            "Builder length mismatch after writing array for encoding {}",
            self.vtable.id(),
        );
        Ok(())
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        self.stats.to_ref(self)
    }

    fn with_children(&self, _this: &ArrayRef, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        let mut inner = self.data.clone();
        V::with_children(&mut inner, children)?;
        // SAFETY: with_children preserves dtype and len.
        Ok(ArrayRef::from_inner(Arc::new(unsafe {
            ArrayInner::from_data_unchecked(
                self.vtable.clone(),
                self.dtype.clone(),
                self.len,
                inner,
                self.stats.clone(),
            )
        })))
    }

    fn children(&self, this: &ArrayRef) -> Vec<ArrayRef> {
        let view = unsafe { ArrayView::new(this, &self.data) };
        (0..V::nchildren(view)).map(|i| V::child(view, i)).collect()
    }

    fn nchildren(&self, this: &ArrayRef) -> usize {
        let view = unsafe { ArrayView::new(this, &self.data) };
        V::nchildren(view)
    }

    fn nth_child(&self, this: &ArrayRef, idx: usize) -> Option<ArrayRef> {
        let view = unsafe { ArrayView::new(this, &self.data) };
        (idx < V::nchildren(view)).then(|| V::child(view, idx))
    }

    fn children_names(&self, this: &ArrayRef) -> Vec<String> {
        let view = unsafe { ArrayView::new(this, &self.data) };
        (0..V::nchildren(view))
            .map(|i| V::child_name(view, i))
            .collect()
    }

    fn named_children(&self, this: &ArrayRef) -> Vec<(String, ArrayRef)> {
        let view = unsafe { ArrayView::new(this, &self.data) };
        (0..V::nchildren(view))
            .map(|i| (V::child_name(view, i), V::child(view, i)))
            .collect()
    }

    fn buffers(&self, this: &ArrayRef) -> Vec<ByteBuffer> {
        let view = unsafe { ArrayView::new(this, &self.data) };
        (0..V::nbuffers(view))
            .map(|i| V::buffer(view, i).to_host_sync())
            .collect()
    }

    fn buffer_handles(&self, this: &ArrayRef) -> Vec<BufferHandle> {
        let view = unsafe { ArrayView::new(this, &self.data) };
        (0..V::nbuffers(view)).map(|i| V::buffer(view, i)).collect()
    }

    fn buffer_names(&self, this: &ArrayRef) -> Vec<String> {
        let view = unsafe { ArrayView::new(this, &self.data) };
        (0..V::nbuffers(view))
            .filter_map(|i| V::buffer_name(view, i))
            .collect()
    }

    fn named_buffers(&self, this: &ArrayRef) -> Vec<(String, BufferHandle)> {
        let view = unsafe { ArrayView::new(this, &self.data) };
        (0..V::nbuffers(view))
            .filter_map(|i| V::buffer_name(view, i).map(|name| (name, V::buffer(view, i))))
            .collect()
    }

    fn nbuffers(&self, this: &ArrayRef) -> usize {
        let view = unsafe { ArrayView::new(this, &self.data) };
        V::nbuffers(view)
    }

    fn metadata(&self, this: &ArrayRef) -> VortexResult<Option<Vec<u8>>> {
        let view = unsafe { ArrayView::new(this, &self.data) };
        V::serialize(V::metadata(view)?)
    }

    fn metadata_fmt(&self, this: &ArrayRef, f: &mut Formatter<'_>) -> std::fmt::Result {
        let view = unsafe { ArrayView::new(this, &self.data) };
        match V::metadata(view) {
            Err(e) => write!(f, "<serde error: {e}>"),
            Ok(metadata) => Debug::fmt(&metadata, f),
        }
    }

    fn is_host(&self, _this: &ArrayRef) -> bool {
        for array in self.depth_first_traversal() {
            if !array.buffer_handles().iter().all(BufferHandle::is_on_host) {
                return false;
            }
        }
        true
    }
}

impl<V: VTable> ArrayHash for ArrayInner<V> {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: hash::Precision) {
        self.vtable.id().hash(state);
        self.with_view(|view| V::array_hash(view, state, precision));
    }
}

impl<V: VTable> ArrayEq for ArrayInner<V> {
    fn array_eq(&self, other: &Self, precision: hash::Precision) -> bool {
        self.with_view(|self_view| {
            other.with_view(|other_view| V::array_eq(self_view, other_view, precision))
        })
    }
}

impl<V: VTable> Matcher for V {
    type Match<'a> = &'a ArrayInner<V>;

    fn matches(array: &dyn DynArray) -> bool {
        DynArray::as_any(array).is::<ArrayInner<V>>()
    }

    fn try_match(array: &dyn DynArray) -> Option<&ArrayInner<V>> {
        DynArray::as_any(array).downcast_ref::<ArrayInner<V>>()
    }
}
