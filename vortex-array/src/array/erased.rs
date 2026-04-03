// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::AnyCanonical;
use crate::Array;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayView;
use crate::Canonical;
use crate::DynVTable;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::ToCanonical;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::aggregate_fn::fns::sum::sum;
use crate::array::ArrayId;
use crate::array::ArrayInner;
use crate::array::DynArray;
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
use crate::matcher::Matcher;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::stats::StatsSetRef;
use crate::validity::Validity;

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
    pub(crate) fn from_inner(inner: Arc<dyn DynArray>) -> Self {
        Self(inner)
    }

    /// Returns the Arc::as_ptr().addr() of the underlying array.
    /// This function is used in a couple of places, and we should migrate them to using array_eq.
    #[doc(hidden)]
    pub fn addr(&self) -> usize {
        Arc::as_ptr(&self.0).addr()
    }

    /// Returns a reference to the inner Arc.
    #[inline(always)]
    pub(crate) fn inner(&self) -> &Arc<dyn DynArray> {
        &self.0
    }

    /// Returns true if the two ArrayRefs point to the same allocation.
    pub fn ptr_eq(this: &ArrayRef, other: &ArrayRef) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }
}

impl Debug for ArrayRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&*self.0, f)
    }
}

impl ArrayHash for ArrayRef {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: crate::Precision) {
        self.0.dyn_array_hash(state as &mut dyn Hasher, precision);
    }
}

impl ArrayEq for ArrayRef {
    fn array_eq(&self, other: &Self, precision: crate::Precision) -> bool {
        self.0.dyn_array_eq(other.0.as_any(), precision)
    }
}

#[allow(clippy::same_name_method)]
impl ArrayRef {
    /// Returns the length of the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the array is empty (has zero rows).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.len() == 0
    }

    /// Returns the logical Vortex [`DType`] of the array.
    #[inline]
    pub fn dtype(&self) -> &DType {
        self.0.dtype()
    }

    /// Returns the vtable of the array.
    #[inline]
    pub fn vtable(&self) -> &dyn DynVTable {
        self.0.vtable()
    }

    /// Returns the encoding ID of the array.
    #[inline]
    pub fn encoding_id(&self) -> ArrayId {
        self.0.encoding_id()
    }

    /// Performs a constant-time slice of the array.
    pub fn slice(&self, range: Range<usize>) -> VortexResult<ArrayRef> {
        let len = self.len();
        let start = range.start;
        let stop = range.end;

        if start == 0 && stop == len {
            return Ok(self.clone());
        }

        vortex_ensure!(start <= len, "OutOfBounds: start {start} > length {}", len);
        vortex_ensure!(stop <= len, "OutOfBounds: stop {stop} > length {}", len);

        vortex_ensure!(start <= stop, "start ({start}) must be <= stop ({stop})");

        if start == stop {
            return Ok(Canonical::empty(self.dtype()).into_array());
        }

        let sliced = SliceArray::try_new(self.clone(), range)?
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

    /// Wraps the array in a [`FilterArray`] such that it is logically filtered by the given mask.
    pub fn filter(&self, mask: Mask) -> VortexResult<ArrayRef> {
        FilterArray::try_new(self.clone(), mask)?
            .into_array()
            .optimize()
    }

    /// Wraps the array in a [`DictArray`] such that it is logically taken by the given indices.
    pub fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        DictArray::try_new(indices, self.clone())?
            .into_array()
            .optimize()
    }

    /// Fetch the scalar at the given index.
    pub fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        vortex_ensure!(index < self.len(), OutOfBounds: index, 0, self.len());
        if self.is_invalid(index)? {
            return Ok(Scalar::null(self.dtype().clone()));
        }
        let scalar = self.0.scalar_at(self, index)?;
        vortex_ensure!(self.dtype() == scalar.dtype(), "Scalar dtype mismatch");
        Ok(scalar)
    }

    /// Returns whether the item at `index` is valid.
    pub fn is_valid(&self, index: usize) -> VortexResult<bool> {
        vortex_ensure!(index < self.len(), OutOfBounds: index, 0, self.len());
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(true),
            Validity::AllInvalid => Ok(false),
            Validity::Array(a) => a
                .scalar_at(index)?
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("validity value at index {} is null", index)),
        }
    }

    /// Returns whether the item at `index` is invalid.
    pub fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        Ok(!self.is_valid(index)?)
    }

    /// Returns whether all items in the array are valid.
    pub fn all_valid(&self) -> VortexResult<bool> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(true),
            Validity::AllInvalid => Ok(false),
            Validity::Array(a) => Ok(a.statistics().compute_min::<bool>().unwrap_or(false)),
        }
    }

    /// Returns whether the array is all invalid.
    pub fn all_invalid(&self) -> VortexResult<bool> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(false),
            Validity::AllInvalid => Ok(true),
            Validity::Array(a) => Ok(!a.statistics().compute_max::<bool>().unwrap_or(true)),
        }
    }

    /// Returns the number of valid elements in the array.
    pub fn valid_count(&self) -> VortexResult<usize> {
        let len = self.len();
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return Ok(len - invalid_count);
        }

        let count = match self.validity()? {
            Validity::NonNullable | Validity::AllValid => len,
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
        vortex_ensure!(count <= len, "Valid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(len - count));

        Ok(count)
    }

    /// Returns the number of invalid elements in the array.
    pub fn invalid_count(&self) -> VortexResult<usize> {
        Ok(self.len() - self.valid_count()?)
    }

    /// Returns the [`Validity`] of the array.
    pub fn validity(&self) -> VortexResult<Validity> {
        self.0.validity(self)
    }

    /// Returns the canonical validity mask for the array.
    pub fn validity_mask(&self) -> VortexResult<Mask> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(Mask::new_true(self.len())),
            Validity::AllInvalid => Ok(Mask::new_false(self.len())),
            Validity::Array(a) => Ok(a.to_bool().to_mask()),
        }
    }

    /// Returns the canonical representation of the array.
    pub fn into_canonical(self) -> VortexResult<Canonical> {
        self.execute(&mut LEGACY_SESSION.create_execution_ctx())
    }

    /// Returns the canonical representation of the array.
    pub fn to_canonical(&self) -> VortexResult<Canonical> {
        self.clone().into_canonical()
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
        self.0.statistics().to_ref(self)
    }

    /// Does the array match the given matcher.
    pub fn is<M: Matcher>(&self) -> bool {
        M::matches(self)
    }

    /// Returns the array downcast by the given matcher.
    pub fn as_<M: Matcher>(&self) -> M::Match<'_> {
        self.as_opt::<M>().vortex_expect("Failed to downcast")
    }

    /// Returns the array downcast by the given matcher.
    pub fn as_opt<M: Matcher>(&self) -> Option<M::Match<'_>> {
        M::try_match(self)
    }

    /// Returns the array downcast to the given `Array<V>` as an owned typed handle.
    pub fn try_into<V: VTable>(self) -> Result<Array<V>, ArrayRef> {
        Array::<V>::try_from_array_ref(self)
    }

    /// Returns a reference to the typed `ArrayInner<V>` if this array matches the given vtable type.
    pub fn as_typed<V: VTable>(&self) -> Option<ArrayView<'_, V>> {
        let inner = self.0.as_any().downcast_ref::<ArrayInner<V>>()?;
        Some(unsafe { ArrayView::new_unchecked(self, &inner.data) })
    }

    /// Returns the constant scalar if this is a constant array.
    pub fn as_constant(&self) -> Option<Scalar> {
        self.as_opt::<Constant>().map(|a| a.scalar().clone())
    }

    /// Total size of the array in bytes, including all children and buffers.
    pub fn nbytes(&self) -> u64 {
        let mut nbytes = 0;
        for array in self.depth_first_traversal() {
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

    /// Returns a new array with the slot at `slot_idx` replaced by `replacement`.
    ///
    /// Takes ownership to allow in-place mutation when the refcount is 1.
    pub fn with_slot(self, slot_idx: usize, replacement: ArrayRef) -> VortexResult<ArrayRef> {
        let nslots = self.slots().len();
        vortex_ensure!(
            slot_idx < nslots,
            "slot index {} out of bounds for array with {} slots",
            slot_idx,
            nslots
        );
        let mut slots = self.slots().to_vec();
        slots[slot_idx] = Some(replacement);
        let vtable = self.vtable().clone_boxed();
        vtable.with_slots(self, slots)
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

    /// Returns the slots of the array.
    pub fn slots(&self) -> Vec<Option<ArrayRef>> {
        self.0.slots(self)
    }

    /// Returns the name of the slot at the given index.
    pub fn slot_name(&self, idx: usize) -> String {
        self.0.slot_name(self, idx)
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
        for array in self.depth_first_traversal() {
            if !array.buffer_handles().iter().all(BufferHandle::is_on_host) {
                return false;
            }
        }
        true
    }

    // ArrayVisitorExt delegation methods

    /// Count the number of buffers encoded by self and all child arrays.
    pub fn nbuffers_recursive(&self) -> usize {
        self.children()
            .iter()
            .map(|c| c.nbuffers_recursive())
            .sum::<usize>()
            + self.nbuffers()
    }

    /// Depth-first traversal of the array and its children.
    pub fn depth_first_traversal(&self) -> DepthFirstArrayIterator {
        DepthFirstArrayIterator {
            stack: vec![self.clone()],
        }
    }
}

impl IntoArray for ArrayRef {
    #[inline(always)]
    fn into_array(self) -> ArrayRef {
        self
    }
}

impl<V: VTable> Matcher for V {
    type Match<'a> = ArrayView<'a, V>;

    fn matches(array: &ArrayRef) -> bool {
        array.0.as_any().is::<ArrayInner<V>>()
    }

    fn try_match<'a>(array: &'a ArrayRef) -> Option<ArrayView<'a, V>> {
        let data = &array.0.as_any().downcast_ref::<ArrayInner<V>>()?.data;
        Some(unsafe { ArrayView::new_unchecked(array, data) })
    }
}
