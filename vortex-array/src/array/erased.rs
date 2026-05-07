// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
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
use crate::Array;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayView;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::aggregate_fn::fns::sum::sum;
use crate::array::ArrayData;
use crate::array::ArrayId;
use crate::array::ArrayInner;
use crate::array::ArraySlots;
use crate::array::DynArrayData;
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
///
/// Wraps `Arc<ArrayInner<dyn DynArrayData>>` — a single 16-byte fat pointer.
/// Metadata (`len`, `dtype`, `encoding_id`) lives in `ArrayInner::meta` and is
/// accessed as a normal struct field read — no vtable dispatch, no extra allocation.
#[derive(Clone)]
pub struct ArrayRef(Arc<ArrayInner<dyn DynArrayData>>);

impl ArrayRef {
    /// Create from an `Arc<ArrayInner<dyn DynArrayData>>`.
    pub(crate) fn from_inner<D: DynArrayData>(inner: Arc<ArrayInner<D>>) -> Self {
        Self(inner)
    }

    /// Returns a reference to the `dyn DynArrayData` inside the inner.
    #[inline(always)]
    pub(crate) fn dyn_array(&self) -> &dyn DynArrayData {
        &self.0.data
    }

    /// Returns a mutable reference to the inner if this is the sole owner.
    #[inline(always)]
    pub(crate) fn inner_mut(&mut self) -> Option<&mut ArrayInner<dyn DynArrayData>> {
        Arc::get_mut(&mut self.0)
    }

    /// Returns the Arc::as_ptr().addr() of the underlying array.
    /// This function is used in a couple of places, and we should migrate them to using array_eq.
    #[doc(hidden)]
    pub fn addr(&self) -> usize {
        Arc::as_ptr(&self.0).addr()
    }

    /// Downcast the inner to a concrete `ArrayInner<ArrayData<V>>`.
    ///
    /// Uses the same raw-pointer technique as `Arc::downcast`.
    #[allow(dead_code)]
    pub(crate) fn downcast_inner<V: VTable>(self) -> Result<Arc<ArrayInner<ArrayData<V>>>, Self> {
        // TODO(joe): can we use encoding id here?
        if self.0.data.as_any().is::<ArrayData<V>>() {
            Ok(unsafe { self.downcast_inner_unchecked() })
        } else {
            Err(self)
        }
    }

    /// Downcast without a runtime type check.
    ///
    /// # Safety
    /// The caller must guarantee the concrete type behind `dyn DynArrayData` is `ArrayData<V>`.
    #[inline(always)]
    pub(crate) unsafe fn downcast_inner_unchecked<V: VTable>(
        self,
    ) -> Arc<ArrayInner<ArrayData<V>>> {
        debug_assert!(self.0.data.as_any().is::<ArrayData<V>>());
        // Recover the original concrete Arc. The fat pointer's data pointer is the
        // same allocation that was originally `Arc<ArrayInner<ArrayData<V>>>` before
        // unsized coercion to `Arc<ArrayInner<dyn DynArrayData>>`.
        let raw = Arc::into_raw(self.0);
        // # Safety all arrays are constructed in this way and type aliased.
        unsafe { Arc::from_raw(raw.cast::<ArrayInner<ArrayData<V>>>()) }
    }

    /// Returns true if the two ArrayRefs point to the same allocation.
    pub fn ptr_eq(this: &ArrayRef, other: &ArrayRef) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }
}

impl Debug for ArrayRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("encoding", &self.0.encoding_id)
            .field("dtype", &self.0.dtype)
            .field("len", &self.0.len)
            .field("data", &self.0.data)
            .finish()
    }
}

impl ArrayHash for ArrayRef {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: crate::Precision) {
        self.0.len.hash(state);
        self.0.dtype.hash(state);
        self.0.encoding_id.hash(state);
        self.0.slots.len().hash(state);
        for slot in &self.0.slots {
            slot.array_hash(state, precision);
        }
        self.0
            .data
            .dyn_array_hash(state as &mut dyn Hasher, precision);
    }
}

impl ArrayEq for ArrayRef {
    fn array_eq(&self, other: &Self, precision: crate::Precision) -> bool {
        self.0.len == other.0.len
            && self.0.dtype == other.0.dtype
            && self.0.encoding_id == other.0.encoding_id
            && self.0.slots.len() == other.0.slots.len()
            && self
                .0
                .slots
                .iter()
                .zip(other.0.slots.iter())
                .all(|(slot, other_slot)| slot.array_eq(other_slot, precision))
            && self.0.data.dyn_array_eq(other, precision)
    }
}
impl ArrayRef {
    /// Returns the length of the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len
    }

    /// Returns whether the array is empty (has zero rows).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// Returns the logical Vortex [`DType`] of the array.
    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.0.dtype
    }

    /// Returns the encoding ID of the array.
    #[inline]
    pub fn encoding_id(&self) -> ArrayId {
        self.0.encoding_id
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
    #[deprecated(
        note = "Use `execute_scalar` instead, which allows passing an execution context for more \
        efficient execution when fetching multiple scalars from the same array."
    )]
    pub fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        self.execute_scalar(index, &mut LEGACY_SESSION.create_execution_ctx())
    }

    /// Execute the array to extract a scalar at the given index.
    pub fn execute_scalar(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        vortex_ensure!(index < self.len(), OutOfBounds: index, 0, self.len());
        if self.dtype().is_nullable() && self.is_invalid(index, ctx)? {
            return Ok(Scalar::null(self.dtype().clone()));
        }
        let scalar = self.0.data.execute_scalar(self, index, ctx)?;
        debug_assert_eq!(self.dtype(), scalar.dtype(), "Scalar dtype mismatch");
        Ok(scalar)
    }

    /// Returns whether the item at `index` is valid.
    pub fn is_valid(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        vortex_ensure!(index < self.len(), OutOfBounds: index, 0, self.len());
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(true),
            Validity::AllInvalid => Ok(false),
            Validity::Array(a) => a
                .execute_scalar(index, ctx)?
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("validity value at index {} is null", index)),
        }
    }

    /// Returns whether the item at `index` is invalid.
    pub fn is_invalid(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        Ok(!self.is_valid(index, ctx)?)
    }

    /// Returns whether all items in the array are valid.
    pub fn all_valid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(true),
            Validity::AllInvalid => Ok(false),
            Validity::Array(a) => Ok(a.statistics().compute_min::<bool>(ctx).unwrap_or(false)),
        }
    }

    /// Returns whether the array is all invalid.
    pub fn all_invalid(&self, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(false),
            Validity::AllInvalid => Ok(true),
            Validity::Array(a) => Ok(!a.statistics().compute_max::<bool>(ctx).unwrap_or(true)),
        }
    }

    /// Returns the number of valid elements in the array.
    pub fn valid_count(&self, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
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
                let array_sum = sum(&a, ctx)?;
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
    pub fn invalid_count(&self, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        Ok(self.len() - self.valid_count(ctx)?)
    }

    /// Returns the [`Validity`] of the array.
    pub fn validity(&self) -> VortexResult<Validity> {
        self.0.data.validity(self)
    }

    /// Returns the canonical representation of the array.
    #[deprecated(note = "use `array.execute::<Canonical>(ctx)` instead")]
    pub fn into_canonical(self) -> VortexResult<Canonical> {
        self.execute(&mut LEGACY_SESSION.create_execution_ctx())
    }

    /// Returns the canonical representation of the array.
    #[deprecated(note = "use `array.execute::<Canonical>(ctx)` instead")]
    pub fn to_canonical(&self) -> VortexResult<Canonical> {
        #[expect(deprecated)]
        let result = self.clone().into_canonical();
        result
    }

    /// Writes the array into the canonical builder.
    pub fn append_to_builder(
        &self,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.0.data.append_to_builder(self, builder, ctx)
    }

    /// Returns the statistics of the array.
    pub fn statistics(&self) -> StatsSetRef<'_> {
        self.0.stats.to_ref(self)
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
    pub fn try_downcast<V: VTable>(self) -> Result<Array<V>, ArrayRef> {
        Array::<V>::try_from_array_ref(self)
    }

    /// Returns the array downcast to the given `Array<V>` as an owned typed handle.
    ///
    /// # Panics
    ///
    /// Panics if the array is not of the given type.
    pub fn downcast<V: VTable>(self) -> Array<V> {
        Self::try_downcast(self)
            .unwrap_or_else(|_| vortex_panic!("Failed to downcast to {}", type_name::<V>()))
    }

    /// Returns a reference to the typed `ArrayData<V>` if this array matches the given vtable type.
    pub fn as_typed<V: VTable>(&self) -> Option<ArrayView<'_, V>> {
        let inner = self.0.data.as_any().downcast_ref::<ArrayData<V>>()?;
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
    /// This is only valid for physical rewrites: the replacement must have the same logical
    /// `DType` and `len` as the existing slot.
    ///
    /// Takes ownership to allow in-place mutation when the refcount is 1.
    pub fn with_slot(self, slot_idx: usize, replacement: ArrayRef) -> VortexResult<ArrayRef> {
        let mut slots: ArraySlots = self.slots().iter().cloned().collect();
        let nslots = slots.len();
        vortex_ensure!(
            slot_idx < nslots,
            "slot index {} out of bounds for array with {} slots",
            slot_idx,
            nslots
        );
        let existing = slots[slot_idx]
            .as_ref()
            .vortex_expect("with_slot cannot replace an absent slot");
        vortex_ensure!(
            existing.dtype() == replacement.dtype(),
            "slot {} dtype changed from {} to {} during physical rewrite",
            slot_idx,
            existing.dtype(),
            replacement.dtype()
        );
        vortex_ensure!(
            existing.len() == replacement.len(),
            "slot {} len changed from {} to {} during physical rewrite",
            slot_idx,
            existing.len(),
            replacement.len()
        );
        slots[slot_idx] = Some(replacement);
        self.with_slots(slots)
    }

    /// Take a slot for executor-owned physical rewrites.
    ///
    /// On return the produced parent has the taken slot set to `None`
    /// callers must put the slot back (typically via [`put_slot_unchecked`]) before the parent is
    /// returned from the execution loop.
    ///
    /// When the `Arc` was shared this allocates a fresh parent.
    ///
    /// # Safety
    /// The caller must put back a slot with the same logical dtype and length before exposing the
    /// parent array, and must only use this for physical rewrites.
    pub(crate) unsafe fn take_slot_unchecked(
        mut self,
        slot_idx: usize,
    ) -> VortexResult<(ArrayRef, ArrayRef)> {
        if let Some(inner) = Arc::get_mut(&mut self.0) {
            let child = inner.slots[slot_idx]
                .take()
                .vortex_expect("take_slot_unchecked cannot take an absent slot");
            return Ok((self, child));
        }

        // Arc is shared: clone the child out and build a fresh parent with slot_idx = None,
        // bypassing encoding-level validation so the absent slot does not panic `V::validate`.
        let child = self.slots()[slot_idx]
            .as_ref()
            .vortex_expect("take_slot_unchecked cannot take an absent slot")
            .clone();

        let mut new_slots: ArraySlots = self.slots().iter().cloned().collect();
        new_slots[slot_idx] = None;

        // SAFETY: ensured by the caller — the None slot is either put back or driven to completion
        // via the builder path before the parent escapes the executor.
        let new_parent = unsafe { self.0.data.with_slots_unchecked(&self, new_slots) };
        Ok((new_parent, child))
    }

    /// Puts an array into `slot_idx` by either, cloning the inner array if the Arc is not exclusive
    /// or replacing the slot in this `ArrayRef`.
    /// This is the mirror of [`take_slot_unchecked`].
    ///
    /// # Safety
    /// The replacement must have the same logical dtype and length as the taken slot, and this
    /// must only be used for physical rewrites.
    pub(crate) unsafe fn put_slot_unchecked(
        mut self,
        slot_idx: usize,
        replacement: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        if let Some(inner) = Arc::get_mut(&mut self.0) {
            inner.slots[slot_idx] = Some(replacement);
            return Ok(self);
        }

        let mut slots: ArraySlots = self.slots().iter().cloned().collect();
        slots[slot_idx] = Some(replacement);
        self.0.data.with_slots(&self, slots)
    }

    /// Returns a new array with the provided slots.
    ///
    /// This is only valid for physical rewrites: slot count, presence, logical `DType`, and
    /// logical `len` must remain unchanged.
    pub fn with_slots(self, slots: impl Into<ArraySlots>) -> VortexResult<ArrayRef> {
        let slots = slots.into();
        let old_slots = self.slots();
        vortex_ensure!(
            old_slots.len() == slots.len(),
            "slot count changed from {} to {} during physical rewrite",
            old_slots.len(),
            slots.len()
        );
        for (idx, (old_slot, new_slot)) in old_slots.iter().zip(slots.iter()).enumerate() {
            vortex_ensure!(
                old_slot.is_some() == new_slot.is_some(),
                "slot {} presence changed during physical rewrite",
                idx
            );
            if let (Some(old_slot), Some(new_slot)) = (old_slot.as_ref(), new_slot.as_ref()) {
                vortex_ensure!(
                    old_slot.dtype() == new_slot.dtype(),
                    "slot {} dtype changed from {} to {} during physical rewrite",
                    idx,
                    old_slot.dtype(),
                    new_slot.dtype()
                );
                vortex_ensure!(
                    old_slot.len() == new_slot.len(),
                    "slot {} len changed from {} to {} during physical rewrite",
                    idx,
                    old_slot.len(),
                    new_slot.len()
                );
            }
        }
        self.0.data.with_slots(&self, slots)
    }

    pub fn reduce(&self) -> VortexResult<Option<ArrayRef>> {
        self.0.data.reduce(self)
    }

    pub fn reduce_parent(
        &self,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        self.0.data.reduce_parent(self, parent, child_idx)
    }

    pub(crate) fn execute_encoding(self, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let inner = Arc::as_ptr(&self.0);
        // SAFETY: the Arc outlives the DynArrayData function call
        unsafe { (&*inner).data.execute(self, ctx) }
    }

    /// Execute a single encoding step without applying `Done`-result postconditions.
    ///
    /// This is for the iterative executor only. It may operate on suspended executor-private
    /// arrays whose slots temporarily contain `None`, so the executor itself must interpret
    /// `Done`, enforce any `len`/`dtype` invariants, and transfer statistics.
    pub(crate) fn execute_encoding_unchecked(
        self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ExecutionResult> {
        let inner = Arc::as_ptr(&self.0);
        // SAFETY: `inner` points at the allocation owned by `self.0`. `self` stays alive for the
        // duration of the call, so the pointee remains valid. Avoiding an extra `Arc` clone here
        // preserves uniqueness so execute-time metadata cursors can use `Arc::get_mut`.
        unsafe { (&*inner).data.execute_unchecked(self, ctx) }
    }

    pub fn execute_parent(
        &self,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        self.0.data.execute_parent(self, parent, child_idx, ctx)
    }

    // ArrayVisitor delegation methods

    /// Returns the children of the array.
    pub fn children(&self) -> Vec<ArrayRef> {
        self.0.data.children(self)
    }

    /// Returns the number of children of the array.
    pub fn nchildren(&self) -> usize {
        self.0.data.nchildren(self)
    }

    /// Returns the nth child of the array without allocating a Vec.
    pub fn nth_child(&self, idx: usize) -> Option<ArrayRef> {
        self.0.data.nth_child(self, idx)
    }

    /// Returns the names of the children of the array.
    pub fn children_names(&self) -> Vec<String> {
        self.0.data.children_names(self)
    }

    /// Returns the array's children with their names.
    pub fn named_children(&self) -> Vec<(String, ArrayRef)> {
        self.0.data.named_children(self)
    }

    /// Returns the data buffers of the array.
    pub fn buffers(&self) -> Vec<ByteBuffer> {
        self.0.data.buffers(self)
    }

    /// Returns the buffer handles of the array.
    pub fn buffer_handles(&self) -> Vec<BufferHandle> {
        self.0.data.buffer_handles(self)
    }

    /// Returns the names of the buffers of the array.
    pub fn buffer_names(&self) -> Vec<String> {
        self.0.data.buffer_names(self)
    }

    /// Returns the array's buffers with their names.
    pub fn named_buffers(&self) -> Vec<(String, BufferHandle)> {
        self.0.data.named_buffers(self)
    }

    /// Returns the number of data buffers of the array.
    pub fn nbuffers(&self) -> usize {
        self.0.data.nbuffers(self)
    }

    /// Returns the slots of the array.
    pub fn slots(&self) -> &[Option<ArrayRef>] {
        &self.0.slots
    }

    /// Returns the name of the slot at the given index.
    pub fn slot_name(&self, idx: usize) -> String {
        self.0.data.slot_name(self, idx)
    }

    /// Formats a human-readable metadata description.
    pub fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.data.metadata_fmt(f)
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
        array.0.data.as_any().is::<ArrayData<V>>()
    }

    fn try_match(array: &'_ ArrayRef) -> Option<ArrayView<'_, V>> {
        let inner = array.0.data.as_any().downcast_ref::<ArrayData<V>>()?;
        // # Safety checked by `downcast_ref`.
        Some(unsafe { ArrayView::new_unchecked(array, &inner.data) })
    }
}
