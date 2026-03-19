// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod visitor;

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;
use std::ops::Range;
use std::sync::Arc;

pub use visitor::*;
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
use crate::arrays::ScalarFnVTable;
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
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnRef;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTableExt;
use crate::vtable::DynVTable;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;

/// The public API trait for all Vortex arrays.
pub trait DynArray:
    'static
    + private::Sealed
    + Send
    + Sync
    + Debug
    + DynArrayEq
    + DynArrayHash
    + ArrayVisitor
    + ReduceNode
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
    fn slice(&self, range: Range<usize>) -> VortexResult<ArrayRef>;

    /// Wraps the array in a [`FilterArray`] such that it is logically filtered by the given mask.
    fn filter(&self, mask: Mask) -> VortexResult<ArrayRef>;

    /// Wraps the array in a [`DictArray`] such that it is logically taken by the given indices.
    fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef>;

    /// Fetch the scalar at the given index.
    ///
    /// This method panics if the index is out of bounds for the array.
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar>;

    /// Returns whether the item at `index` is valid.
    fn is_valid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether the item at `index` is invalid.
    fn is_invalid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether all items in the array are valid.
    ///
    /// This is usually cheaper than computing a precise `valid_count`, but may return false
    /// negatives.
    fn all_valid(&self) -> VortexResult<bool>;

    /// Returns whether the array is all invalid.
    ///
    /// This is usually cheaper than computing a precise `invalid_count`, but may return false
    /// negatives.
    fn all_invalid(&self) -> VortexResult<bool>;

    /// Returns the number of valid elements in the array.
    fn valid_count(&self) -> VortexResult<usize>;

    /// Returns the number of invalid elements in the array.
    fn invalid_count(&self) -> VortexResult<usize>;

    /// Returns the [`Validity`] of the array.
    fn validity(&self) -> VortexResult<Validity>;

    /// Returns the canonical validity mask for the array.
    fn validity_mask(&self) -> VortexResult<Mask>;

    /// Returns the canonical representation of the array.
    fn to_canonical(&self) -> VortexResult<Canonical>;

    /// Writes the array into the canonical builder.
    ///
    /// The [`DType`] of the builder must match that of the array.
    fn append_to_builder(
        &self,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>;

    /// Returns the statistics of the array.
    // TODO(ngates): change how this works. It's weird.
    fn statistics(&self) -> StatsSetRef<'_>;

    /// Replaces the children of the array with the given array references.
    fn with_children(&self, children: Vec<ArrayRef>) -> VortexResult<ArrayRef>;
}

impl DynArray for Arc<dyn DynArray> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        DynArray::as_any(self.as_ref())
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    #[inline]
    fn to_array(&self) -> ArrayRef {
        self.clone()
    }

    #[inline]
    fn len(&self) -> usize {
        self.as_ref().len()
    }

    #[inline]
    fn dtype(&self) -> &DType {
        self.as_ref().dtype()
    }

    fn vtable(&self) -> &dyn DynVTable {
        self.as_ref().vtable()
    }

    #[inline]
    fn encoding_id(&self) -> ArrayId {
        self.as_ref().encoding_id()
    }

    #[inline]
    fn slice(&self, range: Range<usize>) -> VortexResult<ArrayRef> {
        self.as_ref().slice(range)
    }

    fn filter(&self, mask: Mask) -> VortexResult<ArrayRef> {
        self.as_ref().filter(mask)
    }

    fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        self.as_ref().take(indices)
    }

    #[inline]
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        self.as_ref().scalar_at(index)
    }

    #[inline]
    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        self.as_ref().is_valid(index)
    }

    #[inline]
    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        self.as_ref().is_invalid(index)
    }

    #[inline]
    fn all_valid(&self) -> VortexResult<bool> {
        self.as_ref().all_valid()
    }

    #[inline]
    fn all_invalid(&self) -> VortexResult<bool> {
        self.as_ref().all_invalid()
    }

    #[inline]
    fn valid_count(&self) -> VortexResult<usize> {
        self.as_ref().valid_count()
    }

    #[inline]
    fn invalid_count(&self) -> VortexResult<usize> {
        self.as_ref().invalid_count()
    }

    #[inline]
    fn validity(&self) -> VortexResult<Validity> {
        self.as_ref().validity()
    }

    #[inline]
    fn validity_mask(&self) -> VortexResult<Mask> {
        self.as_ref().validity_mask()
    }

    fn to_canonical(&self) -> VortexResult<Canonical> {
        self.as_ref().to_canonical()
    }

    fn append_to_builder(
        &self,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.as_ref().append_to_builder(builder, ctx)
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        self.as_ref().statistics()
    }

    fn with_children(&self, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        self.as_ref().with_children(children)
    }
}

/// A reference counted pointer to a dynamic [`DynArray`] trait object.
pub type ArrayRef = Arc<dyn DynArray>;

impl ToOwned for dyn DynArray {
    type Owned = ArrayRef;

    fn to_owned(&self) -> Self::Owned {
        self.to_array()
    }
}

impl dyn DynArray + '_ {
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

    /// Returns the array downcast to the given `A` as an owned object.
    pub fn try_into<V: VTable>(self: Arc<Self>) -> Result<V::Array, Arc<Self>> {
        match self.is::<V>() {
            true => {
                let arc = self
                    .as_any_arc()
                    .downcast::<ArrayAdapter<V>>()
                    .map_err(|_| vortex_err!("failed to downcast"))
                    .vortex_expect("Failed to downcast");
                Ok(match Arc::try_unwrap(arc) {
                    Ok(array) => array.0,
                    Err(arc) => arc.deref().0.clone(),
                })
            }
            false => Err(self),
        }
    }

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

    impl<V: VTable> Sealed for ArrayAdapter<V> {}
    impl Sealed for Arc<dyn DynArray> {}
}

/// Adapter struct used to lift the [`VTable`] trait into an object-safe [`DynArray`]
/// implementation.
///
/// Since this is a unit struct with `repr(transparent)`, we are able to turn un-adapted array
/// structs into [`dyn Array`] using some cheeky casting inside [`std::ops::Deref`] and
/// [`AsRef`]. See the `vtable!` macro for more details.
#[repr(transparent)]
pub struct ArrayAdapter<V: VTable>(V::Array);

impl<V: VTable> ArrayAdapter<V> {
    /// Provide a reference to the underlying array held within the adapter.
    pub fn as_inner(&self) -> &V::Array {
        &self.0
    }
}

impl<V: VTable> Debug for ArrayAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<V: VTable> ReduceNode for ArrayAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(V::dtype(&self.0).clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        self.0.as_opt::<ScalarFnVTable>().map(|a| a.scalar_fn())
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        self.nth_child(idx)
            .unwrap_or_else(|| vortex_panic!("Child index out of bounds: {}", idx))
    }

    fn child_count(&self) -> usize {
        self.nchildren()
    }
}

impl<V: VTable> DynArray for ArrayAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_array(&self) -> ArrayRef {
        Arc::new(ArrayAdapter::<V>(self.0.clone()))
    }

    fn len(&self) -> usize {
        V::len(&self.0)
    }

    fn dtype(&self) -> &DType {
        V::dtype(&self.0)
    }

    fn vtable(&self) -> &dyn DynVTable {
        V::vtable()
    }

    fn encoding_id(&self) -> ArrayId {
        V::id(&self.0)
    }

    fn slice(&self, range: Range<usize>) -> VortexResult<ArrayRef> {
        let start = range.start;
        let stop = range.end;

        if start == 0 && stop == self.len() {
            return Ok(self.to_array());
        }

        vortex_ensure!(
            start <= self.len(),
            "OutOfBounds: start {start} > length {}",
            self.len()
        );
        vortex_ensure!(
            stop <= self.len(),
            "OutOfBounds: stop {stop} > length {}",
            self.len()
        );

        vortex_ensure!(start <= stop, "start ({start}) must be <= stop ({stop})");

        if start == stop {
            return Ok(Canonical::empty(self.dtype()).into_array());
        }

        let sliced = SliceArray::try_new(self.to_array(), range)?
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

    fn filter(&self, mask: Mask) -> VortexResult<ArrayRef> {
        FilterArray::try_new(self.to_array(), mask)?
            .into_array()
            .optimize()
    }

    fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        DictArray::try_new(indices, self.to_array())?
            .into_array()
            .optimize()
    }

    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        vortex_ensure!(index < self.len(), OutOfBounds: index, 0, self.len());
        if self.is_invalid(index)? {
            return Ok(Scalar::null(self.dtype().clone()));
        }
        let scalar = <V::OperationsVTable as OperationsVTable<V>>::scalar_at(&self.0, index)?;
        vortex_ensure!(self.dtype() == scalar.dtype(), "Scalar dtype mismatch");
        Ok(scalar)
    }

    fn is_valid(&self, index: usize) -> VortexResult<bool> {
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

    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        Ok(!self.is_valid(index)?)
    }

    fn all_valid(&self) -> VortexResult<bool> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(true),
            Validity::AllInvalid => Ok(false),
            Validity::Array(a) => Ok(a.statistics().compute_min::<bool>().unwrap_or(false)),
        }
    }

    fn all_invalid(&self) -> VortexResult<bool> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(false),
            Validity::AllInvalid => Ok(true),
            Validity::Array(a) => Ok(!a.statistics().compute_max::<bool>().unwrap_or(true)),
        }
    }

    // TODO(ngates): deprecate this function since it requires compute.
    fn valid_count(&self) -> VortexResult<usize> {
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return Ok(self.len() - invalid_count);
        }

        let count = match self.validity()? {
            Validity::NonNullable | Validity::AllValid => self.len(),
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
        vortex_ensure!(count <= self.len(), "Valid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(self.len() - count));

        Ok(count)
    }

    fn invalid_count(&self) -> VortexResult<usize> {
        Ok(self.len() - self.valid_count()?)
    }

    fn validity(&self) -> VortexResult<Validity> {
        if self.dtype().is_nullable() {
            let validity = <V::ValidityVTable as ValidityVTable<V>>::validity(&self.0)?;
            if let Validity::Array(array) = &validity {
                vortex_ensure!(array.len() == self.len(), "Validity array length mismatch");
                vortex_ensure!(
                    matches!(array.dtype(), DType::Bool(Nullability::NonNullable)),
                    "Validity array is not non-nullable boolean: {}",
                    self.encoding_id(),
                );
            }
            Ok(validity)
        } else {
            Ok(Validity::NonNullable)
        }
    }

    fn validity_mask(&self) -> VortexResult<Mask> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(Mask::new_true(self.len())),
            Validity::AllInvalid => Ok(Mask::new_false(self.len())),
            Validity::Array(a) => Ok(a.to_bool().to_mask()),
        }
    }

    fn to_canonical(&self) -> VortexResult<Canonical> {
        self.to_array()
            .execute(&mut LEGACY_SESSION.create_execution_ctx())
    }

    fn append_to_builder(
        &self,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if builder.dtype() != self.dtype() {
            vortex_panic!(
                "Builder dtype mismatch: expected {}, got {}",
                self.dtype(),
                builder.dtype(),
            );
        }
        let len = builder.len();

        V::append_to_builder(&self.0, builder, ctx)?;

        assert_eq!(
            len + self.len(),
            builder.len(),
            "Builder length mismatch after writing array for encoding {}",
            self.encoding_id(),
        );
        Ok(())
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        V::stats(&self.0)
    }

    fn with_children(&self, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        let mut this = self.0.clone();
        V::with_children(&mut this, children)?;
        Ok(this.into_array())
    }
}

impl<V: VTable> ArrayHash for ArrayAdapter<V> {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: hash::Precision) {
        self.0.encoding_id().hash(state);
        V::array_hash(&self.0, state, precision);
    }
}

impl<V: VTable> ArrayEq for ArrayAdapter<V> {
    fn array_eq(&self, other: &Self, precision: hash::Precision) -> bool {
        V::array_eq(&self.0, &other.0, precision)
    }
}

impl<V: VTable> ArrayVisitor for ArrayAdapter<V> {
    fn children(&self) -> Vec<ArrayRef> {
        (0..V::nchildren(&self.0))
            .map(|i| V::child(&self.0, i))
            .collect()
    }

    fn nchildren(&self) -> usize {
        V::nchildren(&self.0)
    }

    fn nth_child(&self, idx: usize) -> Option<ArrayRef> {
        (idx < V::nchildren(&self.0)).then(|| V::child(&self.0, idx))
    }

    fn children_names(&self) -> Vec<String> {
        (0..V::nchildren(&self.0))
            .map(|i| V::child_name(&self.0, i))
            .collect()
    }

    fn named_children(&self) -> Vec<(String, ArrayRef)> {
        (0..V::nchildren(&self.0))
            .map(|i| (V::child_name(&self.0, i), V::child(&self.0, i)))
            .collect()
    }

    fn buffers(&self) -> Vec<ByteBuffer> {
        (0..V::nbuffers(&self.0))
            .map(|i| V::buffer(&self.0, i).to_host_sync())
            .collect()
    }

    fn buffer_handles(&self) -> Vec<BufferHandle> {
        (0..V::nbuffers(&self.0))
            .map(|i| V::buffer(&self.0, i))
            .collect()
    }

    fn buffer_names(&self) -> Vec<String> {
        (0..V::nbuffers(&self.0))
            .filter_map(|i| V::buffer_name(&self.0, i))
            .collect()
    }

    fn named_buffers(&self) -> Vec<(String, BufferHandle)> {
        (0..V::nbuffers(&self.0))
            .filter_map(|i| V::buffer_name(&self.0, i).map(|name| (name, V::buffer(&self.0, i))))
            .collect()
    }

    fn nbuffers(&self) -> usize {
        V::nbuffers(&self.0)
    }

    fn metadata(&self) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(V::metadata(&self.0)?)
    }

    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match V::metadata(&self.0) {
            Err(e) => write!(f, "<serde error: {e}>"),
            Ok(metadata) => Debug::fmt(&metadata, f),
        }
    }

    fn is_host(&self) -> bool {
        for array in self.depth_first_traversal() {
            if !array.buffer_handles().iter().all(BufferHandle::is_on_host) {
                return false;
            }
        }

        true
    }
}

/// Implement a matcher for a specific VTable type
impl<V: VTable> Matcher for V {
    type Match<'a> = &'a V::Array;

    fn matches(array: &dyn DynArray) -> bool {
        DynArray::as_any(array).is::<ArrayAdapter<V>>()
    }

    fn try_match<'a>(array: &'a dyn DynArray) -> Option<Self::Match<'a>> {
        DynArray::as_any(array)
            .downcast_ref::<ArrayAdapter<V>>()
            .map(|array_adapter| &array_adapter.0)
    }
}
