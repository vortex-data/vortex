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
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::Canonical;
use crate::DynArrayEq;
use crate::DynArrayHash;
use crate::arrays::BoolVTable;
use crate::arrays::ConstantVTable;
use crate::arrays::DecimalVTable;
use crate::arrays::DictArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::FilterArray;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListViewVTable;
use crate::arrays::NullVTable;
use crate::arrays::PrimitiveVTable;
use crate::arrays::StructVTable;
use crate::arrays::VarBinVTable;
use crate::arrays::VarBinViewVTable;
use crate::builders::ArrayBuilder;
use crate::compute::ComputeFn;
use crate::compute::Cost;
use crate::compute::InvocationArgs;
use crate::compute::IsConstantOpts;
use crate::compute::Output;
use crate::compute::is_constant_opts;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::hash;
use crate::optimizer::ArrayOptimizer;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;
use crate::vtable::BaseArrayVTable;
use crate::vtable::CanonicalVTable;
use crate::vtable::ComputeVTable;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;
use crate::vtable::VisitorVTable;

/// The public API trait for all Vortex arrays.
pub trait Array:
    'static + private::Sealed + Send + Sync + Debug + DynArrayEq + DynArrayHash + ArrayVisitor
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

    /// Returns the encoding of the array.
    fn encoding(&self) -> ArrayVTable;

    /// Returns the encoding ID of the array.
    fn encoding_id(&self) -> ArrayId;

    /// Performs a constant-time slice of the array.
    fn slice(&self, range: Range<usize>) -> ArrayRef;

    /// Wraps the array in a [`FilterArray`] such that it is logically filtered by the given mask.
    fn filter(&self, mask: Mask) -> VortexResult<ArrayRef>;

    /// Wraps the array in a [`DictArray`] such that it is logically taken by the given indices.
    fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef>;

    /// Fetch the scalar at the given index.
    ///
    /// This method panics if the index is out of bounds for the array.
    fn scalar_at(&self, index: usize) -> Scalar;

    /// Returns whether the array is of the given encoding.
    fn is_encoding(&self, encoding: ArrayId) -> bool {
        self.encoding_id() == encoding
    }

    /// Returns whether this array is an arrow encoding.
    // TODO(ngates): this shouldn't live here.
    fn is_arrow(&self) -> bool {
        self.is_encoding(NullVTable.id())
            || self.is_encoding(BoolVTable.id())
            || self.is_encoding(PrimitiveVTable.id())
            || self.is_encoding(VarBinVTable.id())
            || self.is_encoding(VarBinViewVTable.id())
    }

    /// Whether the array is of a canonical encoding.
    // TODO(ngates): this shouldn't live here.
    fn is_canonical(&self) -> bool {
        self.is_encoding(NullVTable.id())
            || self.is_encoding(BoolVTable.id())
            || self.is_encoding(PrimitiveVTable.id())
            || self.is_encoding(DecimalVTable.id())
            || self.is_encoding(StructVTable.id())
            || self.is_encoding(ListViewVTable.id())
            || self.is_encoding(FixedSizeListVTable.id())
            || self.is_encoding(VarBinViewVTable.id())
            || self.is_encoding(ExtensionVTable.id())
    }

    /// Returns whether the item at `index` is valid.
    fn is_valid(&self, index: usize) -> bool;

    /// Returns whether the item at `index` is invalid.
    fn is_invalid(&self, index: usize) -> bool;

    /// Returns whether all items in the array are valid.
    ///
    /// This is usually cheaper than computing a precise `valid_count`.
    fn all_valid(&self) -> bool;

    /// Returns whether the array is all invalid.
    ///
    /// This is usually cheaper than computing a precise `invalid_count`.
    fn all_invalid(&self) -> bool;

    /// Returns the number of valid elements in the array.
    fn valid_count(&self) -> usize;

    /// Returns the number of invalid elements in the array.
    fn invalid_count(&self) -> usize;

    /// Returns the [`Validity`] of the array.
    fn validity(&self) -> VortexResult<Validity>;

    /// Returns the canonical validity mask for the array.
    fn validity_mask(&self) -> Mask;

    /// Returns the canonical representation of the array.
    fn to_canonical(&self) -> Canonical;

    /// Writes the array into the canonical builder.
    ///
    /// The [`DType`] of the builder must match that of the array.
    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder);

    /// Returns the statistics of the array.
    // TODO(ngates): change how this works. It's weird.
    fn statistics(&self) -> StatsSetRef<'_>;

    /// Replaces the children of the array with the given array references.
    fn with_children(&self, children: Vec<ArrayRef>) -> VortexResult<ArrayRef>;

    /// Optionally invoke a kernel for the given compute function.
    ///
    /// These encoding-specific kernels are independent of kernels registered directly with
    /// compute functions using [`ComputeFn::register_kernel`], and are attempted only if none of
    /// the function-specific kernels returns a result.
    ///
    /// This allows encodings the opportunity to generically implement many compute functions
    /// that share some property, for example [`ComputeFn::is_elementwise`], without prior
    /// knowledge of the function itself, while still allowing users to override the implementation
    /// of compute functions for built-in encodings. For an example, see the implementation for
    /// chunked arrays.
    ///
    /// The first input in the [`InvocationArgs`] is always the array itself.
    ///
    /// Warning: do not call `compute_fn.invoke(args)` directly, as this will result in a recursive
    /// call.
    fn invoke(&self, compute_fn: &ComputeFn, args: &InvocationArgs)
    -> VortexResult<Option<Output>>;
}

impl Array for Arc<dyn Array> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self.as_ref().as_any()
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

    #[inline]
    fn encoding(&self) -> ArrayVTable {
        self.as_ref().encoding()
    }

    #[inline]
    fn encoding_id(&self) -> ArrayId {
        self.as_ref().encoding_id()
    }

    #[inline]
    fn slice(&self, range: Range<usize>) -> ArrayRef {
        self.as_ref().slice(range)
    }

    fn filter(&self, mask: Mask) -> VortexResult<ArrayRef> {
        self.as_ref().filter(mask)
    }

    fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        self.as_ref().take(indices)
    }

    #[inline]
    fn scalar_at(&self, index: usize) -> Scalar {
        self.as_ref().scalar_at(index)
    }

    #[inline]
    fn is_valid(&self, index: usize) -> bool {
        self.as_ref().is_valid(index)
    }

    #[inline]
    fn is_invalid(&self, index: usize) -> bool {
        self.as_ref().is_invalid(index)
    }

    #[inline]
    fn all_valid(&self) -> bool {
        self.as_ref().all_valid()
    }

    #[inline]
    fn all_invalid(&self) -> bool {
        self.as_ref().all_invalid()
    }

    #[inline]
    fn valid_count(&self) -> usize {
        self.as_ref().valid_count()
    }

    #[inline]
    fn invalid_count(&self) -> usize {
        self.as_ref().invalid_count()
    }

    #[inline]
    fn validity(&self) -> VortexResult<Validity> {
        self.as_ref().validity()
    }

    #[inline]
    fn validity_mask(&self) -> Mask {
        self.as_ref().validity_mask()
    }

    fn to_canonical(&self) -> Canonical {
        self.as_ref().to_canonical()
    }

    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) {
        self.as_ref().append_to_builder(builder)
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        self.as_ref().statistics()
    }

    fn with_children(&self, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        self.as_ref().with_children(children)
    }

    fn invoke(
        &self,
        compute_fn: &ComputeFn,
        args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        self.as_ref().invoke(compute_fn, args)
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

impl dyn Array + '_ {
    /// Returns the array downcast to the given `A`.
    pub fn as_<V: VTable>(&self) -> &V::Array {
        self.as_opt::<V>().vortex_expect("Failed to downcast")
    }

    /// Returns the array downcast to the given `A`.
    pub fn as_opt<V: VTable>(&self) -> Option<&V::Array> {
        self.as_any()
            .downcast_ref::<ArrayAdapter<V>>()
            .map(|array_adapter| &array_adapter.0)
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

    /// Is self an array with encoding from vtable `V`.
    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    pub fn is_constant(&self) -> bool {
        let opts = IsConstantOpts {
            cost: Cost::Specialized,
        };
        is_constant_opts(self, &opts)
            .inspect_err(|e| tracing::warn!("Failed to compute IsConstant: {e}"))
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    pub fn is_constant_opts(&self, cost: Cost) -> bool {
        let opts = IsConstantOpts { cost };
        is_constant_opts(self, &opts)
            .inspect_err(|e| tracing::warn!("Failed to compute IsConstant: {e}"))
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    pub fn as_constant(&self) -> Option<Scalar> {
        self.is_constant().then(|| self.scalar_at(0))
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
    impl Sealed for Arc<dyn Array> {}
}

/// Adapter struct used to lift the [`VTable`] trait into an object-safe [`Array`]
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

impl<V: VTable> Array for ArrayAdapter<V> {
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
        <V::ArrayVTable as BaseArrayVTable<V>>::len(&self.0)
    }

    fn dtype(&self) -> &DType {
        <V::ArrayVTable as BaseArrayVTable<V>>::dtype(&self.0)
    }

    fn encoding(&self) -> ArrayVTable {
        V::encoding(&self.0)
    }

    fn encoding_id(&self) -> ArrayId {
        V::encoding(&self.0).id()
    }

    fn slice(&self, range: Range<usize>) -> ArrayRef {
        let start = range.start;
        let stop = range.end;

        if start == 0 && stop == self.len() {
            return self.to_array();
        }

        assert!(
            start <= self.len(),
            "OutOfBounds: start {start} > length {}",
            self.len()
        );
        assert!(
            stop <= self.len(),
            "OutOfBounds: stop {stop} > length {}",
            self.len()
        );

        assert!(start <= stop, "start ({start}) must be <= stop ({stop})");

        if start == stop {
            return Canonical::empty(self.dtype()).into_array();
        }

        let sliced = <V::OperationsVTable as OperationsVTable<V>>::slice(&self.0, range);

        assert_eq!(
            sliced.len(),
            stop - start,
            "Slice length mismatch {}",
            self.encoding_id()
        );

        // Slightly more expensive, so only do this in debug builds.
        debug_assert_eq!(
            sliced.dtype(),
            self.dtype(),
            "Slice dtype mismatch {}",
            self.encoding_id()
        );

        // Propagate some stats from the original array to the sliced array.
        if !sliced.is::<ConstantVTable>() {
            self.statistics().with_iter(|iter| {
                sliced.statistics().inherit(iter.filter(|(stat, value)| {
                    matches!(
                        stat,
                        Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted
                    ) && value.as_ref().as_exact().is_some_and(|v| {
                        Scalar::new(DType::Bool(Nullability::NonNullable), v.clone())
                            .as_bool()
                            .value()
                            .unwrap_or_default()
                    })
                }));
            });
        }

        sliced
    }

    fn filter(&self, mask: Mask) -> VortexResult<ArrayRef> {
        vortex_ensure!(self.len() == mask.len(), "Filter mask length mismatch");
        FilterArray::new(self.to_array(), mask)
            .into_array()
            .optimize()
    }

    fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        DictArray::try_new(indices, self.to_array())?
            .into_array()
            .optimize()
    }

    fn scalar_at(&self, index: usize) -> Scalar {
        assert!(index < self.len(), "index {index} out of bounds");
        if self.is_invalid(index) {
            return Scalar::null(self.dtype().clone());
        }
        let scalar = <V::OperationsVTable as OperationsVTable<V>>::scalar_at(&self.0, index);
        assert_eq!(self.dtype(), scalar.dtype(), "Scalar dtype mismatch");
        scalar
    }

    fn is_valid(&self, index: usize) -> bool {
        if index >= self.len() {
            vortex_panic!(OutOfBounds: index, 0, self.len());
        }
        <V::ValidityVTable as ValidityVTable<V>>::is_valid(&self.0, index)
    }

    fn is_invalid(&self, index: usize) -> bool {
        !self.is_valid(index)
    }

    fn all_valid(&self) -> bool {
        <V::ValidityVTable as ValidityVTable<V>>::all_valid(&self.0)
    }

    fn all_invalid(&self) -> bool {
        <V::ValidityVTable as ValidityVTable<V>>::all_invalid(&self.0)
    }

    fn valid_count(&self) -> usize {
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return self.len() - invalid_count;
        }

        let count = <V::ValidityVTable as ValidityVTable<V>>::valid_count(&self.0);
        assert!(count <= self.len(), "Valid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(self.len() - count));

        count
    }

    fn invalid_count(&self) -> usize {
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return invalid_count;
        }

        let count = <V::ValidityVTable as ValidityVTable<V>>::invalid_count(&self.0);
        assert!(count <= self.len(), "Invalid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(count));

        count
    }

    fn validity(&self) -> VortexResult<Validity> {
        if self.dtype().is_nullable() {
            <V::ValidityVTable as ValidityVTable<V>>::validity(&self.0)
        } else {
            Ok(Validity::NonNullable)
        }
    }

    fn validity_mask(&self) -> Mask {
        let mask = <V::ValidityVTable as ValidityVTable<V>>::validity_mask(&self.0);
        assert_eq!(mask.len(), self.len(), "Validity mask length mismatch");
        mask
    }

    fn to_canonical(&self) -> Canonical {
        let canonical = <V::CanonicalVTable as CanonicalVTable<V>>::canonicalize(&self.0);

        assert_eq!(
            self.len(),
            canonical.as_ref().len(),
            "Canonical length mismatch {}. Expected {} but encoded into {}.",
            self.encoding_id(),
            self.len(),
            canonical.as_ref().len()
        );
        assert_eq!(
            self.dtype(),
            canonical.as_ref().dtype(),
            "Canonical dtype mismatch {}. Expected {} but encoded into {}.",
            self.encoding_id(),
            self.dtype(),
            canonical.as_ref().dtype()
        );
        canonical
            .as_ref()
            .statistics()
            .inherit_from(self.statistics());
        canonical
    }

    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) {
        if builder.dtype() != self.dtype() {
            vortex_panic!(
                "Builder dtype mismatch: expected {}, got {}",
                self.dtype(),
                builder.dtype(),
            );
        }
        let len = builder.len();

        <V::CanonicalVTable as CanonicalVTable<V>>::append_to_builder(&self.0, builder);
        assert_eq!(
            len + self.len(),
            builder.len(),
            "Builder length mismatch after writing array for encoding {}",
            self.encoding_id(),
        );
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        <V::ArrayVTable as BaseArrayVTable<V>>::stats(&self.0)
    }

    fn with_children(&self, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        self.encoding().as_dyn().with_children(self, children)
    }

    fn invoke(
        &self,
        compute_fn: &ComputeFn,
        args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        <V::ComputeVTable as ComputeVTable<V>>::invoke(&self.0, compute_fn, args)
    }
}

impl<V: VTable> ArrayHash for ArrayAdapter<V> {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: hash::Precision) {
        self.0.encoding_id().hash(state);
        <V::ArrayVTable as BaseArrayVTable<V>>::array_hash(&self.0, state, precision);
    }
}

impl<V: VTable> ArrayEq for ArrayAdapter<V> {
    fn array_eq(&self, other: &Self, precision: hash::Precision) -> bool {
        <V::ArrayVTable as BaseArrayVTable<V>>::array_eq(&self.0, &other.0, precision)
    }
}

impl<V: VTable> ArrayVisitor for ArrayAdapter<V> {
    fn children(&self) -> Vec<ArrayRef> {
        struct ChildrenCollector {
            children: Vec<ArrayRef>,
        }

        impl ArrayChildVisitor for ChildrenCollector {
            fn visit_child(&mut self, _name: &str, array: &ArrayRef) {
                self.children.push(array.clone());
            }
        }

        let mut collector = ChildrenCollector {
            children: Vec::new(),
        };
        <V::VisitorVTable as VisitorVTable<V>>::visit_children(&self.0, &mut collector);
        collector.children
    }

    fn nchildren(&self) -> usize {
        <V::VisitorVTable as VisitorVTable<V>>::nchildren(&self.0)
    }

    fn children_names(&self) -> Vec<String> {
        struct ChildNameCollector {
            names: Vec<String>,
        }

        impl ArrayChildVisitor for ChildNameCollector {
            fn visit_child(&mut self, name: &str, _array: &ArrayRef) {
                self.names.push(name.to_string());
            }
        }

        let mut collector = ChildNameCollector { names: Vec::new() };
        <V::VisitorVTable as VisitorVTable<V>>::visit_children(&self.0, &mut collector);
        collector.names
    }

    fn named_children(&self) -> Vec<(String, ArrayRef)> {
        struct NamedChildrenCollector {
            children: Vec<(String, ArrayRef)>,
        }

        impl ArrayChildVisitor for NamedChildrenCollector {
            fn visit_child(&mut self, name: &str, array: &ArrayRef) {
                self.children.push((name.to_string(), array.to_array()));
            }
        }

        let mut collector = NamedChildrenCollector {
            children: Vec::new(),
        };

        <V::VisitorVTable as VisitorVTable<V>>::visit_children(&self.0, &mut collector);
        collector.children
    }

    fn buffers(&self) -> Vec<ByteBuffer> {
        struct BufferCollector {
            buffers: Vec<ByteBuffer>,
        }

        impl ArrayBufferVisitor for BufferCollector {
            fn visit_buffer(&mut self, buffer: &ByteBuffer) {
                self.buffers.push(buffer.clone());
            }
        }

        let mut collector = BufferCollector {
            buffers: Vec::new(),
        };
        <V::VisitorVTable as VisitorVTable<V>>::visit_buffers(&self.0, &mut collector);
        collector.buffers
    }

    fn nbuffers(&self) -> usize {
        <V::VisitorVTable as VisitorVTable<V>>::nbuffers(&self.0)
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
}
