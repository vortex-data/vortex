mod convert;
mod statistics;
mod visitor;

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

pub use convert::*;
pub use visitor::*;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{
    BoolEncoding, DecimalEncoding, ExtensionEncoding, ListEncoding, NullEncoding,
    PrimitiveEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use crate::builders::ArrayBuilder;
use crate::compute::{ComputeFn, Cost, InvocationArgs, Output};
use crate::serde::ArrayChildren;
use crate::stats::{Precision, Stat, StatsProviderExt, StatsSetRef};
use crate::vtable::{
    ArrayVTable, CanonicalVTable, ComputeVTable, OperationsVTable, SerdeVTable, VTable,
    ValidityVTable, VisitorVTable,
};
use crate::{Canonical, EncodingId, EncodingRef, SerializeMetadata};

/// The public API trait for all Vortex arrays.
pub trait Array: 'static + private::Sealed + Send + Sync + Debug + ArrayVisitor {
    /// Returns the array as a reference to a generic [`Any`] trait object.
    fn as_any(&self) -> &dyn Any;

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
    fn encoding(&self) -> EncodingRef;

    /// Returns the encoding ID of the array.
    fn encoding_id(&self) -> EncodingId;

    /// Performs a constant-time slice of the array.
    fn slice(&self, start: usize, end: usize) -> VortexResult<ArrayRef>;

    /// Fetch the scalar at the given index.
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar>;

    /// Returns whether the array is of the given encoding.
    fn is_encoding(&self, encoding: EncodingId) -> bool {
        self.encoding_id() == encoding
    }

    /// Returns whether this array is an arrow encoding.
    // TODO(ngates): this shouldn't live here.
    fn is_arrow(&self) -> bool {
        self.is_encoding(NullEncoding.id())
            || self.is_encoding(BoolEncoding.id())
            || self.is_encoding(PrimitiveEncoding.id())
            || self.is_encoding(VarBinEncoding.id())
            || self.is_encoding(VarBinViewEncoding.id())
    }

    /// Whether the array is of a canonical encoding.
    // TODO(ngates): this shouldn't live here.
    fn is_canonical(&self) -> bool {
        self.is_encoding(NullEncoding.id())
            || self.is_encoding(BoolEncoding.id())
            || self.is_encoding(PrimitiveEncoding.id())
            || self.is_encoding(DecimalEncoding.id())
            || self.is_encoding(StructEncoding.id())
            || self.is_encoding(ListEncoding.id())
            || self.is_encoding(VarBinViewEncoding.id())
            || self.is_encoding(ExtensionEncoding.id())
    }

    /// Returns whether the item at `index` is valid.
    fn is_valid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether the item at `index` is invalid.
    fn is_invalid(&self, index: usize) -> VortexResult<bool>;

    /// Returns whether all items in the array are valid.
    ///
    /// This is usually cheaper than computing a precise `valid_count`.
    fn all_valid(&self) -> VortexResult<bool>;

    /// Returns whether the array is all invalid.
    ///
    /// This is usually cheaper than computing a precise `invalid_count`.
    fn all_invalid(&self) -> VortexResult<bool>;

    /// Returns the number of valid elements in the array.
    fn valid_count(&self) -> VortexResult<usize>;

    /// Returns the number of invalid elements in the array.
    fn invalid_count(&self) -> VortexResult<usize>;

    /// Returns the canonical validity mask for the array.
    fn validity_mask(&self) -> VortexResult<Mask>;

    /// Returns the canonical representation of the array.
    fn to_canonical(&self) -> VortexResult<Canonical>;

    /// Writes the array into the canonical builder.
    ///
    /// The [`DType`] of the builder must match that of the array.
    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()>;

    /// Returns the statistics of the array.
    // TODO(ngates): change how this works. It's weird.
    fn statistics(&self) -> StatsSetRef<'_>;

    /// Replaces the children of the array with the given array references.
    fn with_children(&self, children: &[ArrayRef]) -> VortexResult<ArrayRef>;

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
    fn as_any(&self) -> &dyn Any {
        self.as_ref().as_any()
    }

    fn to_array(&self) -> ArrayRef {
        self.clone()
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn dtype(&self) -> &DType {
        self.as_ref().dtype()
    }

    fn encoding(&self) -> EncodingRef {
        self.as_ref().encoding()
    }

    fn encoding_id(&self) -> EncodingId {
        self.as_ref().encoding_id()
    }

    fn slice(&self, start: usize, end: usize) -> VortexResult<ArrayRef> {
        self.as_ref().slice(start, end)
    }

    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        self.as_ref().scalar_at(index)
    }

    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        self.as_ref().is_valid(index)
    }

    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        self.as_ref().is_invalid(index)
    }

    fn all_valid(&self) -> VortexResult<bool> {
        self.as_ref().all_valid()
    }

    fn all_invalid(&self) -> VortexResult<bool> {
        self.as_ref().all_invalid()
    }

    fn valid_count(&self) -> VortexResult<usize> {
        self.as_ref().valid_count()
    }

    fn invalid_count(&self) -> VortexResult<usize> {
        self.as_ref().invalid_count()
    }

    fn validity_mask(&self) -> VortexResult<Mask> {
        self.as_ref().validity_mask()
    }

    fn to_canonical(&self) -> VortexResult<Canonical> {
        self.as_ref().to_canonical()
    }

    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        self.as_ref().append_to_builder(builder)
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        self.as_ref().statistics()
    }

    fn with_children(&self, children: &[ArrayRef]) -> VortexResult<ArrayRef> {
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

    /// Is self an array with encoding from vtable `V`.
    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }
}

impl Display for dyn Array {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({}, len={})",
            self.encoding_id(),
            self.dtype(),
            self.len()
        )
    }
}

impl dyn Array + '_ {
    /// Total size of the array in bytes, including all children and buffers.
    // TODO(ngates): this should return u64
    pub fn nbytes(&self) -> usize {
        let mut nbytes = 0;
        for array in self.depth_first_traversal() {
            for buffer in array.buffers() {
                nbytes += buffer.len();
            }
        }
        nbytes
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

impl<V: VTable> Debug for ArrayAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<V: VTable> Array for ArrayAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_array(&self) -> ArrayRef {
        Arc::new(ArrayAdapter::<V>(self.0.clone()))
    }

    fn len(&self) -> usize {
        <V::ArrayVTable as ArrayVTable<V>>::len(&self.0)
    }

    fn dtype(&self) -> &DType {
        <V::ArrayVTable as ArrayVTable<V>>::dtype(&self.0)
    }

    fn encoding(&self) -> EncodingRef {
        V::encoding(&self.0)
    }

    fn encoding_id(&self) -> EncodingId {
        V::encoding(&self.0).id()
    }

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
        let derived_stats = (!self.0.is_constant_opts(Cost::Negligible)).then(|| {
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

        let sliced = <V::OperationsVTable as OperationsVTable<V>>::slice(&self.0, start, stop)?;

        assert_eq!(
            sliced.len(),
            stop - start,
            "Slice length mismatch {}",
            self.encoding_id()
        );
        assert_eq!(
            sliced.dtype(),
            self.dtype(),
            "Slice dtype mismatch {}",
            self.encoding_id()
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

    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        if index >= self.len() {
            vortex_bail!(OutOfBounds: index, 0, self.len());
        }
        if self.is_invalid(index)? {
            return Ok(Scalar::null(self.dtype().clone()));
        }
        let scalar = <V::OperationsVTable as OperationsVTable<V>>::scalar_at(&self.0, index)?;
        assert_eq!(self.dtype(), scalar.dtype(), "Scalar dtype mismatch");
        Ok(scalar)
    }

    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        if index >= self.len() {
            vortex_bail!(OutOfBounds: index, 0, self.len());
        }
        <V::ValidityVTable as ValidityVTable<V>>::is_valid(&self.0, index)
    }

    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        self.is_valid(index).map(|valid| !valid)
    }

    fn all_valid(&self) -> VortexResult<bool> {
        <V::ValidityVTable as ValidityVTable<V>>::all_valid(&self.0)
    }

    fn all_invalid(&self) -> VortexResult<bool> {
        <V::ValidityVTable as ValidityVTable<V>>::all_invalid(&self.0)
    }

    fn valid_count(&self) -> VortexResult<usize> {
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return Ok(self.len() - invalid_count);
        }

        let count = <V::ValidityVTable as ValidityVTable<V>>::valid_count(&self.0)?;
        assert!(count <= self.len(), "Valid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(self.len() - count));

        Ok(count)
    }

    fn invalid_count(&self) -> VortexResult<usize> {
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return Ok(invalid_count);
        }

        let count = <V::ValidityVTable as ValidityVTable<V>>::invalid_count(&self.0)?;
        assert!(count <= self.len(), "Invalid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(count));

        Ok(count)
    }

    fn validity_mask(&self) -> VortexResult<Mask> {
        let mask = <V::ValidityVTable as ValidityVTable<V>>::validity_mask(&self.0)?;
        assert_eq!(mask.len(), self.len(), "Validity mask length mismatch");
        Ok(mask)
    }

    fn to_canonical(&self) -> VortexResult<Canonical> {
        let canonical = <V::CanonicalVTable as CanonicalVTable<V>>::canonicalize(&self.0)?;
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
        canonical.as_ref().statistics().inherit(self.statistics());
        Ok(canonical)
    }

    fn append_to_builder(&self, builder: &mut dyn ArrayBuilder) -> VortexResult<()> {
        if builder.dtype() != self.dtype() {
            vortex_bail!(
                "Builder dtype mismatch: expected {}, got {}",
                self.dtype(),
                builder.dtype(),
            );
        }
        let len = builder.len();

        <V::CanonicalVTable as CanonicalVTable<V>>::append_to_builder(&self.0, builder)?;
        assert_eq!(
            len + self.len(),
            builder.len(),
            "Builder length mismatch after writing array for encoding {}",
            self.encoding_id(),
        );
        Ok(())
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        <V::ArrayVTable as ArrayVTable<V>>::stats(&self.0)
    }

    fn with_children(&self, children: &[ArrayRef]) -> VortexResult<ArrayRef> {
        struct ReplacementChildren<'a> {
            children: &'a [ArrayRef],
        }

        impl ArrayChildren for ReplacementChildren<'_> {
            fn get(&self, index: usize, dtype: &DType, len: usize) -> VortexResult<ArrayRef> {
                if index >= self.children.len() {
                    vortex_bail!(OutOfBounds: index, 0, self.children.len());
                }
                let child = &self.children[index];
                if child.len() != len {
                    vortex_bail!(
                        "Child length mismatch: expected {}, got {}",
                        len,
                        child.len()
                    );
                }
                if child.dtype() != dtype {
                    vortex_bail!(
                        "Child dtype mismatch: expected {}, got {}",
                        dtype,
                        child.dtype()
                    );
                }
                Ok(child.clone())
            }

            fn len(&self) -> usize {
                self.children.len()
            }
        }

        let metadata = self.metadata()?.ok_or_else(|| {
            vortex_err!("Cannot replace children for arrays that do not support serialization")
        })?;

        // Replace the children of the array by re-building the array from parts.
        self.encoding().build(
            self.dtype(),
            self.len(),
            &metadata,
            &self.buffers(),
            &ReplacementChildren { children },
        )
    }

    fn invoke(
        &self,
        compute_fn: &ComputeFn,
        args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        <V::ComputeVTable as ComputeVTable<V>>::invoke(&self.0, compute_fn, args)
    }
}

impl<V: VTable> ArrayVisitor for ArrayAdapter<V> {
    fn children(&self) -> Vec<ArrayRef> {
        struct ChildrenCollector {
            children: Vec<ArrayRef>,
        }

        impl ArrayChildVisitor for ChildrenCollector {
            fn visit_child(&mut self, _name: &str, array: &dyn Array) {
                self.children.push(array.to_array());
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
            fn visit_child(&mut self, name: &str, _array: &dyn Array) {
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
            fn visit_child(&mut self, name: &str, array: &dyn Array) {
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
        Ok(<V::SerdeVTable as SerdeVTable<V>>::metadata(&self.0)?.map(|m| m.serialize()))
    }

    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match <V::SerdeVTable as SerdeVTable<V>>::metadata(&self.0) {
            Err(e) => write!(f, "<serde error: {}>", e),
            Ok(None) => write!(f, "<serde not supported>"),
            Ok(Some(metadata)) => Debug::fmt(&metadata, f),
        }
    }
}
