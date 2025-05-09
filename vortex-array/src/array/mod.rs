mod canonical;
mod convert;
mod implementation;
mod operations;
mod statistics;
mod validity;
mod visitor;

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

pub use canonical::*;
pub use convert::*;
pub use implementation::*;
pub use operations::*;
pub use statistics::*;
pub use validity::*;
pub use visitor::*;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::{
    BoolEncoding, ConstantEncoding, DecimalEncoding, ExtensionEncoding, ListEncoding, NullEncoding,
    PrimitiveEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use crate::builders::ArrayBuilder;
use crate::compute::{ComputeFn, InvocationArgs, Output};
use crate::stats::{Precision, Stat, StatsProviderExt, StatsSetRef};
use crate::vtable::{CanonicalVTable, OperationsVTable, VTable};
use crate::{Canonical, Encoding, EncodingId, EncodingRef};

/// The base trait for all Vortex arrays.
///
/// Users should invoke functions on this trait. Implementations should implement the corresponding
/// function on the `_Impl` traits, e.g. [`ArrayValidityImpl`]. The functions here dispatch to the
/// implementations, while validating pre- and post-conditions.
pub trait Array:
    'static + private::Sealed + Send + Sync + Debug + ArrayStatistics + ArrayVisitor
{
    /// Returns the array as a reference to a generic [`Any`] trait object.
    fn as_any(&self) -> &dyn Any;

    /// Returns the array as an [`Arc`] reference to a generic [`Any`] trait object.
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

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
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

impl<A: Array + Clone + 'static> TryFromArrayRef for A {
    fn try_from_array(array: ArrayRef) -> Result<Self, ArrayRef> {
        let fallback = array.clone();
        if let Ok(array) = array.as_any_arc().downcast::<A>() {
            // manually drop the fallback value so `Arc::unwrap_or_clone` doesn't always have to clone
            drop(fallback);
            Ok(Arc::unwrap_or_clone(array))
        } else {
            Err(fallback)
        }
    }
}

impl<A: Array + Clone + 'static> TryFromArrayRef for Arc<A> {
    fn try_from_array(array: ArrayRef) -> Result<Self, ArrayRef> {
        let fallback = array.clone();
        array.as_any_arc().downcast::<A>().map_err(|_| fallback)
    }
}

pub trait ArrayExt: Array {
    /// Returns the array downcast to the given `A`.
    fn as_<A: Array>(&self) -> &A {
        self.as_any()
            .downcast_ref::<A>()
            .vortex_expect("Failed to downcast")
    }

    /// Returns the array downcast to the given `A`.
    fn as_opt<A: Array>(&self) -> Option<&A> {
        self.as_any().downcast_ref::<A>()
    }

    /// Is self an array with encoding `A`.
    fn is<A: Array>(&self) -> bool {
        self.as_opt::<A>().is_some()
    }
}

impl<A: Array + ?Sized> ArrayExt for A {}

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

#[macro_export]
macro_rules! try_from_array_ref {
    ($Array:ty) => {
        impl TryFrom<$crate::ArrayRef> for $Array {
            type Error = vortex_error::VortexError;

            fn try_from(value: $crate::ArrayRef) -> Result<Self, Self::Error> {
                Ok(::std::sync::Arc::unwrap_or_clone(
                    value.as_any_arc().downcast::<Self>().map_err(|_| {
                        vortex_error::vortex_err!(
                            "Cannot downcast to {}",
                            std::any::type_name::<Self>()
                        )
                    })?,
                ))
            }
        }
    };
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
/// structs into [`dyn Array`] using some cheeky casting inside [`Deref`] and [`AsRef`]. See
/// the `vtable!` macro for more details.
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

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_array(&self) -> ArrayRef {
        Arc::new(self.clone())
    }

    fn len(&self) -> usize {
        V::len(&self.0)
    }

    fn dtype(&self) -> &DType {
        V::dtype(&self.0)
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
        let derived_stats = (!self.is::<ConstantEncoding>()).then(|| {
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
        todo!()
    }

    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        todo!()
    }

    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        todo!()
    }

    fn all_valid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn all_invalid(&self) -> VortexResult<bool> {
        todo!()
    }

    fn valid_count(&self) -> VortexResult<usize> {
        todo!()
    }

    fn invalid_count(&self) -> VortexResult<usize> {
        todo!()
    }

    fn validity_mask(&self) -> VortexResult<Mask> {
        todo!()
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
        todo!()
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        todo!()
    }

    fn with_children(&self, children: &[ArrayRef]) -> VortexResult<ArrayRef> {
        todo!()
    }

    fn invoke(
        &self,
        compute_fn: &ComputeFn,
        args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        todo!()
    }
}

impl<V: VTable> ArrayVisitor for ArrayAdapter<V> {
    fn children(&self) -> Vec<ArrayRef> {
        todo!()
    }

    fn nchildren(&self) -> usize {
        todo!()
    }

    fn children_names(&self) -> Vec<String> {
        todo!()
    }

    fn named_children(&self) -> Vec<(String, ArrayRef)> {
        todo!()
    }

    fn buffers(&self) -> Vec<ByteBuffer> {
        todo!()
    }

    fn nbuffers(&self) -> usize {
        todo!()
    }

    fn metadata(&self) -> Option<Vec<u8>> {
        todo!()
    }

    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
