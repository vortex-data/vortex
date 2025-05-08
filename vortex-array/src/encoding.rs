//! Traits and types to define shared unique encoding identifiers.

use std::any::Any;
use std::fmt::{Debug, Formatter};

use arcref::ArcRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::serde::ArrayParts;
use crate::vtable::VTable;
use crate::{Array, ArrayContext, ArrayRef, Canonical};

/// EncodingId is a globally unique name of the array's encoding.
pub type EncodingId = ArcRef<str>;

/// Marker trait for array encodings with their associated Array type.
pub trait Encoding: 'static + private::Sealed + Send + Sync + Debug {
    /// Downcast the encoding to [`Any`].
    fn as_any(&self) -> &dyn Any;

    /// Returns the ID of the encoding.
    fn id(&self) -> EncodingId;

    /// Decode an array from the given [`ArrayParts`] and [`ArrayContext`].
    /// The array parts must be valid for the given encoding.
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef>;

    /// Encode the canonical array into this encoding implementation.
    /// Returns `None` if this encoding does not support the given canonical array, for example
    /// if the data type is incompatible.
    ///
    /// Panics if `like` is encoded with a different encoding.
    fn encode(&self, input: &Canonical, like: Option<&dyn Array>)
    -> VortexResult<Option<ArrayRef>>;
}

/// Adapter struct used to lift the [`VTable`] trait into an object-safe [`Encoding`]
/// implementation.
///
/// Since this is a unit struct with `repr(transparent)`, we are able to turn un-adapted array
/// structs into [`dyn Encoding`] using some cheeky casting inside [`Deref`] and [`AsRef`]. See
/// the `vtable!` macro for more details.
#[repr(transparent)]
pub struct EncodingAdapter<V: VTable>(V::Encoding);

impl<V: VTable> Encoding for EncodingAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> EncodingId {
        V::id(&self.0)
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        assert!(
            ctx.lookup_encoding(parts.encoding_id())
                .is_some_and(|e| e.id() == self.id()),
            "ArrayParts do not match the current encoding",
        );
        Ok(ArrayAdapter(V::decode(&self.0, parts, ctx, dtype, len)?).into_array())
    }

    fn encode(
        &self,
        _input: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        todo!()
    }
}

impl<V: VTable> Debug for EncodingAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoding").field("id", self.id()).finish()
    }
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for EncodingAdapter<V> {}
}

/// Adapter struct used to lift the [`VTable`] trait into an object-safe [`Array`] implementation.
///
/// Since this is a unit struct with `repr(transparent)`, we are able to turn un-adapted array
/// structs into [`dyn Array`] using some cheeky casting inside [`Deref`] and [`AsRef`]. See
/// the `vtable!` macro for more details.
#[repr(transparent)]
pub struct ArrayAdapter<V: VTable>(V::Array);

impl<V: VTable> Array for ArrayAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        V::len(&self.0)
    }

    fn dtype(&self) -> &DType {
        V::dtype(&self.0)
    }
}
