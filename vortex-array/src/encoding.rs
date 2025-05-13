//! Traits and types to define shared unique encoding identifiers.

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use arcref::ArcRef;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::serde::ArrayParts;
use crate::vtable::{EncodeVTable, SerdeVTable, VTable};
use crate::{Array, ArrayContext, ArrayRef, Canonical, DeserializeMetadata, IntoArray};

/// EncodingId is a globally unique name of the array's encoding.
pub type EncodingId = ArcRef<str>;

pub type EncodingRef = ArcRef<dyn Encoding>;

/// Marker trait for array encodings with their associated Array type.
pub trait Encoding: 'static + private::Sealed + Send + Sync + Debug {
    /// Downcast the encoding to [`Any`].
    fn as_any(&self) -> &dyn Any;

    fn to_encoding(&self) -> EncodingRef;

    fn into_encoding(self) -> EncodingRef
    where
        Self: Sized;

    /// Returns the ID of the encoding.
    fn id(&self) -> EncodingId;

    /// Decode an array from the given [`ArrayParts`] and [`ArrayContext`].
    /// The array parts must be valid for the given encoding.
    fn decode(
        &self,
        dtype: DType,
        len: usize,
        metadata: Option<&[u8]>,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
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
/// structs into [`dyn Encoding`] using some cheeky casting inside [`std::ops::Deref`] and
/// [`AsRef`]. See the `vtable!` macro for more details.
#[repr(transparent)]
pub struct EncodingAdapter<V: VTable>(V::Encoding);

impl<V: VTable> Encoding for EncodingAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_encoding(&self) -> EncodingRef {
        ArcRef::new_arc(Arc::new(EncodingAdapter::<V>(self.0.clone())))
    }

    fn into_encoding(self) -> EncodingRef
    where
        Self: Sized,
    {
        todo!()
    }

    fn id(&self) -> EncodingId {
        V::id(&self.0)
    }

    fn decode(
        &self,
        dtype: DType,
        len: usize,
        metadata: Option<&[u8]>,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<ArrayRef> {
        let metadata =
            <<V::SerdeVTable as SerdeVTable<V>>::Metadata as DeserializeMetadata>::deserialize(
                metadata,
            )?;
        let array = <V::SerdeVTable as SerdeVTable<V>>::decode(
            &self.0, dtype, len, &metadata, buffers, children, ctx,
        )?;
        assert_eq!(array.len(), len, "Array length mismatch after decode");
        Ok(array.to_array())
    }

    fn encode(
        &self,
        input: &Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let array = <V::EncodeVTable as EncodeVTable<V>>::encode(&self.0, input, like)?;
        Ok(array.map(|a| a.into_array()))
    }
}

impl<V: VTable> Debug for EncodingAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoding").field("id", &self.id()).finish()
    }
}

impl Display for dyn Encoding + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}

impl PartialEq for dyn Encoding + '_ {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for dyn Encoding + '_ {}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for EncodingAdapter<V> {}
}
