//! Traits and types to define shared unique encoding identifiers.

use std::fmt::Debug;

use crate::arcref::ArcRef;
use crate::vtable::{ComputeVTable, EncodingVTable, VTableRef};
use crate::{Array, DeserializeMetadata, SerializeMetadata};

/// EncodingId is a globally unique name of the array's encoding.
pub type EncodingId = ArcRef<str>;

/// Marker trait for array encodings with their associated Array type.
pub trait Encoding: 'static + Send + Sync + EncodingVTable + ComputeVTable {
    type Array: Array;
    type Metadata: SerializeMetadata + DeserializeMetadata + Debug;

    fn vtable(&'static self) -> VTableRef
    where
        Self: Sized,
    {
        VTableRef::new_ref(self)
    }
}
