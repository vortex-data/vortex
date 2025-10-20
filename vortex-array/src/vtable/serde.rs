// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::DeserializeMetadata;
use crate::serde::ArrayChildren;
use crate::vtable::{NotSupported, VTable};

/// VTable trait for building an array from its serialized components.
///
/// # Guarantees
pub trait SerdeVTable<V: VTable> {
    /// Build an array from components.
    ///
    /// This is called on the file and IPC deserialization pathways, to reconstruct the array from
    /// type-erased components.
    ///
    /// Encoding implementers should take note that all validation necessary to ensure the encoding
    /// is safe to read should happen inside of this method.
    ///
    /// # Safety and correctness
    ///
    /// This method should *never* panic, it must always return an error or else it returns a
    /// valid `Array` that meets all the encoding's preconditions.
    ///
    /// For example, the `build` implementation for a dictionary encoding should ensure that all
    /// codes lie in the valid range. For a UTF-8 array, it should check the bytes to ensure they
    /// are all valid string data bytes. Any corrupt files or malformed data buffers should be
    /// caught here, before returning the deserialized array.
    ///
    /// # Validation
    ///
    /// Validation is mainly meant to ensure that all internal pointers in the encoding reference
    /// valid ranges of data, and that all data conforms to its DType constraints. These ensure
    /// that no array operations will panic at runtime, or yield undefined behavior when unsafe
    /// operations like `get_unchecked` use indices in the array buffer.
    ///
    /// Examples of the kinds of validation that should be part of the `build` step:
    ///
    /// * Checking that any offsets buffers point to valid offsets in some other child array
    /// * Checking that any buffers for data or validity have the appropriate size for the
    ///   encoding
    /// * Running UTF-8 validation for any buffers that are expected to hold flat UTF-8 data
    fn build(
        encoding: &V::Encoding,
        dtype: &DType,
        len: usize,
        metadata: &<V::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<V::Array>;
}

impl<V: VTable> SerdeVTable<V> for NotSupported {
    fn build(
        encoding: &V::Encoding,
        _dtype: &DType,
        _len: usize,
        _metadata: &<V::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<V::Array> {
        vortex_bail!("Serde not supported by {} encoding", V::id(encoding));
    }
}
