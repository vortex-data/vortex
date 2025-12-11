// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains the VTable definitions for a Vortex encoding.

mod array;
mod canonical;
mod compute;
mod dyn_;
mod encode;
mod operations;
mod validity;
mod visitor;

use std::fmt::Debug;
use std::ops::Deref;

pub use array::*;
pub use canonical::*;
pub use compute::*;
pub use dyn_::*;
pub use encode::*;
pub use operations::*;
pub use validity::*;
pub use visitor::*;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::Array;
use crate::IntoArray;
use crate::VectorExecutor;
use crate::kernel::BindCtx;
use crate::kernel::KernelRef;
use crate::kernel::kernel;
use crate::serde::ArrayChildren;

/// The array [`VTable`] encapsulates logic for an Array type within Vortex.
///
/// The logic is split across several "VTable" traits to enable easier code organization than
/// simply lumping everything into a single trait.
///
/// Some of these vtables are optional, such as the [`ComputeVTable`] and [`EncodeVTable`],
/// which can be disabled by assigning to the [`NotSupported`] type.
///
/// From this [`VTable`] trait, we derive implementations for the sealed [`Array`] and [`DynVTable`]
/// traits via the [`crate::ArrayAdapter`] and [`ArrayVTableAdapter`] types respectively.
///
/// The functions defined in these vtable traits will typically document their pre- and
/// post-conditions. The pre-conditions are validated inside the [`Array`] and [`DynVTable`]
/// implementations so do not need to be checked in the vtable implementations (for example, index
/// out of bounds). Post-conditions are validated after invocation of the vtable function and will
/// panic if violated.
pub trait VTable: 'static + Sized + Send + Sync + Debug {
    type Array: 'static + Send + Sync + Clone + Debug + Deref<Target = dyn Array> + IntoArray;
    type Metadata: Debug;

    type ArrayVTable: BaseArrayVTable<Self>;
    type CanonicalVTable: CanonicalVTable<Self>;
    type OperationsVTable: OperationsVTable<Self>;
    type ValidityVTable: ValidityVTable<Self>;
    type VisitorVTable: VisitorVTable<Self>;

    /// Optionally enable implementing dynamic compute dispatch for this encoding.
    /// Can be disabled by assigning to the [`NotSupported`] type.
    type ComputeVTable: ComputeVTable<Self>;
    /// Optionally enable the [`EncodeVTable`] for this encoding. This allows it to partake in
    /// compression.
    /// Can be disabled by assigning to the [`NotSupported`] type.
    type EncodeVTable: EncodeVTable<Self>;

    /// Returns the ID of the encoding.
    fn id(&self) -> ArrayId;

    /// Returns the encoding for the array.
    fn encoding(array: &Self::Array) -> ArrayVTable;

    /// Exports metadata for an array.
    ///
    /// All other parts of the array are exported using the [`crate::vtable::VisitorVTable`].
    ///
    /// * If the array does not contain metadata, it should return
    ///   [`crate::metadata::EmptyMetadata`].
    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata>;

    /// Serialize metadata into a byte buffer for IPC or file storage.
    /// Return `None` if the array cannot be serialized.
    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>>;

    /// Deserialize metadata from a byte buffer.
    fn deserialize(bytes: &[u8]) -> VortexResult<Self::Metadata>;

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
    // TODO(ngates): take the parts by ownership, since most arrays need them anyway
    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array>;

    /// Bind this array into a [`KernelRef`] for CPU execution.
    ///
    /// The returned vector must be the appropriate one for the array's logical type (they are
    /// one-to-one with Vortex `DType`s), and should respect the output nullability of the array.
    ///
    /// Debug builds will panic if the returned vector is of the wrong type, wrong length, or
    /// incorrectly contains null values.
    ///
    /// Implementations should recursively call [`Array::bind_kernel`] on child
    /// arrays as needed.
    fn bind_kernel(array: &Self::Array, ctx: &mut BindCtx) -> VortexResult<KernelRef> {
        // TODO(ngates): convert arrays to canonicalize over vectors.
        let array = array.clone();
        let session = ctx.session().clone();
        Ok(kernel(move || {
            let canonical = Self::CanonicalVTable::canonicalize(&array);
            canonical.into_array().execute_vector(&session)
        }))
    }
}

/// Placeholder type used to indicate when a particular vtable is not supported by the encoding.
pub struct NotSupported;

#[macro_export]
macro_rules! vtable {
    ($V:ident) => {
        $crate::aliases::paste::paste! {
            impl AsRef<dyn $crate::Array> for [<$V Array>] {
                fn as_ref(&self) -> &dyn $crate::Array {
                    // We can unsafe cast ourselves to an ArrayAdapter.
                    unsafe { &*(self as *const [<$V Array>] as *const $crate::ArrayAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V Array>] {
                type Target = dyn $crate::Array;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an ArrayAdapter.
                    unsafe { &*(self as *const [<$V Array>] as *const $crate::ArrayAdapter<[<$V VTable>]>) }
                }
            }

            impl $crate::IntoArray for [<$V Array>] {
                fn into_array(self) -> $crate::ArrayRef {
                    // We can unsafe transmute ourselves to an ArrayAdapter.
                    std::sync::Arc::new(unsafe { std::mem::transmute::<[<$V Array>], $crate::ArrayAdapter::<[<$V VTable>]>>(self) })
                }
            }

            impl From<[<$V Array>]> for $crate::ArrayRef {
                fn from(value: [<$V Array>]) -> $crate::ArrayRef {
                    use $crate::IntoArray;
                    value.into_array()
                }
            }
        }
    };
}
