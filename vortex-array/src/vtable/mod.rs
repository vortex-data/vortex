// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains the VTable definitions for a Vortex encoding.

mod array;
mod dyn_;
mod operations;
mod validity;
mod visitor;

use std::fmt::Debug;
use std::ops::Deref;

pub use array::*;
pub use dyn_::*;
pub use operations::*;
pub use validity::*;
pub use visitor::*;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::executor::ExecutionCtx;
use crate::serde::ArrayChildren;

/// The array [`VTable`] encapsulates logic for an Array type within Vortex.
///
/// The logic is split across several "VTable" traits to enable easier code organization than
/// simply lumping everything into a single trait.
///
/// From this [`VTable`] trait, we derive implementations for the sealed [`Array`] and [`DynVTable`]
/// traits.
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
    type OperationsVTable: OperationsVTable<Self>;
    type ValidityVTable: ValidityVTable<Self>;
    type VisitorVTable: VisitorVTable<Self>;

    /// Returns the ID of the array.
    fn id(array: &Self::Array) -> ArrayId;

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

    /// Deserialize array metadata from a byte buffer.
    ///
    /// To reduce the serialized form, arrays do not store their own DType and length. Instead,
    /// this is passed down from the parent array during deserialization. These properties are
    /// exposed here for use during deserialization.
    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata>;

    /// Writes the array into a canonical builder.
    ///
    /// ## Post-conditions
    /// - The length of the builder is incremented by the length of the input array.
    fn append_to_builder(
        array: &Self::Array,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let array = Self::execute(array, ctx)?;
        builder.extend_from_array(array.as_ref());
        Ok(())
    }

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
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array>;

    /// Replaces the children in `array` with `children`. The count must be the same and types
    /// of children must be expected.
    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()>;

    /// Execute this array to produce an [`ArrayRef`].
    ///
    /// Array execution is designed such that repeated execution of an array will eventually
    /// converge to a canonical representation. Implementations of this function should therefore
    /// ensure they make progress towards that goal.
    ///
    /// This includes fully evaluating the array, such us decoding run-end encoding, or executing
    /// one of the array's children and re-building the array with the executed child.
    ///
    /// It is recommended to only perform a single step of execution per call to this function,
    /// such that surrounding arrays have an opportunity to perform their own parent reduction
    /// or execution logic.
    ///
    /// The returned array must be logically equivalent to the input array. In other words, the
    /// recursively canonicalized forms of both arrays must be equal.
    ///
    /// Debug builds will panic if the returned array is of the wrong type, wrong length, or
    /// incorrectly contains null values.
    ///
    // TODO(ngates): in the future, we may pass a "target encoding hint" such that this array
    //  can produce a more optimal representation for the parent. This could be used to preserve
    //  varbin vs varbinview or list vs listview encodings when the parent knows it prefers
    //  one representation over another, such as when exporting to a specific Arrow array.
    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef>;

    /// Attempt to execute the parent of this array.
    ///
    /// This function allows arrays to plug in specialized execution logic for their parent. For
    /// example, strings compressed as FSST arrays can implement a custom equality comparison when
    /// the comparing against a scalar string.
    ///
    /// Returns `Ok(None)` if no specialized execution is possible.
    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        _ = (array, parent, child_idx, ctx);
        Ok(None)
    }

    /// Attempt to reduce the array to a more simple representation.
    ///
    /// Returns `Ok(None)` if no reduction is possible.
    fn reduce(array: &Self::Array) -> VortexResult<Option<ArrayRef>> {
        _ = array;
        Ok(None)
    }

    /// Attempt to perform a reduction of the parent of this array.
    ///
    /// This function allows arrays to plug in reduction rules to their parents, for example
    /// run-end arrays can pull-down scalar functions and apply them only over their values.
    ///
    /// Returns `Ok(None)` if no reduction is possible.
    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        _ = (array, parent, child_idx);
        Ok(None)
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
