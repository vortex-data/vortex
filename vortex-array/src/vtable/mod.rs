// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains the VTable definitions for a Vortex encoding.

mod dyn_;
mod operations;
mod validity;

use std::fmt::Debug;
use std::hash::Hasher;
use std::ops::Deref;
use std::sync::Arc;

pub use dyn_::*;
pub use operations::*;
pub use validity::*;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::DynArray;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::arrays::ConstantArray;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::executor::ExecutionCtx;
use crate::patches::Patches;
use crate::serde::ArrayChildren;
use crate::stats::StatsSetRef;
use crate::validity::Validity;

/// The array [`VTable`] encapsulates logic for an Array type within Vortex.
///
/// The logic is split across several "VTable" traits to enable easier code organization than
/// simply lumping everything into a single trait.
///
/// From this [`VTable`] trait, we derive implementations for the sealed [`DynArray`] and [`DynVTable`]
/// traits.
///
/// The functions defined in these vtable traits will typically document their pre- and
/// post-conditions. The pre-conditions are validated inside the [`DynArray`] and [`DynVTable`]
/// implementations so do not need to be checked in the vtable implementations (for example, index
/// out of bounds). Post-conditions are validated after invocation of the vtable function and will
/// panic if violated.
pub trait VTable: 'static + Clone + Sized + Send + Sync + Debug {
    type Array: 'static + Send + Sync + Clone + Debug + Deref<Target = dyn DynArray> + IntoArray;
    type Metadata: Debug;

    type OperationsVTable: OperationsVTable<Self>;
    type ValidityVTable: ValidityVTable<Self>;

    /// Returns the VTable from the array instance.
    ///
    // NOTE(ngates): this function is temporary while we migrate Arrays over to the unified vtable
    fn vtable(array: &Self::Array) -> &Self;

    /// Returns the ID of the array.
    fn id(&self) -> ArrayId;

    /// Returns the length of the array.
    fn len(array: &Self::Array) -> usize;

    /// Returns the DType of the array.
    fn dtype(array: &Self::Array) -> &DType;

    /// Returns the stats set for the array.
    fn stats(array: &Self::Array) -> StatsSetRef<'_>;

    /// Hashes the array contents.
    fn array_hash<H: Hasher>(array: &Self::Array, state: &mut H, precision: Precision);

    /// Compares two arrays of the same type for equality.
    fn array_eq(array: &Self::Array, other: &Self::Array, precision: Precision) -> bool;

    /// Returns the number of buffers in the array.
    fn nbuffers(array: &Self::Array) -> usize;

    /// Returns the buffer at the given index.
    ///
    /// # Panics
    /// Panics if `idx >= nbuffers(array)`.
    fn buffer(array: &Self::Array, idx: usize) -> BufferHandle;

    /// Returns the name of the buffer at the given index, or `None` if unnamed.
    fn buffer_name(array: &Self::Array, idx: usize) -> Option<String>;

    /// Returns the number of children in the array.
    fn nchildren(array: &Self::Array) -> usize;

    /// Returns the child at the given index.
    ///
    /// # Panics
    /// Panics if `idx >= nchildren(array)`.
    fn child(array: &Self::Array, idx: usize) -> ArrayRef;

    /// Returns the name of the child at the given index.
    ///
    /// # Panics
    /// Panics if `idx >= nchildren(array)`.
    fn child_name(array: &Self::Array, idx: usize) -> String;

    /// Exports metadata for an array.
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
        _buffers: &[BufferHandle],
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
        let canonical = array.to_array().execute::<Canonical>(ctx)?.into_array();
        builder.extend_from_array(&canonical);
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

    /// Execute this array by returning an [`ExecutionResult`] that tells the scheduler what to
    /// do next.
    ///
    /// Instead of recursively executing children, implementations should return
    /// [`ExecutionResult::execute_child`] to request that the scheduler execute a child first,
    /// or [`ExecutionResult::done`] when the encoding can produce a result directly.
    ///
    /// Array execution is designed such that repeated execution of an array will eventually
    /// converge to a canonical representation. Implementations of this function should therefore
    /// ensure they make progress towards that goal.
    ///
    /// The returned array (in `Done`) must be logically equivalent to the input array. In other
    /// words, the recursively canonicalized forms of both arrays must be equal.
    ///
    /// Debug builds will panic if the returned array is of the wrong type, wrong length, or
    /// incorrectly contains null values.
    fn execute(array: Arc<Self::Array>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult>;

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

/// Returns the validity as a child array if it produces one.
///
/// - `NonNullable` and `AllValid` produce no child (returns `None`)
/// - `AllInvalid` produces a `ConstantArray` of `false` values
/// - `Array` returns the validity array
#[inline]
pub fn validity_to_child(validity: &Validity, len: usize) -> Option<ArrayRef> {
    match validity {
        Validity::NonNullable | Validity::AllValid => None,
        Validity::AllInvalid => Some(ConstantArray::new(false, len).into_array()),
        Validity::Array(array) => Some(array.clone()),
    }
}

/// Returns 1 if validity produces a child, 0 otherwise.
#[inline]
pub fn validity_nchildren(validity: &Validity) -> usize {
    match validity {
        Validity::NonNullable | Validity::AllValid => 0,
        Validity::AllInvalid | Validity::Array(_) => 1,
    }
}

/// Returns the number of children produced by patches.
#[inline]
pub fn patches_nchildren(patches: &Patches) -> usize {
    2 + patches.chunk_offsets().is_some() as usize
}

/// Returns the child at the given index within a patches component.
///
/// Index 0 = patch_indices, 1 = patch_values, 2 = patch_chunk_offsets (if present).
#[inline]
pub fn patches_child(patches: &Patches, idx: usize) -> ArrayRef {
    match idx {
        0 => patches.indices().clone(),
        1 => patches.values().clone(),
        2 => patches
            .chunk_offsets()
            .as_ref()
            .vortex_expect("patch_chunk_offsets child out of bounds")
            .clone(),
        _ => vortex_panic!("patches child index {idx} out of bounds"),
    }
}

/// Returns the name of the child at the given index within a patches component.
#[inline]
pub fn patches_child_name(idx: usize) -> &'static str {
    match idx {
        0 => "patch_indices",
        1 => "patch_values",
        2 => "patch_chunk_offsets",
        _ => vortex_panic!("patches child name index {idx} out of bounds"),
    }
}

#[macro_export]
macro_rules! vtable {
    ($V:ident) => {
        $crate::vtable!($V, $V);
    };
    ($Base:ident, $VT:ident) => {
        $crate::aliases::paste::paste! {
            impl AsRef<dyn $crate::DynArray> for [<$Base Array>] {
                fn as_ref(&self) -> &dyn $crate::DynArray {
                    // We can unsafe cast ourselves to an ArrayAdapter.
                    unsafe { &*(self as *const [<$Base Array>] as *const $crate::ArrayAdapter<$VT>) }
                }
            }

            impl std::ops::Deref for [<$Base Array>] {
                type Target = dyn $crate::DynArray;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an ArrayAdapter.
                    unsafe { &*(self as *const [<$Base Array>] as *const $crate::ArrayAdapter<$VT>) }
                }
            }

            impl $crate::IntoArray for [<$Base Array>] {
                fn into_array(self) -> $crate::ArrayRef {
                    // We can unsafe transmute ourselves to an ArrayAdapter.
                    std::sync::Arc::new(unsafe { std::mem::transmute::<[<$Base Array>], $crate::ArrayAdapter::<$VT>>(self) })
                }
            }

            impl From<[<$Base Array>]> for $crate::ArrayRef {
                fn from(value: [<$Base Array>]) -> $crate::ArrayRef {
                    use $crate::IntoArray;
                    value.into_array()
                }
            }

            impl [<$Base Array>] {
                #[deprecated(note = "use `.into_array()` (owned) or `.clone().into_array()` (ref) to make clones explicit")]
                pub fn to_array(&self) -> $crate::ArrayRef {
                    use $crate::IntoArray;
                    self.clone().into_array()
                }

                /// Upcasts an `Arc<Self>` to an [`ArrayRef`] without cloning.
                pub fn into_array_ref(self: std::sync::Arc<Self>) -> $crate::ArrayRef {
                    // SAFETY: ArrayAdapter<V> is #[repr(transparent)] over V::Array,
                    // so Arc<V::Array> and Arc<ArrayAdapter<V>> have identical layout.
                    let raw = std::sync::Arc::into_raw(self) as *const $crate::ArrayAdapter<$VT>;
                    unsafe { std::sync::Arc::from_raw(raw) }
                }
            }
        }
    };
}
