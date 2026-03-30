// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains the VTable definitions for a Vortex encoding.

mod dyn_;
mod operations;
mod typed;
mod validity;

use std::fmt::Debug;
use std::hash::Hasher;
use std::ops::Deref;
use std::sync::Arc;

pub use dyn_::*;
pub use operations::*;
pub use typed::*;
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
    fn metadata(array: &Self::Array) -> VortexResult<Self::Metadata>;

    /// Serialize metadata into a byte buffer for IPC or file storage.
    /// Return `None` if the array cannot be serialized.
    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>>;

    /// Deserialize array metadata from a byte buffer.
    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata>;

    /// Writes the array into a canonical builder.
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
    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<Self::Array>;

    /// Replaces the children in `array` with `children`.
    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()>;

    /// Execute this array by returning an [`ExecutionResult`].
    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult>;

    /// Attempt to execute the parent of this array.
    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        _ = (array, parent, child_idx, ctx);
        Ok(None)
    }

    /// Attempt to reduce the array to a simpler representation.
    fn reduce(array: &Array<Self>) -> VortexResult<Option<ArrayRef>> {
        _ = array;
        Ok(None)
    }

    /// Attempt to perform a reduction of the parent of this array.
    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        _ = (array, parent, child_idx);
        Ok(None)
    }
}

/// Alias for migration — downstream code can start using `ArrayVTable`.
pub use VTable as ArrayVTable;

/// Placeholder type used to indicate when a particular vtable is not supported by the encoding.
pub struct NotSupported;

/// Returns the validity as a child array if it produces one.
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

/// vtable! macro — generates IntoArray, From, Deref, AsRef for inner array types.
///
/// During the migration, IntoArray creates [`Array<V>`] (the new typed wrapper) while
/// Deref/AsRef go through AlsoArrayAdapter for backward-compatible DynArray access.
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
                    use $crate::vtable::VTable;
                    let vtable = $VT::vtable(&self).clone();
                    let dtype = $VT::dtype(&self).clone();
                    let len = $VT::len(&self);
                    let stats = $VT::stats(&self).to_array_stats();
                    // SAFETY: dtype and len are extracted from `self` via VTable methods.
                    std::sync::Arc::new(unsafe {
                        $crate::vtable::Array::new_unchecked(vtable, dtype, len, self, stats)
                    })
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
            }
        }
    };
}
