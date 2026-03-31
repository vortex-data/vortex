// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains the VTable definitions for a Vortex encoding.

mod dyn_;
mod operations;
mod typed;
mod validity;

use std::fmt::Debug;
use std::hash::Hasher;
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
use crate::stats::ArrayStats;
use crate::validity::Validity;

/// The array [`VTable`] encapsulates logic for an Array type within Vortex.
///
/// The logic is split across several "VTable" traits to enable easier code organization than
/// simply lumping everything into a single trait.
///
/// From this [`VTable`] trait, we derive implementations for the sealed `DynArray` and [`DynVTable`]
/// traits.
///
/// The functions defined in these vtable traits will typically document their pre- and
/// post-conditions. The pre-conditions are validated inside the `DynArray` and [`DynVTable`]
/// implementations so do not need to be checked in the vtable implementations (for example, index
/// out of bounds). Post-conditions are validated after invocation of the vtable function and will
/// panic if violated.
pub trait VTable: 'static + Clone + Sized + Send + Sync + Debug {
    type ArrayData: 'static + Send + Sync + Clone + Debug + IntoArray;
    type Metadata: Debug;

    type OperationsVTable: OperationsVTable<Self>;
    type ValidityVTable: ValidityVTable<Self>;

    /// Returns the VTable from the array instance.
    ///
    // NOTE(ngates): this function is temporary while we migrate Arrays over to the unified vtable
    fn vtable(array: &Self::ArrayData) -> &Self;

    /// Returns the ID of the array.
    fn id(&self) -> ArrayId;

    /// Returns the length of the array.
    fn len(array: &Self::ArrayData) -> usize;

    /// Returns the DType of the array.
    fn dtype(array: &Self::ArrayData) -> &DType;

    /// Returns the stats set for the array.
    fn stats(array: &Self::ArrayData) -> &ArrayStats;

    /// Hashes the array contents.
    fn array_hash<H: Hasher>(array: ArrayView<'_, Self>, state: &mut H, precision: Precision);

    /// Compares two arrays of the same type for equality.
    fn array_eq(
        array: ArrayView<'_, Self>,
        other: ArrayView<'_, Self>,
        precision: Precision,
    ) -> bool;

    /// Returns the number of buffers in the array.
    fn nbuffers(array: ArrayView<'_, Self>) -> usize;

    /// Returns the buffer at the given index.
    ///
    /// # Panics
    /// Panics if `idx >= nbuffers(array)`.
    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle;

    /// Returns the name of the buffer at the given index, or `None` if unnamed.
    fn buffer_name(array: ArrayView<'_, Self>, idx: usize) -> Option<String>;

    /// Returns the number of children in the array.
    fn nchildren(array: ArrayView<'_, Self>) -> usize;

    /// Returns the child at the given index.
    ///
    /// # Panics
    /// Panics if `idx >= nchildren(array)`.
    fn child(array: ArrayView<'_, Self>, idx: usize) -> ArrayRef;

    /// Returns the name of the child at the given index.
    ///
    /// # Panics
    /// Panics if `idx >= nchildren(array)`.
    fn child_name(array: ArrayView<'_, Self>, idx: usize) -> String;

    /// Exports metadata for an array.
    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata>;

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
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let canonical = array
            .array_ref()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_array();
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
    ) -> VortexResult<Self::ArrayData>;

    /// Replaces the children in `array` with `children`.
    fn with_children(array: &mut Self::ArrayData, children: Vec<ArrayRef>) -> VortexResult<()>;

    /// Execute this array by returning an [`ExecutionResult`].
    fn execute(
        array: Arc<ArrayInner<Self>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ExecutionResult>;

    /// Attempt to execute the parent of this array.
    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        _ = (array, parent, child_idx, ctx);
        Ok(None)
    }

    /// Attempt to reduce the array to a simpler representation.
    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        _ = array;
        Ok(None)
    }

    /// Attempt to perform a reduction of the parent of this array.
    fn reduce_parent(
        array: ArrayView<'_, Self>,
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

/// vtable! macro — generates IntoArray, From, and type alias for array types.
///
/// Three forms:
/// - `vtable!(Foo)` — short for `vtable!(Foo, Foo)` (legacy form)
/// - `vtable!(Foo, FooVT)` — legacy form where `FooArray` is the inner struct name
/// - `vtable!(Foo, FooVT, FooData)` — new form where `FooData` is the inner struct,
///   and `FooArray` is generated as a type alias for `Array<FooVT>`
#[macro_export]
macro_rules! vtable {
    ($V:ident) => {
        $crate::vtable!($V, $V);
    };
    // Legacy form: FooArray is the inner struct name, no type alias generated.
    ($Base:ident, $VT:ident) => {
        $crate::aliases::paste::paste! {
            impl $crate::IntoArray for [<$Base Array>] {
                fn into_array(self) -> $crate::ArrayRef {
                    use $crate::aliases::vortex_error::VortexExpect;
                    $crate::ArrayRef::from($crate::vtable::Array::<$VT>::try_from_data(self).vortex_expect("data is always valid"))
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
    // New form: Data is the inner struct, FooArray is a type alias for ArrayInner<VT>.
    ($Base:ident, $VT:ident, $Data:ident) => {
        $crate::aliases::paste::paste! {
            /// Type alias: `FooArray = ArrayInner<Foo>`.
            pub type [<$Base Array>] = $crate::vtable::ArrayInner<$VT>;

            impl $crate::IntoArray for $Data {
                fn into_array(self) -> $crate::ArrayRef {
                    use $crate::aliases::vortex_error::VortexExpect;
                    $crate::vtable::ArrayInner::<$VT>::try_from_data(self).vortex_expect("data is always valid").into_array()
                }
            }

            impl From<$Data> for $crate::ArrayRef {
                fn from(value: $Data) -> $crate::ArrayRef {
                    use $crate::IntoArray;
                    value.into_array()
                }
            }
        }
    };
}
