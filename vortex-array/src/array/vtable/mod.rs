// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains the VTable definitions for a Vortex encoding.

mod operations;
mod validity;

use std::fmt::Debug;
use std::fmt::Formatter;

pub use operations::*;
pub use validity::*;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::ArrayView;
use crate::Canonical;
use crate::ExecutionResult;
use crate::IntoArray;
pub use crate::array::plugin::*;
use crate::arrays::ConstantArray;
use crate::arrays::constant::Constant;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::executor::ExecutionCtx;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::patches::Patches;
use crate::scalar::ScalarValue;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// The array [`VTable`] encapsulates logic for an Array type within Vortex.
///
/// The logic is split across several "VTable" traits to enable easier code organization than
/// simply lumping everything into a single trait.
///
/// From this [`VTable`] trait, we derive implementations for the sealed `DynArray` trait and the
/// public [`ArrayPlugin`] registry trait.
///
/// The functions defined in these vtable traits will typically document their pre- and
/// post-conditions. The pre-conditions are validated inside the `DynArray` and [`ArrayRef`]
/// implementations so do not need to be checked in the vtable implementations (for example, index
/// out of bounds). Post-conditions are validated after invocation of the vtable function and will
/// panic if violated.
pub trait VTable: 'static + Clone + Sized + Send + Sync + Debug {
    type ArrayData: 'static + Send + Sync + Clone + Debug + ArrayHash + ArrayEq;

    type OperationsVTable: OperationsVTable<Self>;
    type ValidityVTable: ValidityVTable<Self>;

    /// Returns the ID of the array.
    fn id(&self) -> ArrayId;

    /// Validates that externally supplied logical metadata matches the array data.
    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()>;

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
    ///
    /// The default counts non-None slots.
    fn nchildren(array: ArrayView<'_, Self>) -> usize {
        array.slots().iter().filter(|s| s.is_some()).count()
    }

    /// Returns the child at the given index.
    ///
    /// The default returns the `idx`-th non-None slot.
    ///
    /// # Panics
    /// Panics if `idx >= nchildren(array)`.
    fn child(array: ArrayView<'_, Self>, idx: usize) -> ArrayRef {
        array
            .slots()
            .iter()
            .filter_map(|s| s.clone())
            .nth(idx)
            .vortex_expect("child index out of bounds")
    }

    /// Returns the name of the child at the given index.
    ///
    /// The default returns the slot name of the `idx`-th non-None slot.
    ///
    /// # Panics
    /// Panics if `idx >= nchildren(array)`.
    fn child_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        array
            .slots()
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_some())
            .nth(idx)
            .map(|(slot_idx, _)| Self::slot_name(array, slot_idx))
            .vortex_expect("child_name index out of bounds")
    }

    /// Serialize metadata into a byte buffer for IPC or file storage.
    /// Return `None` if the array cannot be serialized.
    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>>;

    /// Formats a human-readable metadata description for display tooling.
    fn fmt_metadata(array: ArrayView<'_, Self>, f: &mut Formatter<'_>) -> std::fmt::Result {
        match Self::serialize(array) {
            Ok(Some(metadata)) if metadata.is_empty() => f.write_str("EmptyMetadata"),
            _ => Debug::fmt(array.data(), f),
        }
    }

    /// Deserialize an array from serialized components.
    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<crate::array::ArrayParts<Self>>;

    /// Writes the array into a canonical builder.
    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let canonical = array
            .array()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_array();
        builder.extend_from_array(&canonical);
        Ok(())
    }

    /// Returns the name of the slot at the given index.
    ///
    /// # Panics
    /// Panics if `idx >= slots(array).len()`.
    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String;

    /// Execute this array by returning an [`ExecutionResult`].
    ///
    /// Instead of recursively executing children, implementations should return
    /// [`ExecutionResult::execute_slot`] to request that the scheduler execute a slot first,
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
    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult>;

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

use crate::array::ArrayId;

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

/// Reconstruct a [`Validity`] from an optional child array and nullability.
///
/// This is the inverse of [`validity_to_child`].
#[inline]
pub fn child_to_validity(child: &Option<ArrayRef>, nullability: Nullability) -> Validity {
    match child {
        Some(arr) => {
            // Detect constant bool arrays created by validity_to_child.
            // Use direct ScalarValue matching to avoid expensive scalar conversion.
            if let Some(c) = arr.as_opt::<Constant>()
                && let Some(ScalarValue::Bool(val)) = c.scalar().value()
            {
                return if *val {
                    Validity::AllValid
                } else {
                    Validity::AllInvalid
                };
            }
            Validity::Array(arr.clone())
        }
        None => Validity::from(nullability),
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
            impl [<$Base Array>] {
                #[deprecated(note = "use `.into_array()` (owned) or `.clone().into_array()` (ref) to make clones explicit")]
                pub fn to_array(&self) -> $crate::ArrayRef
                where
                    Self: Clone + $crate::IntoArray,
                {
                    use $crate::IntoArray;
                    self.clone().into_array()
                }
            }
        }
    };
    // New form: Data is the inner struct, FooArray is a type alias for Array<VT>.
    ($Base:ident, $VT:ident, $Data:ident) => {
        $crate::aliases::paste::paste! {
            /// Type alias: `FooArray = Array<Foo>`.
            pub type [<$Base Array>] = $crate::Array<$VT>;
        }
    };
}
