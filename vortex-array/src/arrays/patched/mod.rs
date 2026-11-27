// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A primitive base array with sorted replacement patches.

mod compute;
mod execute;
mod plugin;
#[cfg(test)]
mod tests;

use std::env;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::LazyLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::ArrayVTable;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Slice;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::arrays::scalar_fn::ScalarFnValidity;
use crate::arrays::slice::SliceExecuteAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::patches::PatchSlotIndices;
use crate::patches::Patches;
use crate::patches::PatchesData;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::builtin::compare::Compare;
use crate::scalar_fn::builtin::compare::CompareExecuteAdaptor;

/// Sorted patches over a base array of the same logical type.
#[derive(Clone, Debug)]
pub struct Patched;

/// An array using the sorted patch encoding.
pub type PatchedArray = Array<Patched>;

/// Strategy families can grow independently, e.g. numeric addition or integer bit composition.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum PatchStrategy {
    /// Operations on integer values.
    Integer(IntegerPatchStrategy),
    /// Operations on floating-point values.
    Float(FloatPatchStrategy),
}

/// Operations available for integer patches.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, prost::Enumeration)]
#[repr(i32)]
#[non_exhaustive]
pub enum IntegerPatchStrategy {
    /// Replace both value and validity.
    Replace = 0,
}

/// Operations available for floating-point patches.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, prost::Enumeration)]
#[repr(i32)]
#[non_exhaustive]
pub enum FloatPatchStrategy {
    /// Replace both value and validity.
    Replace = 0,
}

impl PatchStrategy {
    fn validate(self, dtype: &DType) -> VortexResult<()> {
        let supported = match self {
            Self::Integer(IntegerPatchStrategy::Replace) => dtype.is_int(),
            Self::Float(FloatPatchStrategy::Replace) => dtype.is_float(),
        };
        vortex_ensure!(
            supported,
            "Patch strategy {self:?} does not support {dtype}"
        );
        Ok(())
    }

    fn replace(dtype: &DType) -> VortexResult<Self> {
        if dtype.is_int() {
            Ok(Self::Integer(IntegerPatchStrategy::Replace))
        } else if dtype.is_float() {
            Ok(Self::Float(FloatPatchStrategy::Replace))
        } else {
            vortex_bail!("Unsupported patched dtype {dtype}")
        }
    }
}

/// Scalar metadata accompanying the four child arrays.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PatchedData {
    patches: PatchesData,
    strategy: PatchStrategy,
}

impl fmt::Display for PatchedData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}, offset={}", self.strategy, self.patches.offset())
    }
}

impl ArrayHash for PatchedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _mode: EqMode) {
        self.hash(state);
    }
}

impl ArrayEq for PatchedData {
    fn array_eq(&self, other: &Self, _mode: EqMode) -> bool {
        self == other
    }
}

#[array_slots(Patched)]
pub struct PatchedSlots {
    /// Base array, also the spine of a fused decode.
    #[slot(0)]
    pub inner: ArrayRef,
    /// Strictly increasing unsigned patch positions.
    #[slot(1)]
    pub patch_indices: ArrayRef,
    /// Replacement values and their validity.
    #[slot(2)]
    pub patch_values: ArrayRef,
    /// Starting patch position for each physical patch chunk.
    #[slot(3)]
    pub patch_chunk_offsets: ArrayRef,
}

const PATCH_SLOTS: PatchSlotIndices = PatchSlotIndices {
    indices: PatchedSlots::PATCH_INDICES,
    values: PatchedSlots::PATCH_VALUES,
    chunk_offsets: PatchedSlots::PATCH_CHUNK_OFFSETS,
};

/// Access the patch description on an owned or borrowed array.
pub trait PatchedArrayExt: TypedArrayRef<Patched> {
    /// The original sorted patch children and their slice offsets.
    fn patches(&self) -> Patches {
        PatchesData::patches_from_slots(Some(&self.patches), self.len(), self.slots(), PATCH_SLOTS)
            .vortex_expect("Patched has required patch slots")
    }

    /// How patch values combine with the base.
    fn strategy(&self) -> PatchStrategy {
        self.strategy
    }
}

impl<T: TypedArrayRef<Patched>> PatchedArrayExt for T {}

impl Patched {
    /// Replace the selected base rows, including their validity.
    pub fn from_array_and_patches(
        inner: ArrayRef,
        patches: &Patches,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PatchedArray> {
        let strategy = PatchStrategy::replace(inner.dtype())?;
        Self::with_strategy(inner, patches, strategy, ctx)
    }

    /// Construct a patched array with an explicit dtype-specific strategy.
    pub fn with_strategy(
        inner: ArrayRef,
        patches: &Patches,
        strategy: PatchStrategy,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PatchedArray> {
        let dtype = inner.dtype().clone();
        strategy.validate(&dtype)?;
        vortex_ensure_eq!(inner.len(), patches.array_len());
        vortex_ensure!(dtype.eq_with_nullability_superset(patches.dtype()));
        let patches = patches.clone().with_chunk_offsets(ctx)?;
        let data = PatchedData {
            patches: PatchesData::from_patches(&patches),
            strategy,
        };
        let slots = PatchedSlots {
            inner,
            patch_indices: patches.indices().clone(),
            patch_values: patches.values().cast(dtype.clone())?,
            patch_chunk_offsets: patches
                .chunk_offsets()
                .clone()
                .vortex_expect("offsets supplied"),
        };
        Array::try_from_parts(ArrayParts::new(
            Self,
            dtype,
            patches.array_len(),
            data,
            slots.into_slots(),
        ))
    }
}

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(ScalarFnValidity.id(), Patched, compute::PatchValidity);
    kernels.register_execute_parent_kernel(Slice.id(), Patched, SliceExecuteAdaptor(Patched));
    kernels.register_execute_parent_kernel(Filter.id(), Patched, FilterExecuteAdaptor(Patched));
    kernels.register_execute_parent_kernel(Dict.id(), Patched, TakeExecuteAdaptor(Patched));
    kernels.register_execute_parent_kernel(Compare.id(), Patched, CompareExecuteAdaptor(Patched));
}

/// Flag indicating if experimental patched array support is enabled.
///
/// This is set using the environment variable `VORTEX_EXPERIMENTAL_PATCHED_ARRAY`.
///
/// When this is true, any arrays with interior `Patches` will be read as a `Patched`
/// array, and eliminate the interior patches.
///
/// The builtin compressor will also generate Patched arrays.
pub fn use_experimental_patches() -> bool {
    static USE_EXPERIMENTAL_PATCHES: LazyLock<bool> =
        LazyLock::new(|| env::var("VORTEX_EXPERIMENTAL_PATCHED_ARRAY").is_ok_and(|v| v == "1"));
    *USE_EXPERIMENTAL_PATCHES
}
