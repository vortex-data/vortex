// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod validity;

use std::hash::Hasher;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityVTableFromChild;
use crate::arrays::PatchedArray;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::patches::Patches;
use crate::serde::ArrayChildren;
use crate::vtable;

#[derive(Clone, Debug)]
pub struct LazyPatched;

vtable!(LazyPatched, LazyPatched, LazyPatchedData);

#[derive(Clone, prost::Message)]
pub struct LazyPatchedMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) num_patches: u32,
    #[prost(uint32, tag = "2")]
    pub(crate) offset: u32,
    #[prost(enumeration = "PType", tag = "3")]
    pub(crate) indices_ptype: i32,
}

impl LazyPatched {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.patched_lazy");
}

const INNER_SLOT: usize = 0;
const PATCH_INDICES_SLOT: usize = 1;
const PATCH_VALUES_SLOT: usize = 2;
const NUM_SLOTS: usize = 3;
const SLOT_NAMES: [&str; NUM_SLOTS] = ["inner", "patch_indices", "patch_values"];

impl VTable for LazyPatched {
    type ArrayData = LazyPatchedData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(&self, data: &Self::ArrayData, dtype: &DType, len: usize) -> VortexResult<()> {
        vortex_ensure_eq!(data.inner().len(), len);
        vortex_ensure_eq!(data.patches().dtype(), dtype);
        Ok(())
    }

    fn array_hash<H: Hasher>(array: &Self::ArrayData, state: &mut H, precision: Precision) {
        array.slots[INNER_SLOT]
            .as_ref()
            .vortex_expect("present")
            .array_hash(state, precision);
        array.slots[PATCH_INDICES_SLOT]
            .as_ref()
            .vortex_expect("present")
            .array_hash(state, precision);
        array.slots[PATCH_VALUES_SLOT]
            .as_ref()
            .vortex_expect("present")
            .array_hash(state, precision);
    }

    fn array_eq(array: &Self::ArrayData, other: &Self::ArrayData, precision: Precision) -> bool {
        array.inner().array_eq(other.inner(), precision)
            && array.patches().array_eq(&other.patches(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("LazyPatched array holds no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        vortex_panic!("LazyPatched array holds no buffers")
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        let num_patches = u32::try_from(array.num_patches())?;
        let offset = u32::try_from(array.offset)?;
        let indices_ptype = array.patch_indices_ptype() as i32;

        Ok(Some(
            LazyPatchedMetadata {
                num_patches,
                offset,
                indices_ptype,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<LazyPatchedData> {
        let metadata = LazyPatchedMetadata::decode(metadata)?;

        // Convert into PType
        let indices_ptype = PType::try_from(100i32)?;
        let num_patches = metadata.num_patches as usize;

        // Child must have expected DType.
        let inner = children.get(0, dtype, len)?;
        let patch_indices = children.get(1, &DType::from(indices_ptype), num_patches)?;
        let patch_values = children.get(2, dtype, num_patches)?;

        Ok(LazyPatchedData {
            offset: metadata.offset as usize,
            slots: vec![Some(inner), Some(patch_indices), Some(patch_values)],
        })
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(
        array: &mut Self::ArrayData,
        mut slots: Vec<Option<ArrayRef>>,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), NUM_SLOTS);

        array.slots[INNER_SLOT] = Some(
            slots
                .remove(0)
                .ok_or_else(|| vortex_err!("inner slot required"))?,
        );

        array.slots[PATCH_INDICES_SLOT] = Some(
            slots
                .remove(0)
                .ok_or_else(|| vortex_err!("patch_indices slot required"))?,
        );
        array.slots[PATCH_VALUES_SLOT] = Some(
            slots
                .remove(0)
                .ok_or_else(|| vortex_err!("patch_values slot required"))?,
        );

        Ok(())
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // Execution => actually transpose the patches, get back a `PatchedArray`.
        let patched =
            PatchedArray::from_array_and_patches(array.inner().clone(), &array.patches(), ctx)?
                .into_array();

        Ok(ExecutionResult::done(patched))
    }
}

#[derive(Debug, Clone)]
pub struct LazyPatchedData {
    /// Slots. Contains the inner, the patch_indices and patch_values.
    /// All slots must be occupied.
    pub(crate) slots: Vec<Option<ArrayRef>>,
    /// Offset into the patches.
    pub(crate) offset: usize,
}

impl LazyPatchedData {
    /// Create a new `LazyPatchedData` from an inner array and an aligned set of [`Patches`].
    ///
    /// # Errors
    ///
    /// Returns an error if the patches are not aligned to the array, i.e. the `array_len` of
    /// the patches does not equal the length of the inner array.
    pub fn try_new(inner: ArrayRef, patches: Patches) -> VortexResult<Self> {
        vortex_ensure_eq!(
            inner.len(),
            patches.array_len(),
            "Patches array_len does not match array len"
        );

        vortex_ensure_eq!(
            inner.dtype(),
            patches.dtype(),
            "Array and Patches types must match"
        );

        let offset = patches.offset();
        let slots = vec![
            Some(inner),
            Some(patches.indices().clone()),
            Some(patches.values().clone()),
        ];

        Ok(Self { slots, offset })
    }

    pub(crate) fn inner(&self) -> &ArrayRef {
        self.slots[INNER_SLOT]
            .as_ref()
            .vortex_expect("always occupied")
    }

    pub(crate) fn patch_indices_ptype(&self) -> PType {
        self.slots[PATCH_INDICES_SLOT]
            .as_ref()
            .vortex_expect("must be occupied")
            .dtype()
            .as_ptype()
    }

    pub(crate) fn patches(&self) -> Patches {
        let patch_indices = self.slots[PATCH_INDICES_SLOT]
            .clone()
            .vortex_expect("must be occupied");
        let patch_values = self.slots[PATCH_VALUES_SLOT]
            .clone()
            .vortex_expect("must be occupied");

        // SAFETY: the components are shredded from an original Patches at construction time,
        //  we are just re-assembling them without modification.
        unsafe {
            Patches::new_unchecked(
                self.inner().len(),
                self.offset,
                patch_indices,
                patch_values,
                None,
                None,
            )
        }
    }

    pub(crate) fn num_patches(&self) -> usize {
        self.slots[PATCH_INDICES_SLOT]
            .as_ref()
            .vortex_expect("must be occupied")
            .len()
    }
}
