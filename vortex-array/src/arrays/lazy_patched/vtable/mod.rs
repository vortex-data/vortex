// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod validity;

use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::DeserializeMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::Precision;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityVTableFromChild;
use crate::arrays::PatchedArray;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::patches::Patches;
use crate::serde::ArrayChildren;
use crate::stats::ArrayStats;
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
    type Metadata = ProstMetadata<LazyPatchedMetadata>;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &LazyPatched
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &Self::ArrayData) -> usize {
        array.inner().len()
    }

    fn dtype(array: &Self::ArrayData) -> &DType {
        array.inner().dtype()
    }

    fn stats(array: &Self::ArrayData) -> &ArrayStats {
        &array.stats
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

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        let num_patches = u32::try_from(array.num_patches())?;
        let offset = u32::try_from(array.offset)?;

        Ok(ProstMetadata(LazyPatchedMetadata {
            num_patches,
            offset,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let deserialized = <Self::Metadata>::deserialize(bytes)?;
        Ok(ProstMetadata(deserialized))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ArrayRef> {
        // There should be 3 children
        // 1. inner
        // 2. patch_indices
        // 3. patch_values
        vortex_ensure!(
            children.len() == 3,
            "expected exactly 3 children from LazyPatched, found {}",
            children.len()
        );

        let inner = children.get(0, dtype, len)?;

        let num_patches = metadata.num_patches as usize;
        let offset = metadata.offset as usize;
        let patch_indices = children.get(1, dtype, num_patches)?;
        let patch_values = children.get(2, dtype, num_patches)?;

        let slots = vec![Some(inner), Some(patch_indices), Some(patch_values)];

        Ok(LazyPatchedData {
            slots,
            offset,
            stats: ArrayStats::default(),
        }
        .into_array())
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

    pub(crate) stats: ArrayStats,
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

        Ok(Self {
            slots,
            offset,
            stats: ArrayStats::default(),
        })
    }

    pub(crate) fn inner(&self) -> &ArrayRef {
        self.slots[INNER_SLOT]
            .as_ref()
            .vortex_expect("always occupied")
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
