// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::FoRData;
use crate::r#for::array::NUM_SLOTS;
use crate::r#for::array::SLOT_NAMES;
use crate::r#for::array::for_decompress::decompress;
use crate::r#for::vtable::kernels::PARENT_KERNELS;
use crate::r#for::vtable::rules::PARENT_RULES;

mod kernels;
mod operations;
mod rules;
mod slice;
mod validity;

vtable!(FoR, FoR, FoRData);

impl VTable for FoR {
    type ArrayData = FoRData;

    type Metadata = Scalar;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &FoRData) -> &Self {
        &FoR
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &FoRData) -> usize {
        array.encoded().len()
    }

    fn dtype(array: &FoRData) -> &DType {
        array.reference_scalar().dtype()
    }

    fn stats(array: &FoRData) -> &ArrayStats {
        array.stats_set()
    }

    fn array_hash<H: std::hash::Hasher>(array: &FoRData, state: &mut H, precision: Precision) {
        array.encoded().array_hash(state, precision);
        array.reference_scalar().hash(state);
    }

    fn array_eq(array: &FoRData, other: &FoRData, precision: Precision) -> bool {
        array.encoded().array_eq(other.encoded(), precision)
            && array.reference_scalar() == other.reference_scalar()
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("FoRArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "FoRArray expects exactly {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(array.reference_scalar().clone())
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        // Note that we **only** serialize the optional scalar value (not including the dtype).
        Ok(Some(ScalarValue::to_proto_bytes(metadata.value())))
    }

    fn deserialize(
        bytes: &[u8],
        dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let scalar_value = ScalarValue::from_proto_bytes(bytes, dtype, session)?;
        Scalar::try_new(dtype.clone(), scalar_value)
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FoRData> {
        if children.len() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                children.len()
            )
        }

        let encoded = children.get(0, dtype, len)?;

        FoRData::try_new(encoded, metadata.clone())
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(decompress(&array, ctx)?.into_array()))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }
}

#[derive(Clone, Debug)]
pub struct FoR;

impl FoR {
    pub const ID: ArrayId = ArrayId::new_ref("fastlanes.for");

    /// Construct a new FoR array from an encoded array and a reference scalar.
    pub fn try_new(encoded: ArrayRef, reference: Scalar) -> VortexResult<FoRArray> {
        Array::try_from_data(FoRData::try_new(encoded, reference)?)
    }

    /// Encode a primitive array using Frame of Reference encoding.
    pub fn encode(array: PrimitiveArray) -> VortexResult<FoRArray> {
        FoRData::encode(array)
    }
}
