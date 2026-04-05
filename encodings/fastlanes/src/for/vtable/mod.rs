// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
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
use vortex_array::vtable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::FoRData;
use crate::r#for::array::FoRArrayExt;
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

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let encoded = slots[0].as_ref().vortex_expect("FoRArray encoded slot");
        FoRData::validate_parts(encoded, &data.reference, dtype, len)
    }

    fn array_hash<H: std::hash::Hasher>(data: &FoRData, state: &mut H, _precision: Precision) {
        data.reference.hash(state);
    }

    fn array_eq(data: &FoRData, other: &FoRData, _precision: Precision) -> bool {
        data.reference == other.reference
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        // Note that we **only** serialize the optional scalar value (not including the dtype).
        Ok(Some(ScalarValue::to_proto_bytes(
            array.reference_scalar().value(),
        )))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.is_empty(),
            "FoRArray expects 0 buffers, got {}",
            buffers.len()
        );
        if children.len() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                children.len()
            )
        }

        let scalar_value = ScalarValue::from_proto_bytes(metadata, dtype, session)?;
        let reference = Scalar::try_new(dtype.clone(), scalar_value)?;
        let encoded = children.get(0, dtype, len)?;
        let slots = vec![Some(encoded.clone())];

        let data = FoRData::try_new(encoded, reference)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
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
        vortex_ensure!(!reference.is_null(), "Reference value cannot be null");
        let dtype = reference
            .dtype()
            .with_nullability(encoded.dtype().nullability());
        let reference = reference.cast(&dtype)?;
        let len = encoded.len();
        let slots = vec![Some(encoded.clone())];
        let data = FoRData::try_new(encoded, reference)?;
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(FoR, dtype, len, data).with_slots(slots))
        })
    }

    /// Encode a primitive array using Frame of Reference encoding.
    pub fn encode(array: PrimitiveArray) -> VortexResult<FoRArray> {
        FoRData::encode(array)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::scalar::ScalarValue;
    use vortex_array::test_harness::check_metadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_for_metadata() {
        let metadata: Vec<u8> = ScalarValue::to_proto_bytes(Some(&ScalarValue::from(i64::MAX)));
        check_metadata("for.metadata", &metadata);
    }
}
