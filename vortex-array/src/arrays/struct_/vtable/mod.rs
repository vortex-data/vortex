// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use kernel::PARENT_KERNELS;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::struct_::StructData;
use crate::arrays::struct_::array::FIELDS_OFFSET;
use crate::arrays::struct_::array::VALIDITY_SLOT;
use crate::arrays::struct_::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
mod kernel;
mod operations;
mod validity;

use crate::Precision;
use crate::array::ArrayId;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;

vtable!(Struct, Struct, StructData);

impl VTable for Struct {
    type ArrayData = StructData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn array_hash<H: std::hash::Hasher>(array: ArrayView<'_, Self>, state: &mut H, precision: Precision) {
        for field in array.iter_unmasked_fields() {
            field.array_hash(state, precision);
        }
        array.data().validity().array_hash(state, precision);
    }

    fn array_eq(array: ArrayView<'_, Self>, other: ArrayView<'_, Self>, precision: Precision) -> bool {
        array.slots.len() == other.slots.len()
            && array
                .iter_unmasked_fields()
                .zip(other.iter_unmasked_fields())
                .all(|(a, b)| a.array_eq(b, precision))
            && array.data().validity().array_eq(&other.data().validity(), precision)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn validate(&self, data: &StructData, dtype: &DType, len: usize, slots: &[Option<ArrayRef>]) -> VortexResult<()> {
        match dtype {
            DType::Struct(..) => {}
            _ => vortex_bail!("Expected struct dtype, found {:?}", dtype),
        }
        if data.len() != len {
            vortex_bail!(
                InvalidArgument: "StructArray length {} does not match outer length {}",
                data.len(),
                len
            );
        }
        let data_dtype = data.dtype();
        if &data_dtype != dtype {
            vortex_bail!(
                InvalidArgument: "StructArray dtype {} does not match outer dtype {}",
                data_dtype,
                dtype
            );
        }
        Ok(())
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("StructArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("StructArray buffer_name index {idx} out of bounds")
    }

    fn serialize(_array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],

        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<StructData> {
        if !metadata.is_empty() {
            vortex_bail!(
                "StructArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        let DType::Struct(struct_dtype, nullability) = dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", dtype)
        };

        let (validity, non_data_children) = if children.len() == struct_dtype.nfields() {
            (Validity::from(*nullability), 0_usize)
        } else if children.len() == struct_dtype.nfields() + 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            (Validity::Array(validity), 1_usize)
        } else {
            vortex_bail!(
                "Expected {} or {} children, found {}",
                struct_dtype.nfields(),
                struct_dtype.nfields() + 1,
                children.len()
            );
        };

        let field_children: Vec<_> = (0..struct_dtype.nfields())
            .map(|i| {
                let child_dtype = struct_dtype
                    .field_by_index(i)
                    .vortex_expect("no out of bounds");
                children.get(non_data_children + i, &child_dtype, len)
            })
            .try_collect()?;

        StructData::try_new_with_dtype(field_children, struct_dtype.clone(), len, validity)
    }

    fn infer_slots(data: &Self::ArrayData) -> Vec<Option<ArrayRef>> {
        data.slots.clone()
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        array.slots()
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx == VALIDITY_SLOT {
            "validity".to_string()
        } else {
            array.names()[idx - FIELDS_OFFSET].to_string()
        }
    }


    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
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
pub struct Struct;

impl Struct {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.struct");
}
