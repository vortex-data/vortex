// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use kernel::PARENT_KERNELS;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::arrays::StructArray;
use crate::arrays::struct_::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::Array;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;
mod kernel;
mod operations;
mod validity;
use std::hash::Hash;

use crate::Precision;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::ArrayId;

vtable!(Struct);

impl VTable for Struct {
    type Array = StructArray;

    type Metadata = EmptyMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    fn vtable(_array: &Self::Array) -> &Self {
        &Struct
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &StructArray) -> usize {
        array.len
    }

    fn dtype(array: &StructArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &StructArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &StructArray, state: &mut H, precision: Precision) {
        array.len.hash(state);
        array.dtype.hash(state);
        for field in array.fields.iter() {
            field.array_hash(state, precision);
        }
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &StructArray, other: &StructArray, precision: Precision) -> bool {
        array.len == other.len
            && array.dtype == other.dtype
            && array.fields.len() == other.fields.len()
            && array
                .fields
                .iter()
                .zip(other.fields.iter())
                .all(|(a, b)| a.array_eq(b, precision))
            && array.validity.array_eq(&other.validity, precision)
    }

    fn nbuffers(_array: &StructArray) -> usize {
        0
    }

    fn buffer(_array: &StructArray, idx: usize) -> BufferHandle {
        vortex_panic!("StructArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &StructArray, idx: usize) -> Option<String> {
        vortex_panic!("StructArray buffer_name index {idx} out of bounds")
    }

    fn nchildren(array: &StructArray) -> usize {
        validity_nchildren(&array.validity) + array.unmasked_fields().len()
    }

    fn child(array: &StructArray, idx: usize) -> ArrayRef {
        let vc = validity_nchildren(&array.validity);
        if idx < vc {
            validity_to_child(&array.validity, array.len())
                .vortex_expect("StructArray validity child out of bounds")
        } else {
            array.unmasked_fields()[idx - vc].clone()
        }
    }

    fn child_name(array: &StructArray, idx: usize) -> String {
        let vc = validity_nchildren(&array.validity);
        if idx < vc {
            "validity".to_string()
        } else {
            array.names()[idx - vc].as_ref().to_string()
        }
    }

    fn metadata(_array: &StructArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        _bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<StructArray> {
        let DType::Struct(struct_dtype, nullability) = dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", dtype)
        };

        let (validity, non_data_children) = if children.len() == struct_dtype.nfields() {
            (Validity::from(*nullability), 0_usize)
        } else if children.len() == struct_dtype.nfields() + 1 {
            // Validity is the first child if it exists.
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

        let children: Vec<_> = (0..struct_dtype.nfields())
            .map(|i| {
                let child_dtype = struct_dtype
                    .field_by_index(i)
                    .vortex_expect("no out of bounds");
                children.get(non_data_children + i, &child_dtype, len)
            })
            .try_collect()?;

        StructArray::try_new_with_dtype(children, struct_dtype.clone(), len, validity)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        let DType::Struct(struct_dtype, _nullability) = &array.dtype else {
            vortex_bail!("Expected struct dtype, found {:?}", array.dtype)
        };

        // First child is validity (if present), followed by fields
        let (validity, non_data_children) = if children.len() == struct_dtype.nfields() {
            (array.validity.clone(), 0_usize)
        } else if children.len() == struct_dtype.nfields() + 1 {
            (Validity::Array(children[0].clone()), 1_usize)
        } else {
            vortex_bail!(
                "Expected {} or {} children, found {}",
                struct_dtype.nfields(),
                struct_dtype.nfields() + 1,
                children.len()
            );
        };

        let fields: Arc<[ArrayRef]> = children.into_iter().skip(non_data_children).collect();
        vortex_ensure!(
            fields.len() == struct_dtype.nfields(),
            "Expected {} field children, found {}",
            struct_dtype.nfields(),
            fields.len()
        );

        array.fields = fields;
        array.validity = validity;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Array<Self>,
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
