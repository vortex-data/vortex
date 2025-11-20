// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_vector::Vector;
use vortex_vector::struct_::StructVector;

use crate::arrays::struct_::StructArray;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{ArrayOperator, EmptyMetadata, EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod operations;
pub mod operator;
pub mod reduce;
mod validity;
mod visitor;

pub use operator::StructExprPartitionRule;

vtable!(Struct);

impl VTable for StructVTable {
    type Array = StructArray;
    type Encoding = StructEncoding;
    type Metadata = EmptyMetadata;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.struct")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(StructEncoding.as_ref())
    }

    fn metadata(_array: &StructArray) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn serialize(_metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(_buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn build(
        _encoding: &StructEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
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

    fn execute(array: &Self::Array, ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        let fields: Box<[_]> = array
            .fields()
            .iter()
            .map(|field| field.execute_batch(ctx))
            .try_collect()?;
        // SAFETY: we know that all field lengths match the struct array length, and the validity
        Ok(unsafe { StructVector::new_unchecked(Arc::new(fields), array.validity_mask()) }.into())
    }
}

#[derive(Clone, Debug)]
pub struct StructEncoding;
