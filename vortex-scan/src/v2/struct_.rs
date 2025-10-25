// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::{StreamExec, StreamExecRef, StreamNodeRef};
use async_trait::async_trait;
use futures::future::try_join_all;
use vortex_array::arrays::StructArray;
use vortex_array::{ArrayRef, IntoArray};
use vortex_dtype::DType;
use vortex_error::{vortex_panic, VortexExpect, VortexResult};
use vortex_mask::Mask;

pub struct StructStreamNode {
    dtype: DType,
    fields: Vec<StreamNodeRef>,
}

pub struct StructStreamExec {
    dtype: DType,
    fields: Vec<StreamExecRef>,
}

#[async_trait]
impl StreamExec for StructStreamExec {
    fn next_batch_size(&self) -> usize {
        // Take the min of the batch sizes of our children
        self.fields
            .iter()
            .map(|f| f.next_batch_size())
            .min()
            .vortex_expect("No fields for struct stream exec")
    }

    async fn next_batch(&mut self, mask: &Mask) -> VortexResult<ArrayRef> {
        let row_count = mask.true_count();
        let arrays: Vec<ArrayRef> =
            try_join_all(self.fields.iter_mut().map(|f| f.next_batch(mask))).await?;

        let DType::Struct(fields, nullability) = &self.dtype else {
            vortex_panic!("Must be a struct dtype")
        };

        Ok(StructArray::try_new(
            fields.names().clone(),
            arrays,
            row_count,
            (*nullability).into(),
        )?
        .into_array())
    }
}

// We use a separate exec implementation for structs with zero fields
pub struct NoFieldsStructStreamExec {
    row_count: u64,
}
