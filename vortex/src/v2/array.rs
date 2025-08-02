// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::ops::{BindContext, BufferId, BufferSource, NodeId, Operator};
use vortex_array::Canonical;
use vortex_array::stats::StatsSet;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

/// Represents a logical in-memory array of data.
pub struct Array {
    /// The length of the array.
    len: usize,
    /// The logical data type of the array.
    dtype: DType,
    /// Logical statistics of the array.
    stats: StatsSet,

    /// The operator that produces the array's data.
    operator: Box<dyn Operator>,

    /// The array's buffers.
    buffers: Vec<ByteBuffer>,
}

impl Array {
    /// Converts this array to a canonical representation.
    pub fn to_canonical(&self) -> VortexResult<Canonical> {
        // Allocate the output array with the same length and data type.

        // Create an evaluation for the operator.
        let eval = self.operator.bind(&BindContext {
            len: self.len,
            dtype: &self.dtype,
            metadata: &[],
            stats: Some(&self.stats),
        })?;

        todo!("Drive the evaluation to populate the canonical array");
    }
}

impl BufferSource for Vec<ByteBuffer> {
    fn get(&self, buffer_id: BufferId) -> VortexResult<Option<ByteBuffer>> {
        if *buffer_id >= self.len() {
            vortex_bail!(
                "Buffer ID {} out of range for buffers of length {}",
                *buffer_id,
                self.len()
            );
        }
        Ok(Some(self[*buffer_id].clone()))
    }
}
