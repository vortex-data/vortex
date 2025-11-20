// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::filter::Filter;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::{Vector, VectorMut, VectorMutOps, VectorOps};

use crate::pipeline::{BitView, N, Sink};

pub struct OutputSink {
    vector: VectorMut,
}

// TODO(ngates): implement type-specific sinks to avoid downcasting in the filtering / extend logic.
impl OutputSink {
    pub fn new(dtype: DType, len: usize) -> Self {
        let vector = VectorMut::with_capacity(&dtype, len);
        Self { vector }
    }

    pub fn into_vector(self) -> Vector {
        self.vector.freeze()
    }
}

impl Sink for OutputSink {
    fn consume(&mut self, selection: &BitView, vector: Vector) -> VortexResult<Vector> {
        match selection.true_count() {
            0 => {
                // No values to append
            }
            n if vector.len() == n => {
                // The vector has already been filtered if len == true_count.
                self.vector.extend_from_vector(&vector);
            }
            _ => {
                // Otherwise, we know that the vector has not yet been filtered.
                assert_eq!(vector.len(), N, "it must therefore be len = N");
                // TODO(ngates): it would be great to have a `filter_into` that avoids the extra
                //  copy here.
                let vector = vector.filter(selection);
                self.vector.extend_from_vector(&vector);
            }
        }
        Ok(vector)
    }
}
