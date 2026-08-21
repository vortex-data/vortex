// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex::mask::MaskValuesRef;

use crate::duckdb::ValidityData;
use crate::duckdb::VectorBuffer;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

/// Exports validity before delegating values to another [`ColumnExporter`].
struct ValidityExporter {
    mask: Mask,

    /// Points into `mask` and keeps its bitmap alive when DuckDB can use it directly.
    zero_copy: Option<ValidityData>,

    exporter: Box<dyn ColumnExporter>,
}

/// Returns the zero-copy validity data for `values`, if DuckDB can read its bit buffer directly.
///
/// The bit buffer must satisfy these requirements:
///
/// - Its bit offset is zero.
/// - Its byte buffer is aligned for `u64`.
/// - Its byte length is a multiple of `size_of::<u64>()`.
fn zero_copy_validity(values: &MaskValuesRef) -> Option<ValidityData> {
    let bit_buffer = values.bit_buffer();
    if bit_buffer.offset() != 0 {
        return None;
    }

    let buffer = bit_buffer.inner().clone();
    let bytes = buffer.as_slice();
    let data_ptr = bytes.as_ptr();

    // DuckDB reads `u64` words. A misaligned pointer causes undefined behavior, and a trailing
    // partial word can cause an out-of-bounds read.
    if !(data_ptr as usize).is_multiple_of(size_of::<u64>())
        || !bytes.len().is_multiple_of(size_of::<u64>())
    {
        return None;
    }

    Some(ValidityData {
        shared_buffer: VectorBuffer::new(buffer),
        data_ptr,
    })
}

pub(crate) fn new_exporter(
    mask: Mask,
    exporter: Box<dyn ColumnExporter>,
) -> Box<dyn ColumnExporter> {
    let zero_copy = match &mask {
        Mask::AllTrue(_) | Mask::AllFalse(0) => return exporter,
        Mask::AllFalse(_) => None,
        Mask::Values(values) => zero_copy_validity(values),
    };

    Box::new(ValidityExporter {
        mask,
        zero_copy,
        exporter,
    })
}

impl ColumnExporter for ValidityExporter {
    fn preferred_batch_len(&self, offset: usize, max_len: usize) -> usize {
        self.exporter.preferred_batch_len(offset, max_len)
    }

    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        assert!(
            offset + len <= self.mask.len(),
            "cannot access outside of array"
        );

        if unsafe {
            vector.set_validity_zero_copy(&self.mask, offset, len, self.zero_copy.as_ref())
        } {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        self.exporter.export(offset, len, vector, ctx)?;

        Ok(())
    }
}
