// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::duckdb::ValidityData;
use crate::duckdb::VectorBuffer;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

struct ValidityExporter {
    mask: Mask,
    /// If the mask's bit buffer is u64-aligned with no sub-byte offset,
    /// we can zero-copy it into DuckDB. We hold the ValidityData to keep
    /// the underlying memory alive via DuckDB's ref-counting.
    zero_copy: Option<ValidityData>,
    exporter: Box<dyn ColumnExporter>,
}

/// Returns true if the bit buffer can be zero-copied as a DuckDB validity mask.
///
/// Requirements:
/// - No sub-byte bit offset (offset == 0)
/// - The underlying byte buffer is u64-aligned
/// - The underlying byte buffer length is a multiple of 8 (so u64 reads are in-bounds)
fn can_zero_copy_validity(mask: &Mask) -> bool {
    let Mask::Values(values) = mask else {
        return false;
    };
    let bit_buf = values.bit_buffer();
    if bit_buf.offset() != 0 {
        return false;
    }
    let inner = bit_buf.inner();
    let slice = inner.as_slice();
    // DuckDB reads validity as u64 words, so the buffer must be u64-aligned and
    // its length must be a multiple of 8 bytes to avoid out-of-bounds reads.
    (slice.as_ptr() as usize).is_multiple_of(size_of::<u64>())
        && slice.len().is_multiple_of(size_of::<u64>())
}

pub(crate) fn new_exporter(
    mask: Mask,
    exporter: Box<dyn ColumnExporter>,
) -> Box<dyn ColumnExporter> {
    if mask.all_true() {
        exporter
    } else {
        let zero_copy = can_zero_copy_validity(&mask).then(|| {
            let Mask::Values(values) = &mask else {
                unreachable!()
            };
            let buffer = values.bit_buffer().inner().clone();
            let data_ptr = buffer.as_slice().as_ptr();
            ValidityData {
                shared_buffer: VectorBuffer::new(buffer),
                data_ptr,
            }
        });
        Box::new(ValidityExporter {
            mask,
            zero_copy,
            exporter,
        })
    }
}

impl ColumnExporter for ValidityExporter {
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
