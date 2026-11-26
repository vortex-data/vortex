// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixups applied to Arrow data on its way out to Java.
//!
//! arrow-java (19.0.0, pinned in `java/gradle/libs.versions.toml`) implements only part of the
//! C Data Interface, so a few Arrow constructs that Vortex produces cannot cross the boundary
//! as-is. Each fixup here compensates for one such gap and can be deleted when the Java
//! dependency catches up.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::array::make_array;
use arrow_data::ArrayData;
use arrow_data::transform::MutableArrayData;
use vortex::error::VortexResult;

/// Copy any column carrying a non-zero array offset into an offset-0 equivalent because
/// arrow-java's C Data importer ignores `offset`.
pub(crate) fn rebase_offsets(batch: RecordBatch) -> VortexResult<RecordBatch> {
    let mut rebased = false;
    let columns = batch
        .columns()
        .iter()
        .map(|column| {
            let data = column.to_data();
            if carries_offset(&data) {
                rebased = true;
                // `concat` of a single array will not do this: it short-circuits to
                // `slice(0, len)`, which keeps the offset.
                let len = data.len();
                let mut copy = MutableArrayData::new(vec![&data], false, len);
                copy.try_extend(0, 0, len)?;
                Ok(make_array(copy.freeze()))
            } else {
                Ok(Arc::clone(column))
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;

    if !rebased {
        return Ok(batch);
    }
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

/// Whether `data` or any of its descendants carries a non-zero offset.
fn carries_offset(data: &ArrayData) -> bool {
    data.offset() != 0 || data.child_data().iter().any(carries_offset)
}
