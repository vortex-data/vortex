mod cache;
mod constant;
mod decimal;
mod dict;
mod primitive;
mod run_end;
mod sequence;
mod varbinview;

use cache::*;
use itertools::Itertools;
use vortex::arrays::{ConstantVTable, StructArray};
use vortex::encodings::dict::DictVTable;
use vortex::encodings::runend::RunEndVTable;
use vortex::encodings::sequence::SequenceVTable;
use vortex::error::{VortexExpect, VortexResult, vortex_bail};
use vortex::iter::ArrayIterator;
use vortex::mask::Mask;
use vortex::{Array, Canonical, ToCanonical};

use crate::duckdb::{DUCKDB_STANDARD_VECTOR_SIZE, DataChunk, Vector};

/// DuckDB exporter for an [`ArrayIterator`], sharing state and caches.
pub struct ArrayIteratorExporter {
    iter: Box<dyn ArrayIterator>,
    cache: ConversionCache,

    array_exporter: Option<ArrayExporter>,
}

impl ArrayIteratorExporter {
    pub fn new(iter: Box<dyn ArrayIterator>) -> Self {
        Self {
            iter,
            cache: ConversionCache::default(),
            array_exporter: None,
        }
    }

    /// Returns `true` if a chunk was exported, `false` if all data has been exported.
    pub fn export(&mut self, chunk: &mut DataChunk) -> VortexResult<bool> {
        loop {
            if self.array_exporter.is_none() {
                if let Some(array) = self.iter.next() {
                    // Create a new array exporter for the current array.
                    let array = array?.to_struct()?;
                    self.array_exporter = Some(ArrayExporter::try_new(&array, &mut self.cache)?);
                } else {
                    // No more arrays to export.
                    return Ok(false);
                }
            }

            if self
                .array_exporter
                .as_mut()
                .vortex_expect("must be present")
                .export(chunk)?
            {
                return Ok(true);
            } else {
                // This exporter is done, so we throw it away and loop.
                self.array_exporter = None;
            }
        }
    }
}

pub struct ArrayExporter {
    fields: Vec<Box<dyn ColumnExporter>>,
    array_len: usize,
    remaining: usize,
}

impl ArrayExporter {
    pub fn try_new(array: &StructArray, cache: &mut ConversionCache) -> VortexResult<Self> {
        let fields = array
            .fields()
            .iter()
            .map(|field| new_array_exporter(field.as_ref(), cache))
            .try_collect()?;
        Ok(Self {
            fields,
            array_len: array.len(),
            remaining: array.len(),
        })
    }

    /// Export the data into the next chunk.
    ///
    /// Returns `true` if a chunk was exported, `false` if all rows have been exported.
    pub fn export(&mut self, chunk: &mut DataChunk) -> VortexResult<bool> {
        if self.remaining == 0 {
            return Ok(false);
        }

        if self.fields.is_empty() {
            // No fields can occur in e.g. count(*) queries. In these cases, we just need to
            // set the length of the chunk and return.
            chunk.set_len(self.remaining);
            self.remaining = 0;

            return Ok(true);
        }

        let chunk_len = DUCKDB_STANDARD_VECTOR_SIZE.min(self.remaining);
        let position = self.array_len - self.remaining;
        self.remaining -= chunk_len;
        chunk.set_len(chunk_len);

        for (i, field) in self.fields.iter_mut().enumerate() {
            let mut vector = chunk.get_vector(i);
            field.export(position, chunk_len, &mut vector)?;
        }

        Ok(true)
    }
}

/// Exporter for a single column of a DuckDB data chunk.
///
/// NOTE(ngates): we could actually convert this into a Vortex compute function that takes
///  the offset, len and `WritableVector` as options. Not sure what it should return though?
///  This would allow Vortex extension authors to plug into the DuckDB exporter system.
pub trait ColumnExporter {
    /// Export the given range of data from the Vortex array to the DuckDB vector.
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()>;
}

/// Create a DuckDB exporter for the given Vortex array.
fn new_array_exporter(
    array: &dyn Array,
    cache: &mut ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if let Some(array) = array.as_opt::<ConstantVTable>() {
        return constant::new_exporter(array);
    }

    if let Some(array) = array.as_opt::<RunEndVTable>() {
        return run_end::new_exporter(array, cache);
    }

    if let Some(array) = array.as_opt::<DictVTable>() {
        return dict::new_exporter(array, cache);
    }

    if let Some(array) = array.as_opt::<SequenceVTable>() {
        return sequence::new_exporter(array);
    }

    // Otherwise, we fall back to canonical
    let array = array.to_canonical()?;
    match array {
        Canonical::Null(_) => {}
        Canonical::Bool(_) => {}
        Canonical::Primitive(array) => return primitive::new_exporter(&array),
        Canonical::Decimal(array) => return decimal::new_exporter(&array),
        Canonical::Struct(_) => {
            // The Arrow exporter does not support struct arrays yet, so we bail out.
            vortex_bail!("Struct arrays are not supported in DuckDB export yet");
        }
        Canonical::List(_) => {
            // The Arrow exporter does not support list arrays yet, so we bail out.
            vortex_bail!("List arrays are not supported in DuckDB export yet");
        }
        Canonical::VarBinView(array) => return varbinview::new_exporter(&array),
        Canonical::Extension(_) => {}
    }

    // Otherwise use Arrow.
    // let array = to_arrow_preferred(array.as_ref())?;
    // Ok(Box::new(ArrowArrayExporter { array }))
    vortex_bail!(
        "DuckDB export for array type {} is not implemented yet",
        array.as_ref().encoding_id()
    );
}
//
// struct ArrowArrayExporter {
//     array: ArrowArrayRef,
// }
//
// impl ColumnExporter for ArrowArrayExporter {
//     fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
//         write_arrow_array_to_vector(&self.array.slice(offset, len), vector)
//             .map_err(|e| vortex_err!("Failed to convert Arrow array to DuckDB vector {e}"))
//     }
// }

pub(crate) trait VectorExt {
    /// Returns true if *all* values within the offset -> len slice are null.
    /// Since we're iterating these values anyway, then it's cheaper for us to check it inline.
    fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool;
}

impl VectorExt for Vector {
    fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        match mask {
            Mask::AllTrue(_) => {
                // We only need to blank out validity if there is already a slice allocated.
                if let Some(validity) = self.validity_slice_mut() {
                    validity.fill(true);
                }
                false
            }
            Mask::AllFalse(_) => {
                self.ensure_validity_slice().fill(false);
                true
            }
            Mask::Values(arr) => {
                // TODO(joe): do this MUCH better, with a shifted u64 copy
                let mut null_count = 0;
                let validity = self.ensure_validity_slice();
                for (idx, v) in arr.boolean_buffer().slice(offset, len).iter().enumerate() {
                    if !v {
                        validity.set(idx, false);
                        null_count += 1;
                    }
                }
                null_count == len
            }
        }
    }
}
