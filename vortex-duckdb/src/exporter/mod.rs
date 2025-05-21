mod dict;
mod frame_of_reference;
mod primitive;
mod run_end;

use duckdb::arrow::array::ArrayRef as ArrowArrayRef;
use duckdb::core::{DataChunkHandle, FlatVector};
use duckdb::ffi::duckdb_data_chunk_get_vector;
use duckdb::vtab::arrow::{WritableVector, write_arrow_array_to_vector};
use itertools::Itertools;
use vortex_array::arrays::StructArray;
use vortex_array::arrow::compute::to_arrow_preferred;
use vortex_array::{Array, Canonical};
use vortex_dict::DictVTable;
use vortex_error::{VortexResult, vortex_err};
use vortex_mask::Mask;
use vortex_runend::RunEndVTable;

use crate::{ConversionCache, DUCKDB_STANDARD_VECTOR_SIZE};

pub struct DuckDBExporter {
    fields: Vec<Box<dyn ArrayExporter>>,
    array_len: usize,
    remaining: usize,
}

impl DuckDBExporter {
    pub fn try_new(array: &StructArray) -> VortexResult<Self> {
        let cache = &mut ConversionCache::default();
        let fields = array
            .fields()
            .iter()
            .map(|field| create_exporter(field.as_ref(), cache))
            .try_collect()?;
        Ok(Self {
            fields,
            array_len: array.len(),
            remaining: array.len(),
        })
    }

    /// Export the data into the next chunk.
    ///
    /// Returns `true` if there are more rows to export, `false` if all rows have been exported.
    pub fn export(
        &mut self,
        chunk: &mut DataChunkHandle,
        cache: &mut ConversionCache,
    ) -> VortexResult<bool> {
        let chunk_len = DUCKDB_STANDARD_VECTOR_SIZE.min(self.remaining);
        let position = self.array_len - self.remaining;
        self.remaining = self.remaining - chunk_len;

        chunk.set_len(chunk_len);

        if self.fields.is_empty() {
            // No fields can occur in e.g. select(*) queries. In these cases, we just need to
            // set the length of the chunk.
            return Ok(self.remaining > 0);
        }

        for (i, field) in self.fields.iter_mut().enumerate() {
            let mut vector = unsafe { duckdb_data_chunk_get_vector(chunk.get_ptr(), i as u64) };
            field.export(position, chunk_len, &mut vector, cache)?;
        }

        Ok(self.remaining > 0)
    }
}

/// Exporter for a single column of a DuckDB data chunk.
pub trait ArrayExporter {
    /// Export the given range of data from the Vortex array to the DuckDB vector.
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()>;
}

/// Create a DuckDB exporter for the given Vortex array.
fn create_exporter(
    array: &dyn Array,
    cache: &mut ConversionCache,
) -> VortexResult<Box<dyn ArrayExporter>> {
    // Constant
    // Chunked
    // VarBinView
    // FSST
    // Dict
    // RunEnd

    if let Some(array) = array.as_opt::<RunEndVTable>() {
        return run_end::new_exporter(array, cache);
    }

    if let Some(array) = array.as_opt::<DictVTable>() {
        return dict::new_exporter(array, cache);
    }

    //println!("ENCODING: {}", array.encoding());

    // Otherwise, we fall back to canonical
    let array = array.to_canonical()?;
    match array {
        Canonical::Null(_) => {}
        Canonical::Bool(_) => {}
        Canonical::Primitive(array) => return primitive::new_exporter(array),
        Canonical::Decimal(_) => {}
        Canonical::Struct(_) => {}
        Canonical::List(_) => {}
        Canonical::VarBinView(_) => {}
        Canonical::Extension(_) => {}
    }

    // Otherwise use Arrow.
    let array = to_arrow_preferred(array.as_ref())?;
    Ok(Box::new(ArrowArrayExporter { array }))
}

struct ArrowArrayExporter {
    array: ArrowArrayRef,
}

impl ArrayExporter for ArrowArrayExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()> {
        write_arrow_array_to_vector(&self.array.slice(offset, len), vector)
            .map_err(|e| vortex_err!("Failed to convert Arrow array to DuckDB vector {e}"))
    }
}

pub(crate) trait FlatVectorExt {
    /// Returns true if *all* values are null.
    fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool;
}

impl FlatVectorExt for FlatVector {
    fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        match mask {
            Mask::AllTrue(_) => {
                if let Some(validity) = self.validity_slice() {
                    validity[..].fill(u64::MAX)
                }
                false
            }
            Mask::AllFalse(_) => {
                self.init_get_validity_slice()[..].fill(u64::MIN);
                true
            }
            Mask::Values(arr) => {
                // TODO(joe): do this MUCH better, with a shifted u64 copy
                let mut null_count = 0;
                for (idx, v) in arr.boolean_buffer().slice(offset, len).iter().enumerate() {
                    if !v {
                        self.set_null(idx);
                        null_count += 1;
                    }
                }
                null_count == len
            }
        }
    }
}
