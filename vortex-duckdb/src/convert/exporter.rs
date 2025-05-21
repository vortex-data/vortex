#![allow(unused_variables)]

use std::marker::PhantomData;

use duckdb::core::{DataChunkHandle, FlatVector};
use duckdb::ffi::duckdb_data_chunk_get_vector;
use duckdb::vtab::arrow::WritableVector;
use itertools::Itertools;
use vortex_array::arrays::{PrimitiveArray, StructArray};
use vortex_array::{Array, Canonical};
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ConversionCache, DUCKDB_STANDARD_VECTOR_SIZE};

pub struct DuckDBExporter {
    fields: Vec<Box<dyn ArrayExporter>>,
    array_len: usize,
    remaining: usize,
}

impl DuckDBExporter {
    pub fn try_new(array: &StructArray) -> VortexResult<Self> {
        let fields = array
            .fields()
            .iter()
            .map(|field| create_exporter(field.as_ref()))
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

        if self.fields.is_empty() {
            // No fields can occur in e.g. select(*) queries. In these cases, we just need to
            // set the length of the chunk.
            chunk.set_len(chunk_len);
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
        &mut self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()>;
}

/// Create a DuckDB exporter for the given Vortex array.
fn create_exporter(array: &dyn Array) -> VortexResult<Box<dyn ArrayExporter>> {
    // Constant
    // Chunked
    // VarBinView
    // FSST
    // Dict
    // RunEnd

    // Otherwise, we fall back to canonical
    let array = array.to_canonical()?;
    Ok(match array {
        Canonical::Null(_) => {
            todo!()
        }
        Canonical::Bool(_) => {
            todo!()
        }
        Canonical::Primitive(array) => {
            match_each_native_ptype!(array.ptype(), |$P| {
                Box::new(PrimitiveExporter::<$P>::try_new(array)?)
            })
        }
        Canonical::Decimal(_) => {
            todo!()
        }
        Canonical::Struct(_) => {
            todo!()
        }
        Canonical::List(_) => {
            todo!()
        }
        Canonical::VarBinView(_) => {
            todo!()
        }
        Canonical::Extension(_) => {
            todo!()
        }
    })
}

#[allow(dead_code)]
struct PrimitiveExporter<T: NativePType> {
    array: PrimitiveArray,
    validity: Mask,
    phantom: PhantomData<T>,
}

impl<T: NativePType> PrimitiveExporter<T> {
    fn try_new(array: PrimitiveArray) -> VortexResult<Self> {
        let validity = array.validity_mask()?;
        Ok(Self {
            array,
            validity,
            phantom: PhantomData,
        })
    }
}

impl<T: NativePType> ArrayExporter for PrimitiveExporter<T> {
    fn export(
        &mut self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        let mut vector = vector.flat_vector();

        // Set validity if necessary.
        if vector.set_validity(&self.validity, offset, len) {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        // Copy the values from the Vortex array to the DuckDB vector.
        vector
            .as_mut_slice_with_len(len)
            .copy_from_slice(&self.array.as_slice::<T>()[offset..offset + len]);

        Ok(())
    }
}

trait FlatVectorExt {
    /// Returns true if *all* values are null.
    fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool;
}

impl FlatVectorExt for FlatVector {
    fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        match mask {
            Mask::AllTrue(len) => {
                if let Some(validity) = self.validity_slice() {
                    validity[..].fill(u64::MAX)
                }
                false
            }
            Mask::AllFalse(len) => {
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
