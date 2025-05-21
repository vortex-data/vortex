#![allow(unused_variables)]
use std::marker::PhantomData;

use duckdb::core::DataChunkHandle;
use duckdb::ffi::duckdb_data_chunk_get_vector;
use duckdb::vtab::arrow::WritableVector;
use itertools::Itertools;
use vortex_array::arrays::{PrimitiveArray, StructArray};
use vortex_array::{Array, Canonical};
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::ConversionCache;

pub struct DuckDBExporter {
    fields: Vec<Box<dyn ArrayExporter>>,
    len: usize,
    pos: usize,
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
            len: array.len(),
            pos: 0,
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
        println!("CHUNK LEN {}", chunk.len());

        for (i, field) in self.fields.iter_mut().enumerate() {
            let mut vector = unsafe { duckdb_data_chunk_get_vector(chunk.get_ptr(), i as u64) };
            field.export(self.pos, chunk.len(), &mut vector, cache)?;
        }

        self.pos += chunk.len();
        Ok(self.pos < self.len)
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
                Box::new(
                    PrimitiveExporter::<$P> {
                        array: array.clone(),
                        phantom: PhantomData::<$P>,
                    }
                )
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
    phantom: PhantomData<T>,
}

impl<T: NativePType> ArrayExporter for PrimitiveExporter<T> {
    fn export(
        &mut self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        todo!()
    }
}
