use std::marker::PhantomData;

use duckdb::core::DataChunkHandle;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::{Array, Canonical};
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::ConversionCache;

/// A trait for exporting Vortex arrays to DuckDB vectors. Since DuckDB passes us mutable vectors
/// of 2k in size, this trait sort of acts like an iterator over the array in a way that allows
/// us to cheaply slice off 2k elements at a time.
pub trait DuckDBExporter {
    /// Export the next chunk of data from the Vortex array to the DuckDB vector.
    ///
    /// Returns `true` if there is more data to export, `false` if the array is exhausted.
    fn export(
        &mut self,
        chunk: &mut DataChunkHandle,
        cache: &mut ConversionCache,
    ) -> VortexResult<bool>;
}

/// Create a DuckDB exporter for the given Vortex array.
pub fn create_exporter(array: &dyn Array) -> VortexResult<Box<dyn DuckDBExporter>> {
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
                        index: 0,
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
    index: usize,
}

impl<T: NativePType> DuckDBExporter for PrimitiveExporter<T> {
    fn export(
        &mut self,
        _chunk: &mut DataChunkHandle,
        _cache: &mut ConversionCache,
    ) -> VortexResult<bool> {
        todo!()
    }
}
