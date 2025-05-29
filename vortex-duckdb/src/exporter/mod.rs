mod constant;
mod decimal;
mod dict;
mod primitive;
mod run_end;
mod varbinview;

use duckdb::arrow::array::ArrayRef as ArrowArrayRef;
use duckdb::core::{DataChunkHandle, FlatVector};
use duckdb::ffi::duckdb_data_chunk_get_vector;
use duckdb::vtab::arrow::{WritableVector, write_arrow_array_to_vector};
use itertools::Itertools;
use vortex::arrays::{ConstantVTable, StructArray};
use vortex::arrow::compute::to_arrow_preferred;
use vortex::encodings::dict::DictVTable;
use vortex::encodings::runend::RunEndVTable;
use vortex::error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::iter::ArrayIterator;
use vortex::mask::Mask;
use vortex::{Array, Canonical, ToCanonical};

use crate::{ConversionCache, DUCKDB_STANDARD_VECTOR_SIZE};

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
    pub fn export(&mut self, chunk: &mut DataChunkHandle) -> VortexResult<bool> {
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
    pub fn export(&mut self, chunk: &mut DataChunkHandle) -> VortexResult<bool> {
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
            let mut vector = unsafe { duckdb_data_chunk_get_vector(chunk.get_ptr(), i as u64) };
            field.export(position, chunk_len, &mut vector)?;
        }
        Ok(true)
    }
}

/// Exporter for a single column of a DuckDB data chunk.
///
/// NOTE(ngates): we could actually convert this into a Vortex compute function that takes
///  the offset, len and [`WritableVector`] as options. Not sure what it should return though?
///  This would allow Vortex extension authors to plug into the DuckDB exporter system.
pub trait ColumnExporter {
    /// Export the given range of data from the Vortex array to the DuckDB vector.
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()>;
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
    let array = to_arrow_preferred(array.as_ref())?;
    Ok(Box::new(ArrowArrayExporter { array }))
}

struct ArrowArrayExporter {
    array: ArrowArrayRef,
}

impl ColumnExporter for ArrowArrayExporter {
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
    /// Returns true if *all* values within the offset -> len slice are null.
    /// Since we're iterating these values anyway, then it's cheaper for us to check it inline.
    fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool;
}

impl FlatVectorExt for FlatVector {
    fn set_validity(&mut self, mask: &Mask, offset: usize, len: usize) -> bool {
        match mask {
            Mask::AllTrue(_) => {
                // We only need to blank out validity if there is already a slice allocated.
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

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex::arrays::{
        BoolArray, ConstantArray, PrimitiveArray, StructArray, VarBinArray, VarBinViewArray,
    };
    use vortex::dtype::{DType, FieldNames, Nullability};
    use vortex::encodings::dict::DictArray;
    use vortex::scalar::Scalar;
    use vortex::validity::Validity;
    use vortex::{Array, ArrayRef, IntoArray, ToCanonical};

    use super::*;
    use crate::{FromDuckDB, NamedDataChunk, ToDuckDBType};

    fn data() -> ArrayRef {
        let xs = PrimitiveArray::from_iter(0..5);
        let ys = VarBinArray::from_vec(
            vec!["a", "b", "c", "d", "e"],
            DType::Utf8(Nullability::NonNullable),
        );
        let zs = BoolArray::from_iter([true, true, true, false, false]);

        let struct_a = StructArray::try_new(
            FieldNames::from(["xs".into(), "ys".into(), "zs".into()]),
            vec![xs.into_array(), ys.into_array(), zs.into_array()],
            5,
            Validity::NonNullable,
        )
        .unwrap();
        struct_a.to_array()
    }

    #[test]
    fn test_vortex_to_duckdb() {
        let arr = data();
        let (nullable, ddb_type): (Vec<_>, Vec<_>) = arr
            .dtype()
            .as_struct()
            .unwrap()
            .fields()
            .map(|f| (f.is_nullable(), f.to_duckdb_type().unwrap()))
            .unzip();
        let struct_arr = arr.to_struct().unwrap();

        let mut output_chunk = DataChunkHandle::new(ddb_type.as_slice());

        ArrayExporter::try_new(&struct_arr, &mut ConversionCache::default())
            .unwrap()
            .export(&mut output_chunk)
            .unwrap();

        let vx_arr = ArrayRef::from_duckdb(&NamedDataChunk::new(
            &output_chunk,
            &nullable,
            FieldNames::from(["xs".into(), "ys".into(), "zs".into()]),
        ))
        .unwrap();
        assert_eq!(
            struct_arr.names(),
            vx_arr.clone().to_struct().unwrap().names()
        );
        for field in vx_arr.to_struct().unwrap().fields() {
            assert_eq!(field.len(), arr.len());
        }
        assert_eq!(vx_arr.len(), arr.len());
        assert_eq!(vx_arr.dtype(), arr.dtype());
    }

    #[test]
    fn test_large_struct_to_duckdb() {
        let len = 5;
        let mut chunk = DataChunkHandle::new(&[
            LogicalTypeHandle::from(LogicalTypeId::Integer),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Boolean),
            LogicalTypeHandle::from(LogicalTypeId::SQLNull),
        ]);
        let pr: PrimitiveArray = (0i32..i32::try_from(len).unwrap()).collect();
        let varbin: VarBinViewArray =
            VarBinViewArray::from_iter_str(["a", "ab", "abc", "abcd", "abcde"]);
        let dict_varbin = DictArray::try_new(
            [0u32, 3, 4, 2, 1]
                .into_iter()
                .collect::<PrimitiveArray>()
                .to_array(),
            varbin.to_array(),
        )
        .unwrap();
        let const1 = ConstantArray::new(
            Scalar::new(DType::Bool(Nullability::Nullable), true.into()),
            len,
        );
        let const2 = ConstantArray::new(Scalar::null(DType::Null), len);
        let str = StructArray::from_fields(&[
            ("pr", pr.to_array()),
            ("varbin", varbin.to_array()),
            ("dict_varbin", dict_varbin.to_array()),
            ("const1", const1.to_array()),
            ("const2", const2.to_array()),
        ])
        .unwrap()
        .to_struct()
        .unwrap();

        ArrayExporter::try_new(&str, &mut ConversionCache::default())
            .unwrap()
            .export(&mut chunk)
            .unwrap();

        chunk.verify();
        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [5 Columns]
- FLAT INTEGER: 5 = [ 0, 1, 2, 3, 4]
- FLAT VARCHAR: 5 = [ a, ab, abc, abcd, abcde]
- DICTIONARY VARCHAR: 5 = [ a, abcd, abcde, abc, ab]
- CONSTANT BOOLEAN: 5 = [ true]
- CONSTANT "NULL": 5 = [ NULL]
"#
        );
    }

    // The values of the dict don't fit in a vectors, this can cause problems.
    #[test]
    fn test_large_dict_to_duckdb() {
        let mut chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);
        let dict_varbin = DictArray::try_new(
            [0u32, 3, 4, 2, 1]
                .into_iter()
                .collect::<PrimitiveArray>()
                .to_array(),
            (0i32..100000).collect::<PrimitiveArray>().to_array(),
        )
        .unwrap();
        let str = StructArray::from_fields(&[("dict", dict_varbin.to_array())])
            .unwrap()
            .to_struct()
            .unwrap();

        ArrayExporter::try_new(&str, &mut ConversionCache::default())
            .unwrap()
            .export(&mut chunk)
            .unwrap();

        chunk.verify();
        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 5 = [ 0, 3, 4, 2, 1]
"#
        );
    }
}
