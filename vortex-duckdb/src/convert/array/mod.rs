mod data_chunk_adaptor;
mod varbinview;

use arrow_array::ArrayRef as ArrowArrayRef;
pub use data_chunk_adaptor::NamedDataChunk;
use duckdb::core::{DataChunkHandle, FlatVector, SelectionVector};
use duckdb::vtab::arrow::{
    WritableVector, flat_vector_to_arrow_array, write_arrow_array_to_vector,
};
use num_traits::AsPrimitive;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::arrays::{
    ChunkedArray, ChunkedEncoding, PrimitiveArray, StructArray, VarBinViewArray, VarBinViewEncoding,
};
use vortex_array::arrow::FromArrowArray;
use vortex_array::compute::{take, to_arrow_preferred};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayRef, ArrayStatistics, IntoArray, ToCanonical};
use vortex_dict::{DictArray, DictEncoding};
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_fsst::{FSSTArray, FSSTEncoding};
use vortex_mask::Mask;

use crate::convert::array::data_chunk_adaptor::{DataChunkHandleSlice, SizedFlatVector};
use crate::convert::scalar::ToDuckDBScalar;
use crate::{DUCKDB_STANDARD_VECTOR_SIZE, ToDuckDBType};

#[derive(Default)]
pub struct ConversionCache {
    pub values_cache: HashMap<usize, FlatVector>,
    // A value which must be unique for a given duckdb pipeline.
    pub instance_id: u64,
}

impl ConversionCache {
    pub fn new(id: u64) -> Self {
        Self {
            instance_id: id,
            ..Self::default()
        }
    }
}

pub trait ToDuckDB {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()>;
}

pub fn to_duckdb(
    array: &ArrayRef,
    chunk: &mut dyn WritableVector,
    cache: &mut ConversionCache,
) -> VortexResult<()> {
    if try_to_duckdb(array, chunk, cache)?.is_some() {
        return Ok(());
    };
    let canonical_array = array.to_canonical()?.into_array();
    if try_to_duckdb(&canonical_array, chunk, cache)?.is_some() {
        return Ok(());
    };
    to_arrow_preferred(&canonical_array)?.to_duckdb(chunk, cache)
}

fn try_to_duckdb(
    array: &ArrayRef,
    chunk: &mut dyn WritableVector,
    cache: &mut ConversionCache,
) -> VortexResult<Option<()>> {
    if let Some(constant) = array.as_constant() {
        let value = constant.try_to_duckdb_scalar()?;
        chunk.flat_vector().assign_to_constant(&value);
        Ok(Some(()))
    } else if array.is_encoding(ChunkedEncoding.id()) {
        array
            .as_any()
            .downcast_ref::<ChunkedArray>()
            .vortex_expect("chunk checked")
            .to_duckdb(chunk, cache)
            .map(Some)
    } else if array.is_encoding(VarBinViewEncoding.id()) {
        array
            .as_any()
            .downcast_ref::<VarBinViewArray>()
            .vortex_expect("varbinview id checked")
            .to_duckdb(chunk, cache)
            .map(Some)
    } else if array.is_encoding(FSSTEncoding.id()) {
        let arr = array
            .as_any()
            .downcast_ref::<FSSTArray>()
            .vortex_expect("FSSTArray id checked");
        arr.to_varbinview()?.to_duckdb(chunk, cache).map(Some)
    } else if array.is_encoding(DictEncoding.id()) {
        array
            .as_any()
            .downcast_ref::<DictArray>()
            .vortex_expect("dict id checked")
            .to_duckdb(chunk, cache)
            .map(Some)
    } else {
        Ok(None)
    }
}

impl ToDuckDB for ChunkedArray {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        // TODO(joe): support multi-chunk arrays without canonical.
        if self.chunks().len() > 1 {
            to_duckdb(&self.to_canonical()?.into_array(), chunk, cache)
        } else {
            to_duckdb(&self.chunks()[0], chunk, cache)
        }
    }
}

impl ToDuckDB for DictArray {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        // Note you can only have nullable values (not codes/selection vectors),
        // so we cannot assign a selection vector.
        if !self.codes().all_valid()? {
            let values = take(self.values(), self.codes())?;
            return to_duckdb(&values, chunk, cache);
        };

        let value_ptr = (self.values().as_ref() as *const dyn Array as *const ()) as usize;

        let mut vector: FlatVector = if self.values().len() <= DUCKDB_STANDARD_VECTOR_SIZE {
            // If the values fit into a single vector, put the values in the pre-allocated vector.
            to_duckdb(self.values(), chunk, cache)?;
            chunk.flat_vector()
        } else {
            // If the values don't fit allocated a larger vector and that the data chunk vector
            // reference this new one.
            let entry = cache.values_cache.get(&value_ptr);
            let value_vector = match entry {
                None => {
                    let mut value_vector = FlatVector::allocate_new_vector_with_capacity(
                        self.values().dtype().to_duckdb_type()?,
                        self.values().len(),
                    );
                    to_duckdb(self.values(), &mut value_vector, cache)?;
                    cache.values_cache.insert(value_ptr, value_vector);
                    cache
                        .values_cache
                        .get(&value_ptr)
                        .vortex_expect("just added")
                }
                Some(entry) => entry,
            };

            let mut vector = chunk.flat_vector();
            vector.reference(value_vector);
            vector
        };
        let sel = selection_vector_from_array(self.codes().to_primitive()?);
        vector.slice(self.values().len() as u64, sel);
        vector.set_dictionary_id(format!("{}-{}", cache.instance_id, value_ptr));
        Ok(())
    }
}

pub fn selection_vector_from_array(prim: PrimitiveArray) -> SelectionVector {
    match_each_integer_ptype!(prim.ptype(), |$P| {
        selection_vector_from_slice(prim.as_slice::<$P>())
    })
}

pub fn selection_vector_from_slice<P: NativePType + AsPrimitive<u32>>(
    slice: &[P],
) -> SelectionVector {
    slice.iter().map(|v| (*v).as_()).collect()
}

pub fn write_validity_from_mask(mask: Mask, flat_vector: &mut FlatVector) {
    // Check that both the target vector is large enough and the mask too.
    // If we later allow vectors larger than 2k (against duckdb defaults), we can revisit this.
    assert!(mask.len() <= flat_vector.capacity());
    match mask {
        Mask::AllTrue(len) => {
            if let Some(slice) = flat_vector.validity_slice() {
                // This is only needed if the vector as previously allocated.
                slice[0..len].fill(u64::MAX)
            }
        }
        Mask::AllFalse(len) => {
            let slice = flat_vector.init_get_validity_slice();
            slice[0..len].fill(u64::MIN)
        }
        Mask::Values(arr) => {
            // TODO(joe): do this MUCH better, with a shifted u64 copy
            for (idx, v) in arr.boolean_buffer().iter().enumerate() {
                if !v {
                    flat_vector.set_null(idx);
                }
            }
        }
    }
}

pub fn to_duckdb_chunk(
    struct_array: &StructArray,
    chunk: &mut DataChunkHandle,
    cache: &mut ConversionCache,
) -> VortexResult<()> {
    if struct_array.fields().is_empty() {
        // This happens If the file result is a count(*), then there will be struct fields,
        // but a single chunk, column.
        // We just need to set the length and can ignore the values.
        assert!(chunk.num_columns() <= 1);
        chunk.set_len(struct_array.len());
        return Ok(());
    }

    assert_eq!(struct_array.fields().len(), chunk.num_columns());

    chunk.set_len(struct_array.len());
    for (idx, field) in struct_array.fields().iter().enumerate() {
        to_duckdb(field, &mut DataChunkHandleSlice::new(chunk, idx), cache)?;
    }
    Ok(())
}

impl ToDuckDB for ArrowArrayRef {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        _: &mut ConversionCache,
    ) -> VortexResult<()> {
        write_arrow_array_to_vector(self, chunk)
            .map_err(|e| vortex_err!("Failed to convert vortex duckdb array: {}", e.to_string()))
    }
}

pub trait FromDuckDB<V> {
    fn from_duckdb(vector: V) -> VortexResult<ArrayRef>;
}

impl<'a> FromDuckDB<&'a NamedDataChunk<'a>> for ArrayRef {
    fn from_duckdb(named_chunk: &'a NamedDataChunk<'a>) -> VortexResult<ArrayRef> {
        let chunk = &named_chunk.chunk;
        let names = &named_chunk.names;
        let len = chunk.len();

        let columns = (0..chunk.num_columns())
            .map(|i| {
                let vector = chunk.flat_vector(i);
                let array = ArrayRef::from_duckdb(SizedFlatVector {
                    vector,
                    nullable: named_chunk.nullable.map(|null| null[i]).unwrap_or(true),
                    len,
                })?;

                // Figure out the column names
                Ok((
                    names
                        .as_ref()
                        .map(|names| names[i].clone())
                        .unwrap_or_else(|| i.to_string().into()),
                    array,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let (names, arrays): (Vec<_>, Vec<_>) = columns.into_iter().unzip();

        // All top level struct are non-nullable in duckdb, only inner columns can be nullable.
        StructArray::try_new(names.into(), arrays, len, Validity::NonNullable)
            .map(StructArray::into_array)
    }
}

impl FromDuckDB<SizedFlatVector> for ArrayRef {
    // TODO(joe): going via is slow, make it faster.
    fn from_duckdb(mut sized_vector: SizedFlatVector) -> VortexResult<ArrayRef> {
        let len = sized_vector.len;
        let arrow_arr = flat_vector_to_arrow_array(&mut sized_vector.vector, len)
            .map_err(|e| vortex_err!("Failed to convert duckdb array to vortex: {}", e))?;
        Ok(ArrayRef::from_arrow(arrow_arr, sized_vector.nullable))
    }
}

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex_array::arrays::{
        BoolArray, ConstantArray, PrimitiveArray, StructArray, VarBinArray, VarBinViewArray,
    };
    use vortex_array::validity::Validity;
    use vortex_array::variants::StructArrayTrait;
    use vortex_array::{Array, ArrayRef, ToCanonical};
    use vortex_dict::DictArray;
    use vortex_dtype::{DType, FieldNames, Nullability};
    use vortex_scalar::Scalar;

    use crate::convert::array::data_chunk_adaptor::NamedDataChunk;
    use crate::convert::array::{ConversionCache, to_duckdb_chunk};
    use crate::{FromDuckDB, ToDuckDBType};

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
        to_duckdb_chunk(
            &struct_arr,
            &mut output_chunk,
            &mut ConversionCache::default(),
        )
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
    fn test_const_vortex_to_duckdb() {
        let arr = ConstantArray::new::<i64>(23444233, 100).to_array();
        let arr2 = ConstantArray::new::<i32>(234, 100).to_array();
        let st = StructArray::from_fields(&[("1", arr.clone()), ("2", arr2.clone())]).unwrap();
        let mut output_chunk = DataChunkHandle::new(&[
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
            LogicalTypeHandle::from(LogicalTypeId::Integer),
        ]);
        to_duckdb_chunk(&st, &mut output_chunk, &mut ConversionCache::default()).unwrap();

        assert_eq!(
            format!("{:?}", output_chunk),
            "Chunk - [2 Columns]\n- CONSTANT BIGINT: 100 = [ 23444233]\n- CONSTANT INTEGER: 100 = [ 234]\n"
        )
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
        let const2 = ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), len);
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
        to_duckdb_chunk(&str, &mut chunk, &mut ConversionCache::default()).unwrap();

        chunk.verify();
        assert_eq!(
            format!("{:?}", chunk),
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
        to_duckdb_chunk(&str, &mut chunk, &mut ConversionCache::default()).unwrap();

        chunk.verify();
        assert_eq!(
            format!("{:?}", chunk),
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 5 = [ 0, 3, 4, 2, 1]
"#
        );
    }
}
