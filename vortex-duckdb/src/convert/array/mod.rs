mod data_chunk_adaptor;

use arrow_array::ArrayRef as ArrowArrayRef;
use duckdb::core::{DataChunkHandle, SelectionVector};
use duckdb::vtab::arrow::{
    WritableVector, flat_vector_to_arrow_array, write_arrow_array_to_vector,
};
use vortex_array::arrays::StructArray;
use vortex_array::arrow::{FromArrowArray, IntoArrowArray};
use vortex_array::compute::try_cast;
use vortex_array::validity::Validity;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayRef, ArrayStatistics, ToCanonical};
use vortex_dict::{DictArray, DictEncoding};
use vortex_dtype::DType;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::PType::U32;
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::convert::array::data_chunk_adaptor::{
    DataChunkHandleSlice, NamedDataChunk, SizedFlatVector,
};
use crate::convert::scalar::ToDuckDBScalar;

pub trait ToDuckDB {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()>;
}

pub fn to_duckdb(array: ArrayRef, chunk: &mut dyn WritableVector) -> VortexResult<()> {
    if let Some(constant) = array.as_constant() {
        let value = constant.to_duckdb_scalar();
        chunk.flat_vector().assign_to_constant(&value);
        Ok(())
    } else if array.is_encoding(DictEncoding.id()) {
        array
            .as_any()
            .downcast_ref::<DictArray>()
            .vortex_expect("dict id checked")
            .to_duckdb(chunk)
    } else {
        array.into_arrow_preferred()?.to_duckdb(chunk)
    }
}

impl ToDuckDB for DictArray {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()> {
        to_duckdb(self.values().clone(), chunk)?;
        let indices =
            try_cast(self.codes(), &DType::Primitive(U32, NonNullable))?.to_primitive()?;
        let indices = indices.as_slice::<u32>();
        let sel = SelectionVector::new_copy(indices);
        chunk.flat_vector().slice(sel);
        Ok(())
    }
}

pub fn to_duckdb_chunk(
    struct_array: &StructArray,
    chunk: &mut DataChunkHandle,
) -> VortexResult<()> {
    assert_eq!(struct_array.fields().len(), chunk.num_columns());

    chunk.set_len(struct_array.len());
    for (idx, field) in struct_array.fields().iter().enumerate() {
        to_duckdb(field.clone(), &mut DataChunkHandleSlice::new(chunk, idx))?;
    }
    Ok(())
}

impl ToDuckDB for ArrowArrayRef {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()> {
        write_arrow_array_to_vector(self, chunk)
            .map_err(|e| vortex_err!("Failed to convert vrotex duckdb array: {}", e.to_string()))
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

        // all top level struct are non nullable is duckdb, only inner columns can be.
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
        BoolArray, ConstantArray, PrimitiveArray, StructArray, VarBinArray,
    };
    use vortex_array::validity::Validity;
    use vortex_array::variants::StructArrayTrait;
    use vortex_array::{Array, ArrayRef, ToCanonical};
    use vortex_dtype::{DType, FieldNames, Nullability};

    use crate::convert::array::data_chunk_adaptor::NamedDataChunk;
    use crate::convert::array::to_duckdb_chunk;
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
        to_duckdb_chunk(&struct_arr, &mut output_chunk).unwrap();

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
        to_duckdb_chunk(&st, &mut output_chunk).unwrap();

        assert_eq!(
            format!("{:?}", output_chunk),
            "Chunk - [2 Columns]\n- CONSTANT BIGINT: 100 = [ 23444233]\n- CONSTANT INTEGER: 100 = [ 234]\n"
        )
    }
}
