mod array_ref;
mod cache;
mod chunked;
mod data_chunk_adaptor;
mod dict;
mod run_end;
mod table;
mod validity;
mod varbinview;

pub use cache::ConversionCache;
pub use data_chunk_adaptor::NamedDataChunk;
use duckdb::vtab::arrow::WritableVector;
pub use table::to_duckdb_chunk;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

/// Takes an array `self` and a target `chunk` (a duckdb vector), and writes the values from `self`
/// into `chunk`.
/// An `cache` is also provided which can optionally be used to store intermediate expensive
/// to compute values in.
/// The capacity of the vector must be non-strictly larger that the len of the struct array.
pub trait ToDuckDB {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()>;
}

/// Takes a duckdb `vector` and returns a vortex array.
pub trait FromDuckDB<V> {
    fn from_duckdb(vector: V) -> VortexResult<ArrayRef>;
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

    use crate::convert::array::ConversionCache;
    use crate::convert::array::data_chunk_adaptor::NamedDataChunk;
    use crate::{FromDuckDB, ToDuckDBType, to_duckdb_chunk};

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
