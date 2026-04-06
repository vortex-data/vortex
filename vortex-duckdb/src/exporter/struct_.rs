// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayParts;
use vortex::array::builtins::ArrayBuiltins;
use vortex::error::VortexResult;

use crate::duckdb::LogicalType;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::all_invalid;
use crate::exporter::new_array_exporter;
use crate::exporter::validity;

struct StructExporter {
    children: Vec<Box<dyn ColumnExporter>>,
}

pub(crate) fn new_exporter(
    array: StructArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let len = array.len();
    let StructArrayParts {
        validity,
        struct_fields,
        fields,
        ..
    } = array.into_parts();
    let validity = validity.to_array(len).execute::<BoolArray>(ctx)?;

    if validity.to_bit_buffer().true_count() == 0 {
        let ltype = LogicalType::try_from(struct_fields)?;
        return Ok(all_invalid::new_exporter(len, &ltype));
    }

    let children = fields
        .iter()
        .map(|child| {
            if validity.to_bit_buffer().true_count() != validity.len() {
                // TODO(joe): use new mask.
                new_array_exporter(
                    child.clone().mask(validity.clone().into_array())?,
                    cache,
                    ctx,
                )
            } else {
                new_array_exporter(child.clone().into_array(), cache, ctx)
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(validity::new_exporter(
        validity.to_mask(),
        Box::new(StructExporter { children }),
    ))
}

impl ColumnExporter for StructExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for (idx, child) in self.children.iter().enumerate() {
            child.export(offset, len, vector.struct_vector_get_child_mut(idx), ctx)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::ConstantArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::BitBuffer;
    use vortex::buffer::buffer;
    use vortex::error::VortexExpect;

    use super::*;
    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    #[test]
    fn test_struct_exporter() {
        let prim = PrimitiveArray::from_iter(0..10).into_array();
        let strings =
            VarBinViewArray::from_iter_str(vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
                .into_array();
        let arr =
            StructArray::from_fields(&[("a", prim), ("b", strings)]).vortex_expect("struct array");
        let mut chunk = DataChunk::new([LogicalType::struct_type(
            vec![
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER),
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_VARCHAR),
            ],
            vec![CString::new("col1").unwrap(), CString::new("col2").unwrap()],
        )
        .vortex_expect("LogicalTypeRef creation should succeed for test data")]);

        let mut ctx = SESSION.create_execution_ctx();
        new_exporter(arr, &ConversionCache::default(), &mut ctx)
            .unwrap()
            .export(0, 10, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(10);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT STRUCT(col1 INTEGER, col2 VARCHAR): 10 = [ {'col1': 0, 'col2': a}, {'col1': 1, 'col2': b}, {'col1': 2, 'col2': c}, {'col1': 3, 'col2': d}, {'col1': 4, 'col2': e}, {'col1': 5, 'col2': f}, {'col1': 6, 'col2': g}, {'col1': 7, 'col2': h}, {'col1': 8, 'col2': i}, {'col1': 9, 'col2': j}]
"#
        );
    }

    #[test]
    fn test_struct_exporter_with_nulls() {
        let prim = PrimitiveArray::from_option_iter([
            Some(1),
            None,
            Some(2),
            None,
            Some(3),
            None,
            Some(4),
            None,
            Some(5),
            None,
        ])
        .into_array();
        let strings = VarBinViewArray::from_iter_nullable_str(vec![
            None,
            Some("b"),
            Some("c"),
            Some("d"),
            None,
            None,
            Some("g"),
            Some("h"),
            None,
            Some("j"),
        ])
        .into_array();
        let arr = StructArray::try_new(
            ["col1", "col2"].into(),
            vec![prim, strings],
            10,
            Validity::from(BitBuffer::from_iter([
                true, true, true, false, false, false, true, true, true, true,
            ])),
        )
        .vortex_expect("StructArray creation should succeed for test data");
        let mut chunk = DataChunk::new([LogicalType::struct_type(
            vec![
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER),
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_VARCHAR),
            ],
            vec![CString::new("col1").unwrap(), CString::new("col2").unwrap()],
        )
        .vortex_expect("LogicalTypeRef creation should succeed for test data")]);

        let mut ctx = SESSION.create_execution_ctx();
        new_exporter(arr, &ConversionCache::default(), &mut ctx)
            .unwrap()
            .export(0, 10, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(10);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT STRUCT(col1 INTEGER, col2 VARCHAR): 10 = [ {'col1': 1, 'col2': NULL}, {'col1': NULL, 'col2': b}, {'col1': 2, 'col2': c}, NULL, NULL, NULL, {'col1': 4, 'col2': g}, {'col1': NULL, 'col2': h}, {'col1': 5, 'col2': NULL}, {'col1': NULL, 'col2': j}]
"#
        );
    }

    #[test]
    fn struct_export_non_flat_vectors() {
        let prim = ConstantArray::new(42, 10).into_array();
        let strings = DictArray::try_new(
            buffer![0u8, 1, 1, 2, 2, 2, 2, 3, 3, 4].into_array(),
            VarBinViewArray::from_iter_str(vec!["b", "c", "d", "g", "h"]).into_array(),
        )
        .vortex_expect("DictArray creation should succeed for test data")
        .into_array();
        let arr = StructArray::try_new(
            ["col1", "col2"].into(),
            vec![prim, strings],
            10,
            Validity::from(BitBuffer::from_iter([
                true, true, true, false, false, false, true, true, true, true,
            ])),
        )
        .vortex_expect("StructArray creation should succeed for test data");
        let mut chunk = DataChunk::new([LogicalType::struct_type(
            vec![
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER),
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_VARCHAR),
            ],
            vec![CString::new("col1").unwrap(), CString::new("col2").unwrap()],
        )
        .vortex_expect("LogicalTypeRef creation should succeed for test data")]);

        let mut ctx = SESSION.create_execution_ctx();
        new_exporter(arr, &ConversionCache::default(), &mut ctx)
            .unwrap()
            .export(0, 10, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(10);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT STRUCT(col1 INTEGER, col2 VARCHAR): 10 = [ {'col1': 42, 'col2': b}, {'col1': 42, 'col2': c}, {'col1': 42, 'col2': c}, NULL, NULL, NULL, {'col1': 42, 'col2': d}, {'col1': 42, 'col2': g}, {'col1': 42, 'col2': g}, {'col1': 42, 'col2': h}]
"#
        );
    }
}
