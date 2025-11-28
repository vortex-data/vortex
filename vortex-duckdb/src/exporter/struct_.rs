// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::arrays::StructArray;
use vortex::compute::mask;
use vortex::error::VortexResult;

use crate::duckdb::Vector;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::new_array_exporter;
use crate::exporter::validity;

struct StructExporter {
    children: Vec<Box<dyn ColumnExporter>>,
}

pub(crate) fn new_exporter(
    array: &StructArray,
    cache: &ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let validity = array.validity_mask();
    // DuckDB requires that the validity of the child be a subset of the parent struct so we mask out children with
    // parents nullability
    let validity_for_mask = array.dtype().is_nullable().then(|| !&validity);

    let children = array
        .fields()
        .iter()
        .map(|child| {
            if let Some(mv) = validity_for_mask.as_ref() {
                new_array_exporter(&mask(child, mv)?, cache)
            } else {
                new_array_exporter(child, cache)
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let struct_exporter = Box::new(StructExporter { children });
    Ok(if array.dtype().is_nullable() {
        validity::new_exporter(validity, struct_exporter)
    } else {
        struct_exporter
    })
}

impl ColumnExporter for StructExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        for (idx, child) in self.children.iter().enumerate() {
            child.export(offset, len, &mut vector.struct_vector_get_child(idx))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use vortex::array::IntoArray;
    use vortex::array::arrays::ConstantArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::BitBuffer;
    use vortex::buffer::buffer;
    use vortex::error::VortexExpect;
    use vortex::error::VortexUnwrap;

    use super::*;
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
        .vortex_unwrap()]);

        new_exporter(&arr, &ConversionCache::default())
            .unwrap()
            .export(0, 10, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(10);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
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
        .vortex_unwrap();
        let mut chunk = DataChunk::new([LogicalType::struct_type(
            vec![
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER),
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_VARCHAR),
            ],
            vec![CString::new("col1").unwrap(), CString::new("col2").unwrap()],
        )
        .vortex_unwrap()]);

        new_exporter(&arr, &ConversionCache::default())
            .unwrap()
            .export(0, 10, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(10);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
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
        .vortex_unwrap()
        .into_array();
        let arr = StructArray::try_new(
            ["col1", "col2"].into(),
            vec![prim, strings],
            10,
            Validity::from(BitBuffer::from_iter([
                true, true, true, false, false, false, true, true, true, true,
            ])),
        )
        .vortex_unwrap();
        let mut chunk = DataChunk::new([LogicalType::struct_type(
            vec![
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER),
                LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_VARCHAR),
            ],
            vec![CString::new("col1").unwrap(), CString::new("col2").unwrap()],
        )
        .vortex_unwrap()]);

        new_exporter(&arr, &ConversionCache::default())
            .unwrap()
            .export(0, 10, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(10);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT STRUCT(col1 INTEGER, col2 VARCHAR): 10 = [ {'col1': 42, 'col2': b}, {'col1': 42, 'col2': c}, {'col1': 42, 'col2': c}, NULL, NULL, NULL, {'col1': 42, 'col2': d}, {'col1': 42, 'col2': g}, {'col1': 42, 'col2': g}, {'col1': 42, 'col2': h}]
"#
        );
    }
}
