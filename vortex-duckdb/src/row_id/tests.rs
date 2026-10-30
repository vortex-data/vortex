// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;

use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::StructArray;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::scan::selection::Selection;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;

use super::RowIds;
use super::encode;
use super::replace_row_id;
use crate::SESSION;
use crate::cpp;
use crate::duckdb::DataChunk;
use crate::duckdb::LogicalType;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;

#[test]
fn row_ids_select_only_the_rows_belonging_to_each_file() -> VortexResult<()> {
    let ids = RowIds(BTreeSet::from([
        encode(2, 7)?,
        encode(9, 12)?,
        encode(2, 99)?,
    ]));
    for (file, expected) in [(2, vec![7, 99]), (9, vec![12]), (3, vec![])] {
        let Selection::IncludeByIndex(rows) = ids.selection(file, 100, &Selection::All)? else {
            vortex_bail!("expected an index selection");
        };
        assert_eq!(rows.as_ref(), expected.as_slice());
    }
    let selection =
        Selection::IncludeByIndex(StrictSortedBuffer::try_new(Buffer::from_iter([99u64]))?);
    let Selection::IncludeByIndex(rows) = ids.selection(2, 50, &selection)? else {
        vortex_bail!("expected an index selection");
    };
    assert!(rows.is_empty());
    Ok(())
}

#[test]
fn packed_ids_preserve_the_entire_row_number() -> VortexResult<()> {
    assert_eq!(encode(0, u64::MAX)?, i128::from(u64::MAX));
    assert!(encode(1, 0)? > encode(0, u64::MAX)?);
    assert_ne!(encode(1, 7)?, encode(7, 1)?);
    assert!(encode(u64::MAX, 0).is_err());
    Ok(())
}

#[test]
fn packed_ids_export_as_duckdb_hugeints() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = StructArray::from_fields(&[("id", Buffer::from_iter([7u64, 12]).into_array())])?;
    let array = replace_row_id(array, 0, 2, &mut ctx)?;
    let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_HUGEINT)]);
    let mut exporter = ArrayExporter::try_new(&array, &ConversionCache::default(), ctx)?;
    exporter.export(&mut chunk, None)?;
    assert_eq!(
        String::try_from(&*chunk)?,
        format!(
            "Chunk - [1 Columns]\n- FLAT HUGEINT: 2 = [ {}, {}]\n",
            encode(2, 7)?,
            encode(2, 12)?
        )
    );
    Ok(())
}
