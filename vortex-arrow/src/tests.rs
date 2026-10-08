// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::MapArray as ArrowMapArray;
use arrow_array::builder::Int32Builder;
use arrow_array::builder::Int64Builder;
use arrow_array::builder::MapBuilder;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute as _;
use vortex_array::array_session;
use vortex_array::arrays::ListView;
use vortex_array::arrays::Map;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_error::VortexResult;

use crate::ArrowSession;
use crate::convert::from_arrow_dyn;

fn import(array: ArrowArrayRef, nullable: bool, use_session: bool) -> VortexResult<ArrayRef> {
    if use_session {
        ArrowSession::default().from_arrow_array(array, nullable)
    } else {
        from_arrow_dyn(array.as_ref(), nullable)
    }
}

#[rstest]
#[case::prefix(0, 2)]
#[case::middle(500, 2)]
#[case::empty(500, 0)]
#[case::empty_at_end(1000, 0)]
fn test_sliced_map_imports_only_its_entries(
    #[values(false, true)] use_session: bool,
    #[values(false, true)] nullable: bool,
    #[case] start: usize,
    #[case] len: usize,
) -> VortexResult<()> {
    let maps = |rows: Range<i32>| -> VortexResult<ArrowMapArray> {
        let mut builder = MapBuilder::new(None, Int32Builder::new(), Int64Builder::new());
        for row in rows {
            builder.keys().append_value(row);
            builder.values().append_value(i64::from(row));
            builder.keys().append_value(row + 1);
            builder.values().append_value(i64::from(row + 1));
            builder.append(!nullable || row % 2 != 0)?;
        }
        Ok(builder.finish())
    };
    let sliced = import(
        Arc::new(maps(0..1000)?.slice(start, len)),
        nullable,
        use_session,
    )?;
    let rows = i32::try_from(start)?..i32::try_from(start + len)?;
    let fresh = import(Arc::new(maps(rows)?), nullable, use_session)?;
    // Empty nullable arrays can use a constant validity marker.
    if len != 0 || !nullable {
        assert_eq!(sliced.nbytes(), fresh.nbytes());
    }
    let map = sliced.as_::<Map>();
    let entries = map.entries().as_::<ListView>();
    assert_eq!(entries.elements().len(), len * 2);
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(sliced, fresh, &mut ctx);
    Ok(())
}
