// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::GenericListViewArray;
use arrow_array::Int64Array;
use arrow_array::MapArray as ArrowMapArray;
use arrow_array::builder::Int32Builder;
use arrow_array::builder::Int64Builder;
use arrow_array::builder::MapBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
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
    let fresh = import(
        Arc::new(maps(start as i32..(start + len) as i32)?),
        nullable,
        use_session,
    )?;
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

#[rstest]
#[case::overlap(vec![502, 500, 501], vec![2, 2, 2], 500..504)]
#[case::gap(vec![500, 510], vec![2, 2], 500..512)]
#[case::empty_lists(vec![0, 500, 1000], vec![0, 2, 0], 500..502)]
#[case::all_empty(vec![200, 900], vec![0, 0], 0..0)]
#[case::empty_slice(vec![], vec![], 0..0)]
fn test_sliced_list_view_trims_child_bounds(
    #[values(false, true)] use_session: bool,
    #[values(false, true)] large: bool,
    #[case] offsets: Vec<i64>,
    #[case] sizes: Vec<i64>,
    #[case] referenced: Range<usize>,
) -> VortexResult<()> {
    let views = |offsets: Vec<i64>, sizes: Vec<i64>, values: ArrowArrayRef| -> ArrowArrayRef {
        let field = Arc::new(Field::new("item", DataType::Int64, false));
        if large {
            Arc::new(GenericListViewArray::<i64>::new(
                field,
                offsets.into(),
                sizes.into(),
                values,
                None,
            ))
        } else {
            Arc::new(GenericListViewArray::<i32>::new(
                field,
                offsets.into_iter().map(|offset| offset as i32).collect(),
                sizes.into_iter().map(|size| size as i32).collect(),
                values,
                None,
            ))
        }
    };
    let values: ArrowArrayRef = Arc::new(Int64Array::from_iter_values(0..1000));
    let parent_offsets = std::iter::once(0)
        .chain(offsets.iter().copied())
        .chain(std::iter::once(999))
        .collect();
    let parent_sizes = std::iter::once(1)
        .chain(sizes.iter().copied())
        .chain(std::iter::once(1))
        .collect();
    let parent = views(parent_offsets, parent_sizes, values.clone());
    let sliced = import(parent.slice(1, offsets.len()), false, use_session)?;
    let fresh_offsets = offsets
        .iter()
        .zip(sizes.iter())
        .map(|(&offset, &size)| {
            if size == 0 {
                0
            } else {
                offset - referenced.start as i64
            }
        })
        .collect();
    let fresh = import(
        views(
            fresh_offsets,
            sizes,
            values.slice(referenced.start, referenced.len()),
        ),
        false,
        use_session,
    )?;
    assert_eq!(sliced.nbytes(), fresh.nbytes());
    assert_eq!(sliced.as_::<ListView>().elements().len(), referenced.len());
    let mut ctx = array_session().create_execution_ctx();
    assert_arrays_eq!(sliced, fresh, &mut ctx);
    Ok(())
}
