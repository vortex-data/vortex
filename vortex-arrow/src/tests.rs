// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::DurationMicrosecondArray;
use arrow_array::DurationMillisecondArray;
use arrow_array::DurationNanosecondArray;
use arrow_array::DurationSecondArray;
use arrow_array::MapArray as ArrowMapArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::builder::Int32Builder;
use arrow_array::builder::Int64Builder;
use arrow_array::builder::MapBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::TimeUnit as ArrowTimeUnit;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute as _;
use vortex_array::arrays::ListView;
use vortex_array::arrays::Map;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::Duration;
use vortex_array::extension::datetime::TimeUnit;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrowSessionExt;
use crate::ExecuteArrowExt as _;
use crate::convert::from_arrow_dyn;

fn import(array: ArrowArrayRef, nullable: bool, use_session: bool) -> VortexResult<ArrayRef> {
    if use_session {
        VortexSession::empty()
            .arrow()
            .from_arrow_array(array, nullable)
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

    assert_eq!(sliced.nbytes(), fresh.nbytes());

    let map = sliced.as_::<Map>();
    let entries = map.entries().as_::<ListView>();
    assert_eq!(entries.elements().len(), len * 2);

    let mut ctx = VortexSession::empty().create_execution_ctx();
    assert_arrays_eq!(sliced, fresh, &mut ctx);

    Ok(())
}

const DURATION_VALUES: [Option<i64>; 4] = [Some(i64::MIN), Some(42), Some(i64::MAX), None];

#[rstest]
#[case::seconds(TimeUnit::Seconds, Arc::new(DurationSecondArray::from(DURATION_VALUES.to_vec())))]
#[case::milliseconds(TimeUnit::Milliseconds, Arc::new(DurationMillisecondArray::from(DURATION_VALUES.to_vec())))]
#[case::microseconds(TimeUnit::Microseconds, Arc::new(DurationMicrosecondArray::from(DURATION_VALUES.to_vec())))]
#[case::nanoseconds(TimeUnit::Nanoseconds, Arc::new(DurationNanosecondArray::from(DURATION_VALUES.to_vec())))]
fn test_native_duration_round_trip(
    #[case] unit: TimeUnit,
    #[case] input: ArrowArrayRef,
    #[values(false, true)] nullable: bool,
) -> VortexResult<()> {
    let session = VortexSession::empty();
    let arrow = session.arrow();
    let mut ctx = session.create_execution_ctx();
    let input = if nullable {
        input
    } else {
        input.slice(0, input.len() - 1)
    };
    let field = Field::new("elapsed", input.data_type().clone(), nullable);
    let expected = DType::Extension(Duration::try_new(unit, Nullability::from(nullable))?.erased());
    assert_eq!(expected.as_extension().id().as_str(), "vortex.duration");

    assert_eq!(arrow.from_arrow_field(&field)?, expected);
    let imported = arrow.from_arrow_array(Arc::clone(&input), &field)?;
    assert_eq!(imported.dtype(), &expected);
    assert_eq!(arrow.to_arrow_field("elapsed", imported.dtype())?, field);
    assert_eq!(
        imported.clone().execute_arrow(None, &mut ctx)?.as_ref(),
        input.as_ref()
    );
    assert_eq!(
        imported.execute_arrow(Some(&field), &mut ctx)?.as_ref(),
        input.as_ref()
    );

    let nested: ArrowArrayRef = Arc::new(ArrowStructArray::try_new(
        vec![field].into(),
        vec![input],
        None,
    )?);
    let imported = arrow.from_arrow_array(Arc::clone(&nested), false)?;
    assert_eq!(
        imported.execute_arrow(None, &mut ctx)?.as_ref(),
        nested.as_ref()
    );
    Ok(())
}

#[test]
fn test_duration_export_rejects_a_different_unit() -> VortexResult<()> {
    let session = VortexSession::empty();
    let array: ArrowArrayRef = Arc::new(DurationMillisecondArray::from(vec![42]));
    let imported = session.arrow().from_arrow_array(array, false)?;
    let target = Field::new("elapsed", DataType::Duration(ArrowTimeUnit::Second), false);
    assert!(
        imported
            .execute_arrow(Some(&target), &mut session.create_execution_ctx())
            .is_err()
    );
    Ok(())
}
