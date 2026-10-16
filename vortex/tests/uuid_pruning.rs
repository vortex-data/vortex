// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `vortex.uuid` columns carry min/max statistics in their zone maps and file footer, so an
//! equality filter on a UUID prunes the zones and files that cannot hold it.

#![expect(clippy::tests_outside_test_module)]

use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::validity::Validity;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::extension::ExtDType;
use vortex::encodings::uuid::Uuid;
use vortex::encodings::uuid::UuidMetadata;
use vortex::error::VortexResult;
use vortex::expr::eq;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::expr::stats::Stat;
use vortex::expr::stats::StatsProvider;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::mask::Mask;
use vortex::session::VortexSession;

const ZONE_LEN: usize = 4;
const ZONES: usize = 4;

/// A non-nullable `vortex.uuid` array of `uuids`, each stored as its big-endian bytes.
fn uuid_array(uuids: impl IntoIterator<Item = u128>) -> VortexResult<ArrayRef> {
    let bytes = uuids
        .into_iter()
        .flat_map(u128::to_be_bytes)
        .collect::<PrimitiveArray>();
    let len = bytes.len() / 16;
    let storage = FixedSizeListArray::try_new(bytes.into_array(), 16, Validity::NonNullable, len)?;
    let dtype =
        ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage.dtype().clone())?.erased();
    Ok(ExtensionArray::new(dtype, storage.into_array()).into_array())
}

#[tokio::test]
async fn uuid_equality_prunes_zones_and_files() -> VortexResult<()> {
    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();

    // UUIDs increase with the row, so every zone holds a disjoint range.
    let ids = uuid_array((0..ZONES * ZONE_LEN).map(|row| (row as u128) << 120 | 0x4000))?;
    let table = StructArray::from_fields(&[("id", ids.clone())])?.into_array();

    let mut buf = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(
            WriteStrategyBuilder::from_session(&session)
                .with_row_block_size(ZONE_LEN)
                .build(),
        )
        .write(&mut buf, table.to_array_stream())
        .await?;
    let file = session.open_options().open_buffer(buf)?;

    // The footer bounds the column by its first and last UUID.
    let (stats, dtype) = file.file_stats().expect("written files have stats").get(0);
    let stats = stats.as_typed_ref(dtype);
    assert_eq!(
        stats.get(Stat::Min).as_exact(),
        Some(ids.execute_scalar(0, &mut ctx)?)
    );
    assert_eq!(
        stats.get(Stat::Max).as_exact(),
        Some(ids.execute_scalar(ids.len() - 1, &mut ctx)?)
    );

    // Zone maps keep only the zone that can hold the UUID.
    let target_row = 2 * ZONE_LEN + 1;
    let target = ids.execute_scalar(target_row, &mut ctx)?;
    let filter = eq(get_item("id", root()), lit(target.clone()))
        .bind(file.dtype())?
        .optimize_recursive()?;
    let row_count = file.row_count();
    let kept = file
        .layout_reader()?
        .pruning_evaluation(&(0..row_count), &filter, Mask::new_true(ids.len()))?
        .await?;
    assert_eq!(
        kept,
        Mask::from_iter((0..ids.len()).map(|row| row / ZONE_LEN == target_row / ZONE_LEN))
    );

    let matched = file
        .scan()?
        .with_filter(filter)
        .into_array_stream()?
        .read_all()
        .await?;
    assert_eq!(matched.len(), 1);
    assert_eq!(
        matched.execute_scalar(0, &mut ctx)?,
        table.execute_scalar(target_row, &mut ctx)?
    );

    // File statistics alone prune a UUID beyond the column's range, but not one inside it.
    let absent = uuid_array([u128::MAX])?.execute_scalar(0, &mut ctx)?;
    assert!(file.can_prune(&eq(get_item("id", root()), lit(absent)))?);
    assert!(!file.can_prune(&eq(get_item("id", root()), lit(target)))?);
    Ok(())
}
