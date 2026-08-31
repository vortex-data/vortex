//! End-to-end tests: write a column indexed by [`ReverseIndex`], then probe it.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::col;
use vortex_array::expr::eq;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::indexed::INDEXED_LAYOUT_ID;
use vortex_layout::layouts::indexed::IndexConfig;
use vortex_layout::layouts::indexed::IndexSessionExt;
use vortex_layout::layouts::indexed::IndexedStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ReverseIndex;

/// Small enough that a 12-row file spans three blocks, making the row/block granularity
/// difference visible in a single assertion.
const BLOCK_LEN: usize = 4;

const VALUE_FIELD: &str = "value";

/// `20` appears at rows 1 and 9; nothing else repeats. With `BLOCK_LEN` of 4 those land in blocks
/// 0 and 2, leaving block 1 prunable. `999` never appears, to exercise the "claimed but no match"
/// path distinctly from "no index claimed this at all".
const VALUES: [i32; 12] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 20, 100, 110];

fn reverse_index_configs() -> Vec<IndexConfig> {
    vec![IndexConfig::with_defaults(ReverseIndex::new_ref())]
}

/// The array/layout encodings these tests need to write.
///
/// The default Vortex file writer only permits array/layout ids covered by the session's enabled
/// editions, but those first-party declarations live in the `vortex` facade crate, which this
/// out-of-tree crate deliberately does not depend on. Declaring and enabling a tiny test-only
/// edition here is the local equivalent.
const TEST_EDITION: EditionId = EditionId::new("vortex-reverse-index-test", 2026, 1, 0);

static TEST_EDITION_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: TEST_EDITION,
        min_vortex_version: None,
    },
    added: &[
        EditionMember::array(&"vortex.struct"),
        EditionMember::array(&"vortex.primitive"),
        // The index child's postings column is serialized roaring bitmaps, written as varbinview.
        EditionMember::array(&"vortex.varbinview"),
        EditionMember::layout(&"vortex.struct"),
        EditionMember::layout(&"vortex.chunked"),
        EditionMember::layout(&"vortex.flat"),
        EditionMember::layout(&INDEXED_LAYOUT_ID),
    ],
};

/// A session knowing the `vortex.indexed` layout and the reverse index.
///
/// The `vortex.indexed` layout is registered by default in every [`LayoutSession`], the same as
/// any other built-in layout, so only the edition declaration needs setting up here.
///
/// Deliberately not a shared global session: sessions clone by sharing one `Arc`, so registering
/// into a shared session would leak between tests, and
/// [`unregistered_index_kind_falls_back_to_the_data_child`] depends on two sessions with different
/// index registries.
fn session() -> VortexSession {
    let session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with::<EditionSession>();
    session
        .register_edition(&TEST_EDITION_DECLARATION)
        .expect("test edition declaration should be valid");
    session
        .enable_edition(TEST_EDITION)
        .expect("test edition was just registered");
    session
}

/// A session that additionally knows the reverse index.
fn session_with_reverse_index() -> VortexSession {
    let session = session();
    session.indexes().register(ReverseIndex::new_ref());
    session
}

fn value_column() -> VortexResult<ArrayRef> {
    let values: PrimitiveArray = VALUES.into_iter().collect();
    struct_column(values.into_array())
}

fn struct_column(value_column: ArrayRef) -> VortexResult<ArrayRef> {
    Ok(StructArray::from_fields([(VALUE_FIELD, value_column)].as_slice())?.into_array())
}

/// A write strategy that attaches `configs` to the value column.
///
/// The indexed wrapper sits directly above repartitioning — the same slot `ZonedStrategy`
/// occupies — so it sees whole chunks in row order and knows the data child's block size.
fn strategy(configs: Vec<IndexConfig>) -> Arc<dyn LayoutStrategy> {
    let data = RepartitionStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        RepartitionWriterOptions {
            block_size_minimum: 0,
            block_len_multiple: BLOCK_LEN,
            block_size_target: None,
            canonicalize: false,
        },
    );
    let indexed = IndexedStrategy::new(data, FlatLayoutStrategy::default(), configs)
        .with_data_block_len(BLOCK_LEN as u64);

    WriteStrategyBuilder::default()
        .with_row_block_size(BLOCK_LEN)
        .with_field_writer(FieldPath::from_name(VALUE_FIELD), Arc::new(indexed))
        .build()
}

async fn write_file(
    session: &VortexSession,
    configs: Vec<IndexConfig>,
) -> VortexResult<ByteBuffer> {
    write_file_with_column(session, configs, value_column()?).await
}

async fn write_file_with_column(
    session: &VortexSession,
    configs: Vec<IndexConfig>,
    column: ArrayRef,
) -> VortexResult<ByteBuffer> {
    let mut bytes = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(strategy(configs))
        .write(&mut bytes, column.to_array_stream())
        .await?;
    Ok(bytes.freeze())
}

fn value_reader(file: &VortexFile) -> VortexResult<Arc<dyn vortex_layout::LayoutReader>> {
    let value_layout = file
        .footer()
        .layout()
        .slot(1)?
        .vortex_expect("root struct always has the value column");
    value_layout.new_reader(
        VALUE_FIELD.into(),
        file.segment_source(),
        file.session(),
        &Default::default(),
    )
}

/// The mask the value column's reader produces for `value == target`.
///
/// An `Exact` plan serves `filter_evaluation` directly, so the result is the index's own answer
/// rather than the data child's, intersected with `input`.
async fn exact_mask(file: &VortexFile, target: i32, input: MaskFuture) -> VortexResult<Mask> {
    let reader = value_reader(file)?;
    let row_count = reader.row_count();
    let filter = eq(root(), lit(target)).bind(reader.dtype())?;

    reader
        .filter_evaluation(&(0..row_count), &filter, input)?
        .await
}

async fn scan_matching(file: &VortexFile, target: i32) -> VortexResult<Vec<i32>> {
    let filter = eq(col(VALUE_FIELD), lit(target)).bind(file.dtype())?;
    let result = file
        .scan()?
        .with_filter(filter)
        .into_array_stream()?
        .read_all()
        .await?;

    let mut ctx = file.session().create_execution_ctx();
    let values = result
        .execute::<StructArray>(&mut ctx)?
        .unmasked_field_by_name(VALUE_FIELD)?
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)?;

    Ok(values.as_slice::<i32>().to_vec())
}

#[tokio::test]
async fn unregistered_index_kind_falls_back_to_the_data_child() -> VortexResult<()> {
    // Written by a session that knows the reverse index...
    let bytes = write_file(&session_with_reverse_index(), reverse_index_configs()).await?;

    // ...and read by one that does not. The spec goes inert, nothing probes the index child, and
    // the data child answers everything — indexes are strictly optional accelerators.
    let file = session().open_options().open_buffer(bytes)?;

    assert_eq!(scan_matching(&file, 20).await?, vec![20, 20]);
    Ok(())
}

#[tokio::test]
async fn exact_index_answers_the_filter_itself() -> VortexResult<()> {
    let session = session_with_reverse_index();
    let bytes = write_file(&session, reverse_index_configs()).await?;
    let file = session.open_options().open_buffer(bytes)?;

    let mask = exact_mask(&file, 20, MaskFuture::new_true(VALUES.len())).await?;
    assert_eq!(
        mask,
        Mask::from_iter((0..VALUES.len()).map(|row| VALUES[row] == 20))
    );

    // The post-condition is that the result is intersected with the input mask, so an input that
    // excludes both matches must yield nothing.
    let excluded = exact_mask(
        &file,
        20,
        MaskFuture::ready(Mask::from_iter(
            (0..VALUES.len()).map(|row| VALUES[row] != 20),
        )),
    )
    .await?;
    assert!(excluded.all_false());

    Ok(())
}

#[tokio::test]
async fn absent_value_resolves_to_no_matches() -> VortexResult<()> {
    let session = session_with_reverse_index();
    let bytes = write_file(&session, reverse_index_configs()).await?;
    let file = session.open_options().open_buffer(bytes)?;

    // The index claims the expression (it is an equality over an integer column) but finds no
    // posting for 999, distinct from "no index claimed this at all".
    let mask = exact_mask(&file, 999, MaskFuture::new_true(VALUES.len())).await?;
    assert!(mask.all_false());
    assert_eq!(scan_matching(&file, 999).await?, Vec::<i32>::new());
    Ok(())
}

#[tokio::test]
async fn layout_carries_one_auxiliary_child_per_index() -> VortexResult<()> {
    let session = session();
    let bytes = write_file(&session, reverse_index_configs()).await?;
    let file = session.open_options().open_buffer(bytes)?;

    let value_layout = file
        .footer()
        .layout()
        .slot(1)?
        .vortex_expect("root struct always has the value column");
    assert_eq!(value_layout.encoding_id().as_str(), INDEXED_LAYOUT_ID);

    assert_eq!(
        (0..value_layout.nslots())
            .filter_map(|slot| value_layout.slot_type(slot))
            .collect::<Vec<_>>(),
        vec![
            LayoutChildType::Transparent("data".into()),
            LayoutChildType::Auxiliary(format!("index:{}", crate::REVERSE_INDEX_ID).into()),
        ],
    );

    // Index content is an ordinary layout tree, so it inherits chunking and zone maps for free.
    let index_child = value_layout
        .slot(1)?
        .vortex_expect("a reverse index was configured");
    assert_eq!(
        index_child
            .dtype()
            .as_struct_fields()
            .names()
            .iter()
            .map(|name| name.to_string())
            .collect::<Vec<_>>(),
        vec!["key".to_string(), "postings".to_string()],
    );
    // 11 distinct values across 12 rows (20 repeats once).
    assert_eq!(index_child.row_count(), 11);
    Ok(())
}

/// The reverse index is not limited to integer columns: it decodes keys as
/// [`Scalar`](vortex_array::scalar::Scalar)s, so any dtype the layout can carry works the same
/// way.
#[tokio::test]
async fn string_valued_column_is_indexed_exactly() -> VortexResult<()> {
    const STRING_VALUES: [&str; 6] = ["b", "a", "b", "c", "a", "d"];

    let session = session_with_reverse_index();
    let values = VarBinViewArray::from_iter_str(STRING_VALUES);
    let bytes = write_file_with_column(
        &session,
        reverse_index_configs(),
        struct_column(values.into_array())?,
    )
    .await?;
    let file = session.open_options().open_buffer(bytes)?;

    let reader = value_reader(&file)?;
    let row_count = reader.row_count();
    let filter = eq(root(), lit("b")).bind(reader.dtype())?;
    let mask = reader
        .filter_evaluation(
            &(0..row_count),
            &filter,
            MaskFuture::new_true(STRING_VALUES.len()),
        )?
        .await?;

    assert_eq!(
        mask,
        Mask::from_iter(STRING_VALUES.iter().map(|value| *value == "b"))
    );
    Ok(())
}
