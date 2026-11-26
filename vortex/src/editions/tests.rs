// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::session::ArraySessionExt;
use vortex_array::stats::Stat;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_edition::ComponentKind;
use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionMember;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::test_harness::validate_edition;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::col;
use vortex_expr::eq;
use vortex_expr::gt_eq;
use vortex_expr::lit;
use vortex_expr::lt;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::FoRReferences;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_pivco::Pivco;
use vortex_pivco::PivcoOptions;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;
use vortex_session::registry::Id;
use vortex_zigzag::ZigZag;
use vortex_zigzag::ZigZagArraySlotsExt;
use vortex_zigzag::zigzag_encode;

use super::CORE_2025_05_0;
use super::CORE_2025_10_0;
use super::CORE_2026_08_0;
use super::CORE_2026_08_1;
use super::CORE_2026_08_2;
use super::CORE_2026_08_3;
use super::DEFAULT_CORE_EDITION;
use super::DEFAULT_PREVIEW_EDITION;
use super::EDITION_DECLARATIONS;
use super::UNSTABLE;
use crate::VortexSessionDefault;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

fn session() -> VortexResult<EditionSession> {
    let session = EditionSession::empty();
    for family in super::EDITION_FAMILIES {
        session.declare_family(family)?;
    }
    for declaration in EDITION_DECLARATIONS {
        session.declare(declaration)?;
    }
    Ok(session)
}

#[test]
fn every_declared_edition_validates() -> VortexResult<()> {
    let session = session()?;
    for declaration in EDITION_DECLARATIONS {
        validate_edition(&session, &declaration.edition.id)?;
    }
    Ok(())
}

#[test]
fn core_2026_08_1_dtype_set_is_pinned() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    let dtypes = session.components_in(&CORE_2026_08_1, ComponentKind::DType);
    let ids: Vec<&str> = dtypes
        .iter()
        .map(|inclusion| inclusion.component_id.as_str())
        .collect();
    assert_eq!(ids, ["vortex.date", "vortex.time", "vortex.timestamp"]);
}

#[test]
fn core_2026_08_2_is_frozen() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    assert!(
        !session
            .find(&CORE_2026_08_2)
            .unwrap_or_else(|| panic!("{CORE_2026_08_2} is not registered"))
            .is_draft()
    );
    assert!(
        session
            .components_in(&CORE_2026_08_2, ComponentKind::Array)
            .iter()
            .any(|inclusion| inclusion.component_id.as_str() == "vortex.map")
    );
    assert!(
        session
            .components_in(&CORE_2026_08_1, ComponentKind::Array)
            .iter()
            .all(|inclusion| inclusion.component_id.as_str() != "vortex.map")
    );
}

#[test]
fn core_2026_08_3_is_frozen_and_adds_variants() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    assert!(
        !session
            .find(&CORE_2026_08_3)
            .unwrap_or_else(|| panic!("{CORE_2026_08_3} is not registered"))
            .is_draft()
    );
    assert!(
        session
            .components_in(&CORE_2026_08_3, ComponentKind::Array)
            .iter()
            .any(|inclusion| inclusion.component_id.as_str() == "vortex.variant")
    );
    assert!(
        session
            .components_in(&CORE_2026_08_3, ComponentKind::Array)
            .iter()
            .any(|inclusion| inclusion.component_id.as_str() == "vortex.parquet.variant")
    );
    assert!(
        session
            .components_in(&CORE_2026_08_2, ComponentKind::Array)
            .iter()
            .all(|inclusion| inclusion.component_id.as_str() != "vortex.parquet.variant")
    );
    assert!(
        session
            .components_in(&CORE_2026_08_3, ComponentKind::DType)
            .iter()
            .any(|inclusion| inclusion.component_id.as_str() == "vortex.uuid")
    );
}

#[test]
fn earlier_editions_are_subsets() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    let first = session.components_in(&CORE_2025_05_0, ComponentKind::Array);
    let latest = session.components_in(&DEFAULT_CORE_EDITION, ComponentKind::Array);
    assert!(first.iter().all(|inclusion| {
        latest
            .iter()
            .any(|latest| latest.component_id == inclusion.component_id)
    }));
    assert!(first.len() < latest.len());
}

#[test]
fn core_2026_08_editions_add_onpair_before_map() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    assert!(
        session
            .components_in(&CORE_2026_08_0, ComponentKind::Array)
            .iter()
            .all(|inclusion| inclusion.component_id.as_str() != "vortex.onpair")
    );
    assert!(
        session
            .components_in(&CORE_2026_08_0, ComponentKind::Layout)
            .iter()
            .any(|inclusion| inclusion.component_id.as_str() == "vortex.zoned")
    );
    assert!(
        session
            .components_in(&CORE_2026_08_0, ComponentKind::Aggregate)
            .iter()
            .any(|inclusion| inclusion.component_id.as_str() == "vortex.min")
    );

    assert!(
        session
            .components_in(&CORE_2026_08_1, ComponentKind::Array)
            .iter()
            .any(|inclusion| inclusion.component_id.as_str() == "vortex.onpair")
    );
    assert!(
        session
            .components_in(&CORE_2026_08_1, ComponentKind::Array)
            .iter()
            .all(|inclusion| inclusion.component_id.as_str() != "vortex.map")
    );
    assert!(
        session
            .components_in(&CORE_2026_08_2, ComponentKind::Array)
            .iter()
            .any(|inclusion| inclusion.component_id.as_str() == "vortex.map")
    );
}

#[test]
fn default_session_enables_the_write_editions() {
    let session = VortexSession::default();
    let enabled = session.enabled_editions().editions();
    assert!(enabled.contains(&DEFAULT_CORE_EDITION));
    assert!(
        session
            .enabled_component_ids(ComponentKind::Array)
            .contains(&Id::from("vortex.pco"))
    );

    assert!(!enabled.contains(&DEFAULT_PREVIEW_EDITION));
    assert!(!enabled.contains(&UNSTABLE));
}

#[test]
fn enabling_unstable_permits_only_persisted_aggregates() -> VortexResult<()> {
    let session = VortexSession::default();
    session.enable_edition(UNSTABLE)?;
    assert!(
        session
            .enabled_component_ids(ComponentKind::Aggregate)
            .iter()
            .all(|id| !id.as_str().starts_with("vortex.all_"))
    );
    Ok(())
}

#[test]
fn core_edition_ids_are_registered_array_encodings() {
    use vortex_array::session::ArraySessionExt;

    let session = VortexSession::default();
    let registry = session.arrays().registry().clone();
    for inclusion in session
        .editions()
        .components_in(&DEFAULT_CORE_EDITION, ComponentKind::Array)
    {
        assert!(
            registry.contains_key(&inclusion.component_id),
            "{} is declared in core but not registered as an array encoding",
            inclusion.component_id
        );
    }
}

#[test]
fn core_dtype_ids_are_registered_extension_dtypes() {
    use vortex_array::dtype::session::DTypeSessionExt;

    let session = VortexSession::default();
    let registry = session.dtypes().registry().clone();
    for inclusion in session
        .editions()
        .components_in(&DEFAULT_CORE_EDITION, ComponentKind::DType)
    {
        assert!(
            registry.contains_key(&inclusion.component_id),
            "{} is declared in core but not registered as an extension dtype",
            inclusion.component_id
        );
    }
}

/// A declared aggregate that no longer resolves would fail every write that records it, since
/// aggregates outside the enabled editions are rejected rather than dropped.
#[test]
fn core_aggregate_ids_are_registered_aggregate_fns() {
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;

    let session = VortexSession::default();
    let declared = session
        .editions()
        .components_in(&DEFAULT_CORE_EDITION, ComponentKind::Aggregate);
    assert!(
        declared
            .iter()
            .any(|inclusion| inclusion.component_id.as_str() == "vortex.min")
    );
    for inclusion in declared {
        assert!(
            session
                .aggregate_fns()
                .find_plugin(&inclusion.component_id)
                .is_some(),
            "{} is declared in core but not registered as an aggregate function",
            inclusion.component_id
        );
    }
}

/// The default session enables an edition declaring aggregates, which arms the writer's
/// aggregate filter. The declared set must therefore cover every aggregate the default zone
/// maps record, for every dtype, or ordinary writes fail.
#[tokio::test]
async fn default_session_writes_every_default_zone_aggregate() -> VortexResult<()> {
    let session = VortexSession::default();
    // Strings take the bounded min/max branch of the default aggregates, integers the plain
    // min/max branch, so one file exercises both.
    let strings =
        || VarBinViewArray::from_iter_str((0..4096).map(|i| format!("row-{i}"))).into_array();
    let array = StructArray::from_fields(&[
        ("numbers", sequential_integers().into_array()),
        (
            "strings",
            ChunkedArray::from_iter((0..16).map(|_| strings())).into_array(),
        ),
    ])?
    .into_array();

    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buffer, array.to_array_stream())
        .await?;

    // The write succeeding is not enough: a file with no zone maps at all would also succeed.
    // Every zone map names its aggregates in the stats table's field names.
    let file = session.open_options().open_buffer(buffer)?;
    let mut zone_stat_names = Vec::new();
    let mut stack = vec![file.footer().layout().to_layout()];
    while let Some(layout) = stack.pop() {
        let children = layout.children()?;
        if layout.encoding_id().as_str() == "vortex.zoned"
            && let Some(DType::Struct(fields, _)) = children.get(1).map(|zones| zones.dtype())
        {
            zone_stat_names.extend(fields.names().iter().map(|name| name.to_string()));
        }
        stack.extend(children);
    }
    for aggregate in ["vortex.min", "vortex.max", "vortex.bounded_min"] {
        assert!(
            zone_stat_names.iter().any(|name| name.contains(aggregate)),
            "no {aggregate} zone stat in {zone_stat_names:?}"
        );
    }
    Ok(())
}

fn sequential_integers() -> PrimitiveArray {
    PrimitiveArray::from_iter(0..65_536i32)
}

const WRITER_TEST_EDITION: EditionId = EditionId::new("writer-test", 2026, 7, 0);

static WRITER_TEST_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: WRITER_TEST_EDITION,
        immutable: true,
        min_library_version: None,
    },
    added: &[
        EditionMember::array(&"vortex.chunked"),
        EditionMember::array(&"vortex.constant"),
        EditionMember::array(&"vortex.primitive"),
        EditionMember::array(&"vortex.struct"),
        EditionMember::layout(&"vortex.chunked"),
        EditionMember::layout(&"vortex.dict"),
        EditionMember::layout(&"vortex.flat"),
        EditionMember::layout(&"vortex.list"),
        EditionMember::layout(&"vortex.stats"),
        EditionMember::layout(&"vortex.struct"),
        EditionMember::layout(&"vortex.zoned"),
        EditionMember::aggregate(&"vortex.bounded_max"),
        EditionMember::aggregate(&"vortex.bounded_min"),
        EditionMember::aggregate(&"vortex.max"),
        EditionMember::aggregate(&"vortex.min"),
        EditionMember::aggregate(&"vortex.nan_count"),
        EditionMember::aggregate(&"vortex.null_count"),
    ],
};

fn writer_test_session() -> VortexResult<VortexSession> {
    let session = VortexSession::empty()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    session.register_edition(&WRITER_TEST_DECLARATION)?;
    session.enable_edition(WRITER_TEST_EDITION)?;
    Ok(session)
}

#[tokio::test]
async fn disabling_editions_allows_uneditioned_components() -> VortexResult<()> {
    let enforced_session = VortexSession::empty()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&enforced_session);
    let array = ExtensionArray::try_new(
        Date::new(TimeUnit::Days, Nullability::NonNullable).erased(),
        sequential_integers().into_array(),
    )?
    .into_array();

    let mut rejected = ByteBufferMut::empty();
    let error = enforced_session
        .write_options()
        .write(&mut rejected, array.clone().to_array_stream())
        .await
        .err()
        .ok_or_else(|| {
            vortex_err!("writer with no enabled editions accepted an extension dtype")
        })?;
    assert!(
        error
            .to_string()
            .contains("Extension DType vortex.date not permitted"),
        "unexpected error: {error}"
    );

    let uneditioned_session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&uneditioned_session);
    let mut buffer = ByteBufferMut::empty();
    uneditioned_session
        .write_options()
        .disable_editions()
        .write(&mut buffer, array.clone().to_array_stream())
        .await?;
    assert_written_values(&uneditioned_session, buffer, array).await?;
    Ok(())
}

/// Write `array` with `session` and return the buffer, or the error the writer raised.
async fn write_with(session: &VortexSession, array: ArrayRef) -> VortexResult<ByteBufferMut> {
    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buffer, array.to_array_stream())
        .await?;
    Ok(buffer)
}

async fn assert_written_values(
    session: &VortexSession,
    buffer: ByteBufferMut,
    expected: ArrayRef,
) -> VortexResult<()> {
    let actual = session
        .open_options()
        .open_buffer(buffer)?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;
    assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
    Ok(())
}

/// The layout encodings a written file actually contains, depth first.
fn written_layout_ids(session: &VortexSession, buffer: ByteBufferMut) -> VortexResult<Vec<Id>> {
    let file = session.open_options().open_buffer(buffer)?;
    let mut ids = Vec::new();
    let mut stack = vec![file.footer().layout().to_layout()];
    while let Some(layout) = stack.pop() {
        ids.push(layout.encoding_id());
        stack.extend(layout.children()?);
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

/// A session enabling a draft edition that declares every registered array, every aggregate the
/// default writer records, plus the given members of other kinds.
fn session_declaring(members: &[(ComponentKind, Id)]) -> VortexResult<VortexSession> {
    const EDITION: EditionId = EditionId::new("kind-test", 2026, 8, 0);
    const AGGREGATES: [&str; 6] = [
        "vortex.bounded_max",
        "vortex.bounded_min",
        "vortex.max",
        "vortex.min",
        "vortex.nan_count",
        "vortex.null_count",
    ];

    let session = VortexSession::empty()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    let editions = session.editions();
    editions.declare_edition(Edition {
        id: EDITION,
        immutable: true,
        min_library_version: None,
    })?;
    for inclusion in session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>())
        .iter()
        .map(|id| EditionInclusion::array(id, EDITION))
        .chain(
            AGGREGATES
                .into_iter()
                .map(|id| EditionInclusion::new(ComponentKind::Aggregate, id, EDITION)),
        )
        .chain(
            members
                .iter()
                .map(|(kind, id)| EditionInclusion::new(*kind, id, EDITION)),
        )
    {
        editions.declare_inclusion(inclusion)?;
    }
    session.enable_edition(EDITION)?;
    Ok(session)
}

/// Every layout a write emits must be declared; declaring none or only some of them fails the
/// write instead of silently writing a layout an older reader could not decode.
#[tokio::test]
async fn writer_restricts_layouts_to_the_enabled_editions() -> VortexResult<()> {
    let array = sequential_integers().into_array();

    // No layout members declared: the first layout fails the write.
    let session = session_declaring(&[])?;
    let error = write_with(&session, array.clone())
        .await
        .expect_err("a write with no declared layouts must fail");
    assert!(
        error.to_string().contains("not permitted by ctx"),
        "unexpected error: {error}"
    );

    // Discover the layout tree through the fully declared core edition.
    let baseline = VortexSession::default();
    let written = written_layout_ids(&baseline, write_with(&baseline, array.clone()).await?)?;
    assert!(written.len() > 1, "expected a layout tree, got {written:?}");

    // Declaring every layout the write emits permits the file.
    let members: Vec<_> = written
        .iter()
        .map(|id| (ComponentKind::Layout, *id))
        .collect();
    let session = session_declaring(&members)?;
    let buffer = write_with(&session, array.clone()).await?;
    assert_written_values(&session, buffer, array.clone()).await?;

    // Dropping one of them fails the write at the layout it may not emit.
    let session = session_declaring(&members[1..])?;
    let error = write_with(&session, array)
        .await
        .err()
        .ok_or_else(|| vortex_err!("write emitted layout {} outside its edition", written[0]))?;
    assert!(
        error.to_string().contains(&format!(
            "Layout encoding {} not permitted by ctx",
            written[0]
        )),
        "unexpected error: {error}"
    );
    Ok(())
}

/// The writer emits the list layout exactly when the enabled editions include it.
#[tokio::test]
async fn list_layout_follows_enabled_editions() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    use crate::VortexSessionDefault;

    let list = ListArray::try_new(
        buffer![1i32, 2, 3, 4, 5].into_array(),
        buffer![0u32, 2, 5].into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?;
    let array = StructArray::from_fields(&[("l", list.into_array())])?.into_array();
    let list_layout = Id::from("vortex.list");

    // The default core edition does not include the list layout, so lists stay flat.
    let session = VortexSession::default();
    let written = written_layout_ids(&session, write_with(&session, array.clone()).await?)?;
    assert!(!written.contains(&list_layout), "{written:?}");

    let session = VortexSession::default();
    session.editions().declare_inclusion(EditionInclusion::new(
        ComponentKind::Layout,
        "vortex.list",
        DEFAULT_PREVIEW_EDITION,
    ))?;
    session.enable_edition(DEFAULT_PREVIEW_EDITION)?;

    // Permitted: the default writer emits it.
    let written = written_layout_ids(&session, write_with(&session, array).await?)?;
    assert!(written.contains(&list_layout), "{written:?}");
    Ok(())
}

fn forbidden_sequence(len: usize) -> VortexResult<ArrayRef> {
    Ok(Sequence::try_new_typed(0i32, 1i32, Nullability::NonNullable, len)?.into_array())
}

fn forbidden_sequence_compressor(
    chunk: &ArrayRef,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if matches!(
        chunk.dtype(),
        DType::Primitive(PType::I32, Nullability::NonNullable)
    ) {
        forbidden_sequence(chunk.len())
    } else {
        Ok(chunk.clone())
    }
}

/// The writer configures BtrBlocks from the enabled edition, so an effective but unavailable
/// encoding is skipped rather than produced and rejected during serialization.
#[tokio::test]
async fn btrblocks_respects_enabled_array_encodings() -> VortexResult<()> {
    let session = writer_test_session()?;
    let array = sequential_integers().into_array();
    let buffer = write_with(&session, array.clone()).await?;
    assert_written_values(&session, buffer, array).await?;
    Ok(())
}

/// An explicitly supplied strategy is not reconfigured by the writer. Its unsupported output is
/// still caught by the serialization context.
#[tokio::test]
async fn explicit_btrblocks_strategy_is_not_reconfigured() -> VortexResult<()> {
    let session = writer_test_session()?;
    let strategy = WriteStrategyBuilder::from_session(&session)
        .with_btrblocks_builder(BtrBlocksCompressorBuilder::from_session(&session).unrestricted())
        .build();
    let mut buffer = ByteBufferMut::empty();

    let error = session
        .write_options()
        .with_strategy(strategy)
        .write(
            &mut buffer,
            sequential_integers().into_array().to_array_stream(),
        )
        .await
        .err()
        .ok_or_else(|| vortex_err!("explicit BtrBlocks strategy was unexpectedly reconfigured"))?;
    assert!(
        error
            .to_string()
            .contains("Serialized array ID vortex.sequence not permitted by ctx"),
        "unexpected error: {error}"
    );

    Ok(())
}

/// Compressors operate on the current in-memory array model and do not interpret edition wire
/// IDs. The serialization context is the final compatibility boundary and rejects a compressor
/// result whose serialized ID is not enabled.
#[tokio::test]
async fn serialization_context_rejects_unsupported_compressor_output() -> VortexResult<()> {
    let session = writer_test_session()?;
    let strategy = WriteStrategyBuilder::from_session(&session)
        .with_compressor(forbidden_sequence_compressor)
        .build();
    let mut buffer = ByteBufferMut::empty();

    let error = session
        .write_options()
        .with_strategy(strategy)
        .write(
            &mut buffer,
            sequential_integers().into_array().to_array_stream(),
        )
        .await
        .err()
        .ok_or_else(|| vortex_err!("Sequence unexpectedly had a permitted wire variant"))?;
    assert!(
        error
            .to_string()
            .contains("Serialized array ID vortex.sequence not permitted by ctx"),
        "unexpected error: {error}"
    );

    Ok(())
}

/// The same compressor output is writable when its wire ID is enabled, without configuring the
/// compressor itself from the edition.
#[tokio::test]
async fn serialization_context_accepts_supported_compressor_output() -> VortexResult<()> {
    let session = VortexSession::default();
    let strategy = WriteStrategyBuilder::from_session(&session)
        .with_compressor(forbidden_sequence_compressor)
        .build();
    let mut buffer = ByteBufferMut::empty();

    session
        .write_options()
        .with_strategy(strategy)
        .write(
            &mut buffer,
            sequential_integers().into_array().to_array_stream(),
        )
        .await?;

    assert_written_values(&session, buffer, sequential_integers().into_array()).await?;
    Ok(())
}

/// A writer that enables `core2025.10.0` writes files for readers before `vortex.zoned`. The
/// zone maps use the legacy `vortex.stats` layout, and the file has only `core2025.10.0` layouts.
#[tokio::test]
async fn core_2025_10_0_writes_legacy_zone_maps() -> VortexResult<()> {
    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    session.enable_edition(CORE_2025_10_0)?;

    let len = 65_536usize;
    let strings = VarBinViewArray::from_iter_nullable_str(
        (0..len).map(|i| (i % 7 != 0).then(|| format!("row-{i:06}"))),
    )
    .into_array();
    let lists = ListArray::try_new(
        VarBinViewArray::from_iter_str((0..len * 2).map(|i| format!("item-{i}"))).into_array(),
        PrimitiveArray::from_iter((0..=len as i64).map(|i| i * 2)).into_array(),
        Validity::NonNullable,
        &mut session.create_execution_ctx(),
    )?
    .into_array();
    let nested = StructArray::from_fields(&[
        ("a", PrimitiveArray::from_iter(0..len as i64).into_array()),
        (
            "b",
            PrimitiveArray::from_option_iter((0..len).map(|i| (i % 3 != 0).then_some(i as f64)))
                .into_array(),
        ),
    ])?
    .into_array();
    let array = StructArray::from_fields(&[
        ("int", sequential_integers().into_array()),
        (
            "bool",
            BoolArray::from_iter((0..len).map(|i| i % 2 == 0)).into_array(),
        ),
        ("string", strings),
        ("list", lists),
        ("nested", nested),
    ])?
    .into_array();

    let buffer = write_with(&session, array.clone()).await?;

    let written = written_layout_ids(&session, buffer.clone())?;
    let permitted: Vec<Id> = session
        .editions()
        .components_in(&CORE_2025_10_0, ComponentKind::Layout)
        .into_iter()
        .map(|inclusion| inclusion.component_id)
        .collect();
    for id in &written {
        assert!(
            permitted.contains(id),
            "layout {id} is not in core2025.10.0"
        );
    }
    assert!(
        written.iter().any(|id| id.as_str() == "vortex.stats"),
        "no legacy zone map in {written:?}"
    );

    assert_written_values(&session, buffer.clone(), array).await?;

    // The legacy zone maps prune: the filter keeps only the last zone.
    let file = session.open_options().open_buffer(buffer)?;
    let filter = gt_eq(col("int"), lit(65_000i32)).bind(file.dtype())?;
    let filtered = file
        .scan()?
        .with_filter(filter)
        .into_array_stream()?
        .read_all()
        .await?;
    assert_eq!(filtered.len(), 536);

    // The editions permit no aggregates, so the file statistics use the legacy format only. All
    // readers read them, by the paths of the top-level fields.
    let statistics = file
        .footer()
        .statistics()
        .vortex_expect("file has statistics");
    let (int_stats, int_dtype) = statistics
        .get_by_path(&FieldPath::from_name(FieldName::from("int")))
        .vortex_expect("int column has statistics");
    let int_stats = int_stats.to_stats_set();
    assert!(
        int_stats
            .get_as::<i32>(Stat::Min, int_dtype)
            .as_exact()
            .is_some()
    );
    assert!(
        int_stats
            .get_as::<i32>(Stat::Max, int_dtype)
            .as_exact()
            .is_some()
    );

    // A string column has the bounded min and max, which fill the legacy min and max slots.
    let (string_stats, string_dtype) = statistics
        .get_by_path(&FieldPath::from_name(FieldName::from("string")))
        .vortex_expect("string column has statistics");
    let string_stats = string_stats.to_stats_set();
    assert!(!string_stats.get(Stat::Min).is_absent());
    assert!(!string_stats.get(Stat::Max).is_absent());
    assert_eq!(
        string_stats
            .get_as::<u64>(
                Stat::NullCount,
                &Stat::NullCount
                    .dtype(string_dtype)
                    .vortex_expect("null count has a dtype"),
            )
            .as_exact(),
        Some(9_363)
    );

    assert_eq!(statistics.paths().len(), 5);
    Ok(())
}

/// End-to-end `.vortex` file round-trip for `vortex.pivco` under `UNSTABLE`:
/// - `DEFAULT_CORE_EDITION` rejects `vortex.pivco` serialization.
/// - Enabling `UNSTABLE` permits writing `PivcoArray`s (`!has_extra_bits`, `has_extra_bits` with
///   `ebits > 57` and nulls, and `ZigZag(Pivco)`), and reading them back via full scan, `==` and
///   `<` filter pushdown, `with_row_indices` (`take`), and `with_row_range` (`slice`).
#[tokio::test]
async fn unstable_pivco_vortex_file_roundtrip_and_queries() -> VortexResult<()> {
    let len = 10_000usize;
    let mut rng = StdRng::seed_from_u64(2026_0505);

    // Column 1: skewed non-negative `u64` (`!has_extra_bits`, `OP-PHA` for `<` & `==` pushdown).
    let gt_prim = PrimitiveArray::from_iter((0..len).map(|_| {
        let d = rng.random_range(0..1000u32);
        if d < 850 {
            0u64
        } else if d < 970 {
            1u64
        } else if d < 995 {
            2u64
        } else {
            5u64
        }
    }));

    // Column 2: nullable `u64` with `has_extra_bits` (including `ebits > 57`).
    let wide_prim = PrimitiveArray::from_option_iter((0..len).map(|i| {
        if i % 13 == 0 {
            None
        } else if i % 5 == 0 {
            Some(rng.random_range(0..32u64))
        } else if i % 5 == 1 {
            Some(64 + rng.random_range(0..500u64))
        } else {
            Some((1u64 << 60) + (i as u64) * 1_009)
        }
    }));

    // Column 3: signed Laplacian `i32` (`ZigZag(Pivco)`).
    let signed_prim = PrimitiveArray::from_option_iter((0..len).map(|i| {
        if i % 17 == 0 {
            None
        } else {
            let mag = i32::try_from(i % 41).vortex_expect("fits i32");
            Some(if i % 2 == 0 { -mag } else { mag })
        }
    }));

    // Column 4: shifted `u64` (`1_700_000_000_000 + skewed_delta`) compressed as `FoR(Pivco)`.
    let shifted_prim = PrimitiveArray::from_iter((0..len).map(|_| {
        let d = rng.random_range(0..1000u32);
        let delta = if d < 900 {
            0u64
        } else if d < 980 {
            1u64
        } else {
            rng.random_range(2..1_024u64)
        };
        1_700_000_000_000u64 + delta
    }));

    let expected_struct = StructArray::from_fields(&[
        ("gt", gt_prim.clone().into_array()),
        ("wide", wide_prim.clone().into_array()),
        ("signed", signed_prim.clone().into_array()),
        ("shifted", shifted_prim.clone().into_array()),
    ])?
    .into_array();

    let pivco_compressor = |chunk: &ArrayRef, ctx: &mut ExecutionCtx| -> VortexResult<ArrayRef> {
        match chunk.dtype() {
            DType::Primitive(PType::U64, Nullability::NonNullable) => {
                let prim = chunk.clone().execute::<PrimitiveArray>(ctx)?;
                let first = prim.as_slice::<u64>().first().copied().unwrap_or(0);
                if first >= 1_000_000_000_000 {
                    let for_arr = FoR::encode(prim, ctx)?;
                    let biased = for_arr.encoded().clone().execute::<PrimitiveArray>(ctx)?;
                    let child = Pivco::encode_with_options(
                        &biased,
                        PivcoOptions::op_pha().with_block_shift(11),
                        ctx,
                    )?
                    .into_array();
                    let FoRReferences::Global(reference) = for_arr.references() else {
                        unreachable!()
                    };
                    return Ok(FoR::try_new(child, reference.clone())?.into_array());
                }
                Ok(Pivco::encode_with_options(
                    &prim,
                    PivcoOptions::op_pha().with_block_shift(11),
                    ctx,
                )?
                .into_array())
            }
            DType::Primitive(PType::U64, Nullability::Nullable) => {
                let prim = chunk.clone().execute::<PrimitiveArray>(ctx)?;
                Ok(Pivco::encode_with_options(
                    &prim,
                    PivcoOptions::op_pha().with_block_shift(11),
                    ctx,
                )?
                .into_array())
            }
            DType::Primitive(PType::I32, _) => {
                let prim = chunk.clone().execute::<PrimitiveArray>(ctx)?;
                let zz = zigzag_encode(prim.as_view())?;
                let zz_prim = zz.encoded().clone().execute::<PrimitiveArray>(ctx)?;
                let zz_pivco = Pivco::encode_with_options(
                    &zz_prim,
                    PivcoOptions::pha().with_block_shift(11),
                    ctx,
                )?
                .into_array();
                Ok(ZigZag::try_new(zz_pivco)?.into_array())
            }
            _ => Ok(chunk.clone()),
        }
    };

    // 1. Default `core` session rejects `vortex.pivco` produced by the compressor.
    let core_session = VortexSession::default();
    let core_strategy = WriteStrategyBuilder::from_session(&core_session)
        .with_compressor(pivco_compressor)
        .build();
    let mut rejected = ByteBufferMut::empty();
    let err = core_session
        .write_options()
        .with_strategy(core_strategy)
        .write(&mut rejected, expected_struct.clone().to_array_stream())
        .await
        .err()
        .ok_or_else(|| vortex_err!("default core edition unexpectedly accepted vortex.pivco"))?;
    assert!(
        err.to_string()
            .contains("Serialized array ID vortex.pivco not permitted by ctx"),
        "unexpected error: {err}"
    );

    // 2. Enabling `UNSTABLE` permits writing `vortex.pivco` directly to a `.vortex` file.
    let session = VortexSession::default();
    session.enable_edition(UNSTABLE)?;
    let pivco_strategy = WriteStrategyBuilder::from_session(&session)
        .with_compressor(pivco_compressor)
        .build();
    let mut file_buf = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(pivco_strategy)
        .write(&mut file_buf, expected_struct.clone().to_array_stream())
        .await?;
    let file_bytes = file_buf.freeze();

    // 3. Full scan matches canonical values.
    let file = session.open_options().open_buffer(file_bytes)?;
    let full = file.scan()?.into_array_stream()?.read_all().await?;
    assert_arrays_eq!(full, expected_struct, &mut session.create_execution_ctx());

    // 4. `==` filter pushdown (`gt == 1u64`).
    let eq_filter = eq(col("gt"), lit(1u64)).bind(file.dtype())?;
    let eq_actual = file
        .scan()?
        .with_filter(eq_filter)
        .into_array_stream()?
        .read_all()
        .await?;
    let mut ctx = session.create_execution_ctx();
    let eq_mask_arr = gt_prim.clone().into_array().compare(
        ConstantArray::new(Scalar::from(1u64), len).into_array(),
        CompareOperator::Eq,
    )?;
    let eq_mask = eq_mask_arr.fill_null(false)?.execute::<Mask>(&mut ctx)?;
    let eq_expected = expected_struct.filter(eq_mask)?;
    assert_arrays_eq!(eq_actual, eq_expected, &mut ctx);

    // 5. `<` filter pushdown (`wide < 64u64` and `gt < 2u64`).
    let lt_filter = lt(col("gt"), lit(2u64)).bind(file.dtype())?;
    let lt_actual = file
        .scan()?
        .with_filter(lt_filter)
        .into_array_stream()?
        .read_all()
        .await?;
    let lt_mask_arr = gt_prim.clone().into_array().compare(
        ConstantArray::new(Scalar::from(2u64), len).into_array(),
        CompareOperator::Lt,
    )?;
    let lt_mask = lt_mask_arr.fill_null(false)?.execute::<Mask>(&mut ctx)?;
    let lt_expected = expected_struct.filter(lt_mask)?;
    assert_arrays_eq!(lt_actual, lt_expected, &mut ctx);

    let lt_wide_filter = lt(col("wide"), lit(64u64)).bind(file.dtype())?;
    let lt_wide_actual = file
        .scan()?
        .with_filter(lt_wide_filter)
        .into_array_stream()?
        .read_all()
        .await?;
    let lt_wide_mask_arr = wide_prim.clone().into_array().compare(
        ConstantArray::new(Scalar::from(64u64), len).into_array(),
        CompareOperator::Lt,
    )?;
    let lt_wide_mask = lt_wide_mask_arr
        .fill_null(false)?
        .execute::<Mask>(&mut ctx)?;
    let lt_wide_expected = expected_struct.filter(lt_wide_mask)?;
    assert_arrays_eq!(lt_wide_actual, lt_wide_expected, &mut ctx);

    // 6. `with_row_indices` (`take` across multiple blocks).
    let take_idx: Vec<u64> = (0..len as u64).step_by(173).collect();
    let take_actual = file
        .scan()?
        .with_row_indices(StrictSortedBuffer::try_new(Buffer::from_iter(
            take_idx.clone(),
        ))?)
        .into_array_stream()?
        .read_all()
        .await?;
    let take_expected = expected_struct.take(PrimitiveArray::from_iter(take_idx).into_array())?;
    assert_arrays_eq!(take_actual, take_expected, &mut ctx);

    // 7. `with_row_range` (`slice` crossing block boundaries).
    let slice_actual = file
        .scan()?
        .with_row_range(137..8_901)
        .into_array_stream()?
        .read_all()
        .await?;
    let slice_expected = expected_struct.slice(137..8_901)?;
    assert_arrays_eq!(slice_actual, slice_expected, &mut ctx);

    // 8. Default `BtrBlocks` compressor under `UNSTABLE` round-trips cleanly through `.vortex` file.
    let auto_buf = write_with(&session, expected_struct.clone()).await?;
    assert_written_values(&session, auto_buf, expected_struct).await?;

    Ok(())
}
