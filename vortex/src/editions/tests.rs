// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::ByteBufferMut;
use vortex_edition::ComponentKind;
use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionError;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionMember;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::test_harness::validate_edition;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::session::RuntimeSession;
use vortex_layout::session::LayoutSession;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;
use vortex_session::registry::Id;

use super::CORE_2025_05_0;
use super::CORE_2026_08_0;
use super::CORE_2026_08_1;
use super::CORE_2026_08_2;
use super::CORE_2026_08_3;
use super::DEFAULT_CORE_EDITION;
use super::DEFAULT_PREVIEW_EDITION;
use super::EDITION_DECLARATIONS;
use super::PREVIEW_2026_08_0;

fn session() -> Result<EditionSession, EditionError> {
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
fn every_declared_edition_validates() -> Result<(), EditionError> {
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
fn preview_starts_empty() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    for kind in [
        ComponentKind::Array,
        ComponentKind::Layout,
        ComponentKind::DType,
        ComponentKind::Aggregate,
    ] {
        assert!(session.components_in(&PREVIEW_2026_08_0, kind).is_empty());
    }
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
    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    let enabled = session.enabled_editions().editions();
    assert!(enabled.contains(&DEFAULT_CORE_EDITION));
    assert!(
        session
            .enabled_component_ids(ComponentKind::Array)
            .contains(&Id::from("vortex.pco"))
    );

    #[cfg(feature = "unstable_encodings")]
    assert!(enabled.contains(&DEFAULT_PREVIEW_EDITION));
    #[cfg(not(feature = "unstable_encodings"))]
    assert!(!enabled.contains(&DEFAULT_PREVIEW_EDITION));
}

#[test]
fn core_edition_ids_are_registered_array_encodings() {
    use vortex_array::session::ArraySessionExt;

    use crate::VortexSessionDefault;

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

    use crate::VortexSessionDefault;

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

    use crate::VortexSessionDefault;

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
    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    // Strings take the bounded min/max branch of the default aggregates, integers the plain
    // min/max branch, so one file exercises both.
    let strings = || {
        vortex_array::arrays::VarBinViewArray::from_iter_str((0..4096).map(|i| format!("row-{i}")))
            .into_array()
    };
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
    let session = array_session()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    session
        .register_edition(&WRITER_TEST_DECLARATION)
        .map_err(|error| vortex_err!("{error}"))?;
    session
        .enable_edition(WRITER_TEST_EDITION)
        .map_err(|error| vortex_err!("{error}"))?;
    Ok(session)
}

#[tokio::test]
async fn disabling_editions_allows_uneditioned_components() -> VortexResult<()> {
    let enforced_session = array_session()
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

    let uneditioned_session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&uneditioned_session);
    let mut buffer = ByteBufferMut::empty();
    uneditioned_session
        .write_options()
        .disable_editions()
        .write(&mut buffer, array.to_array_stream())
        .await?;
    assert!(!buffer.is_empty());
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

    let session = array_session()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: EDITION,
            min_library_version: None,
        })
        .map_err(|error| vortex_err!("{error}"))?;
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
        editions
            .declare_inclusion(inclusion)
            .map_err(|error| vortex_err!("{error}"))?;
    }
    session
        .enable_edition(EDITION)
        .map_err(|error| vortex_err!("{error}"))?;
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
    use crate::VortexSessionDefault;

    let baseline = VortexSession::default();
    let written = written_layout_ids(&baseline, write_with(&baseline, array.clone()).await?)?;
    assert!(written.len() > 1, "expected a layout tree, got {written:?}");

    // Declaring every layout the write emits permits the file.
    let members: Vec<_> = written
        .iter()
        .map(|id| (ComponentKind::Layout, *id))
        .collect();
    let session = session_declaring(&members)?;
    write_with(&session, array.clone()).await?;

    // Dropping one of them fails the write at the layout it may not emit.
    let session = session_declaring(&members[1..])?;
    let error = write_with(&session, array)
        .await
        .err()
        .ok_or_else(|| vortex_err!("write emitted layout {} outside its edition", written[0]))?;
    assert!(
        error.to_string().contains("not permitted by ctx"),
        "unexpected error: {error}"
    );
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
    write_with(&session, sequential_integers().into_array()).await?;
    Ok(())
}

/// An explicitly supplied strategy is not reconfigured by the writer. Its unsupported output is
/// still caught by the serialization context.
#[tokio::test]
async fn explicit_btrblocks_strategy_is_not_reconfigured() -> VortexResult<()> {
    let session = writer_test_session()?;
    let strategy = WriteStrategyBuilder::default().build();
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
    let strategy = WriteStrategyBuilder::default()
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
    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    let strategy = WriteStrategyBuilder::default()
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

    Ok(())
}
