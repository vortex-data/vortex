// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::field_path;
use vortex_array::session::ArraySessionExt;
use vortex_array::stream::ArrayStreamExt;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
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
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::session::LayoutSession;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;
use vortex_session::registry::Id;
use vortex_utils::aliases::hash_set::HashSet;

use super::CORE_2025_05_0;
use super::CORE_2026_07_0;
use super::CORE_2026_08;
use super::DEFAULT_CORE_EDITION;
use super::DEFAULT_UNSTABLE_EDITION;
use super::EDITION_DECLARATIONS;
use super::UNSTABLE_2026_06_0;

fn session() -> Result<EditionSession, EditionError> {
    let session = EditionSession::empty();
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

/// The full encoding set of the newest frozen `core` edition. This set is frozen: the only
/// way it may change is by declaring a *new* edition, so a failure here means a frozen
/// declaration was edited.
#[test]
fn core_2026_07_encoding_set_is_pinned() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    let encodings = session.components_in(&CORE_2026_07_0, ComponentKind::Array);
    let ids: Vec<&str> = encodings
        .iter()
        .map(|inclusion| inclusion.component_id.as_str())
        .collect();
    assert_eq!(
        ids,
        [
            "fastlanes.bitpacked",
            "fastlanes.for",
            "fastlanes.rle",
            "vortex.alp",
            "vortex.alprd",
            "vortex.bool",
            "vortex.bytebool",
            "vortex.chunked",
            "vortex.constant",
            "vortex.datetimeparts",
            "vortex.decimal",
            "vortex.decimal_byte_parts",
            "vortex.dict",
            "vortex.ext",
            "vortex.fixed_size_list",
            "vortex.fsst",
            "vortex.list",
            "vortex.listview",
            "vortex.masked",
            "vortex.null",
            "vortex.pco",
            "vortex.primitive",
            "vortex.runend",
            "vortex.sequence",
            "vortex.sparse",
            "vortex.struct",
            "vortex.varbin",
            "vortex.varbinview",
            "vortex.variant",
            "vortex.zigzag",
            "vortex.zstd",
        ]
    );
}

#[test]
fn encodings_in_editions_unions_families() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    let core_only: Vec<_> = session
        .components_in(&CORE_2026_07_0, ComponentKind::Array)
        .into_iter()
        .map(|inclusion| inclusion.component_id)
        .collect();
    let mut both = core_only.clone();
    both.extend(
        session
            .components_in(&UNSTABLE_2026_06_0, ComponentKind::Array)
            .into_iter()
            .map(|inclusion| inclusion.component_id),
    );
    both.sort_unstable();
    both.dedup();

    assert!(both.len() > core_only.len());
    assert!(both.iter().any(|id| id.as_str() == "fastlanes.delta"));
    assert!(both.iter().any(|id| id.as_str() == "vortex.onpair"));
    assert!(core_only.iter().all(|id| both.contains(id)));
}

#[test]
fn earlier_editions_are_subsets() {
    let session = session().unwrap_or_else(|e| panic!("registering editions: {e}"));
    let first = session.components_in(&CORE_2025_05_0, ComponentKind::Array);
    let latest = session.components_in(&CORE_2026_08, ComponentKind::Array);
    assert!(first.iter().all(|inclusion| {
        latest
            .iter()
            .any(|latest| latest.component_id == inclusion.component_id)
    }));
    assert!(first.len() < latest.len());
}

#[test]
fn default_session_enables_the_write_editions() {
    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    let enabled = session.enabled_editions().editions();
    assert!(enabled.contains(&DEFAULT_CORE_EDITION));

    #[cfg(feature = "unstable_encodings")]
    assert!(enabled.contains(&DEFAULT_UNSTABLE_EDITION));
    #[cfg(not(feature = "unstable_encodings"))]
    assert!(!enabled.contains(&DEFAULT_UNSTABLE_EDITION));
}

#[test]
fn core_edition_ids_are_registered_array_encodings() {
    use vortex_array::session::ArraySessionExt;

    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    let registry = session.arrays().registry().clone();
    for inclusion in session
        .editions()
        .components_in(&CORE_2026_08, ComponentKind::Array)
    {
        assert!(
            registry.contains_key(&inclusion.component_id),
            "{} is declared in core but not registered as an array encoding",
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
        .components_in(&CORE_2026_08, ComponentKind::Aggregate);
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
    // min/max/sum branch, so one file exercises both.
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

fn baseline_core_session() -> VortexResult<VortexSession> {
    use crate::VortexSessionDefault;

    let session = VortexSession::default();
    session
        .enable_edition(CORE_2025_05_0)
        .map_err(|error| vortex_err!("{error}"))?;
    Ok(session)
}

fn sequential_integers() -> PrimitiveArray {
    PrimitiveArray::from_iter(0..65_536i32)
}

const WRITER_TEST_EDITION: EditionId = EditionId::new("writer-test", 2026, 7, 0);

static WRITER_TEST_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: WRITER_TEST_EDITION,
        min_vortex_version: None,
    },
    added: &[
        EditionMember::array(&"vortex.chunked"),
        EditionMember::array(&"vortex.constant"),
        EditionMember::array(&"vortex.primitive"),
        EditionMember::array(&"vortex.struct"),
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

/// A session enabling a draft edition that declares every registered array, plus the given
/// members of other kinds.
fn session_declaring(members: &[(ComponentKind, Id)]) -> VortexResult<VortexSession> {
    const EDITION: EditionId = EditionId::new("kind-test", 2026, 8, 0);

    let session = array_session()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: EDITION,
            min_vortex_version: None,
        })
        .map_err(|error| vortex_err!("{error}"))?;
    for inclusion in session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>())
        .iter()
        .map(|id| EditionInclusion::array(id, EDITION))
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

/// A kind with no declared members is unrestricted, and declaring the layouts a write emits
/// permits it; declaring only some of them fails the write instead of silently writing a
/// layout an older reader could not decode.
#[tokio::test]
async fn writer_restricts_layouts_to_the_enabled_editions() -> VortexResult<()> {
    let array = sequential_integers().into_array();

    // No layout members declared: the layout filter stays disarmed.
    let session = session_declaring(&[])?;
    let written = written_layout_ids(&session, write_with(&session, array.clone()).await?)?;
    assert!(written.len() > 1, "expected a layout tree, got {written:?}");

    // Declaring every layout the write emits permits exactly the same file.
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

fn custom_compressing_flat_strategy() -> Arc<dyn LayoutStrategy> {
    Arc::new(CompressingStrategy::new(
        FlatLayoutStrategy::default(),
        forbidden_sequence_compressor,
    ))
}

async fn assert_round_trip_encodings_are_enabled(
    session: &VortexSession,
    strategy: Option<Arc<dyn LayoutStrategy>>,
    array: ArrayRef,
) -> VortexResult<()> {
    let mut buffer = ByteBufferMut::empty();
    let write_options = match strategy {
        Some(strategy) => session.write_options().with_strategy(strategy),
        None => session.write_options(),
    };
    if let Err(error) = write_options
        .write(&mut buffer, array.to_array_stream())
        .await
    {
        let message = error.to_string();
        if message.contains("not permitted by ctx")
            || message.contains("normalize forbids encoding")
        {
            return Ok(());
        }
        return Err(error);
    }

    let round_tripped = session
        .open_options()
        .open_buffer(buffer)?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;
    let actual: HashSet<_> = round_tripped
        .depth_first_traversal()
        .map(|array| array.encoding_id())
        .collect();
    let allowed: HashSet<_> = session
        .enabled_component_ids(ComponentKind::Array)
        .into_iter()
        .collect();
    let mut forbidden: Vec<_> = actual.difference(&allowed).map(|id| id.as_str()).collect();
    forbidden.sort_unstable();
    if !forbidden.is_empty() {
        return Err(vortex_err!(
            "round-tripped array contains encodings outside {WRITER_TEST_EDITION}: {forbidden:?}"
        ));
    }

    Ok(())
}

#[tokio::test]
async fn default_strategy_round_trip_uses_only_enabled_encodings() -> VortexResult<()> {
    let session = writer_test_session()?;
    assert_round_trip_encodings_are_enabled(&session, None, sequential_integers().into_array())
        .await
}

#[tokio::test]
async fn replacement_default_builder_round_trip_uses_only_enabled_encodings() -> VortexResult<()> {
    let session = writer_test_session()?;
    assert_round_trip_encodings_are_enabled(
        &session,
        Some(WriteStrategyBuilder::default().build()),
        sequential_integers().into_array(),
    )
    .await
}

#[tokio::test]
async fn replacement_btrblocks_builder_round_trip_uses_only_enabled_encodings() -> VortexResult<()>
{
    let session = writer_test_session()?;
    let strategy = WriteStrategyBuilder::default()
        .with_btrblocks_builder(BtrBlocksCompressorBuilder::default())
        .build();
    assert_round_trip_encodings_are_enabled(
        &session,
        Some(strategy),
        sequential_integers().into_array(),
    )
    .await
}

#[tokio::test]
async fn opaque_compressor_round_trip_uses_only_enabled_encodings() -> VortexResult<()> {
    let session = writer_test_session()?;
    let strategy = WriteStrategyBuilder::default()
        .with_compressor(forbidden_sequence_compressor)
        .build();
    assert_round_trip_encodings_are_enabled(
        &session,
        Some(strategy),
        sequential_integers().into_array(),
    )
    .await
}

#[tokio::test]
async fn custom_flat_strategy_round_trip_uses_only_enabled_encodings() -> VortexResult<()> {
    let session = writer_test_session()?;
    let strategy = WriteStrategyBuilder::default()
        .with_flat_strategy(Arc::new(FlatLayoutStrategy::default()))
        .build();
    assert_round_trip_encodings_are_enabled(
        &session,
        Some(strategy),
        sequential_integers().into_array(),
    )
    .await
}

#[tokio::test]
async fn custom_field_writer_round_trip_uses_only_enabled_encodings() -> VortexResult<()> {
    let session = writer_test_session()?;
    let strategy = WriteStrategyBuilder::default()
        .with_field_writer(field_path!(values), custom_compressing_flat_strategy())
        .build();
    let array =
        StructArray::from_fields(&[("values", sequential_integers().into_array())])?.into_array();
    assert_round_trip_encodings_are_enabled(&session, Some(strategy), array).await
}

#[tokio::test]
async fn replacement_strategy_round_trip_uses_only_enabled_encodings() -> VortexResult<()> {
    let session = writer_test_session()?;
    assert_round_trip_encodings_are_enabled(
        &session,
        Some(custom_compressing_flat_strategy()),
        sequential_integers().into_array(),
    )
    .await
}

#[tokio::test]
async fn replacement_flat_strategy_round_trip_uses_only_enabled_encodings() -> VortexResult<()> {
    let session = writer_test_session()?;
    assert_round_trip_encodings_are_enabled(
        &session,
        Some(Arc::new(FlatLayoutStrategy::default())),
        forbidden_sequence(65_536)?,
    )
    .await
}

#[tokio::test]
async fn probe_compressor_round_trip_uses_only_enabled_encodings() -> VortexResult<()> {
    let session = writer_test_session()?;
    let strategy = WriteStrategyBuilder::default()
        .with_probe_compressor(forbidden_sequence_compressor)
        .build();
    assert_round_trip_encodings_are_enabled(
        &session,
        Some(strategy),
        sequential_integers().into_array(),
    )
    .await
}

#[tokio::test]
async fn default_writer_filters_compressor_to_enabled_editions() -> VortexResult<()> {
    let session = baseline_core_session()?;
    let mut buffer = ByteBufferMut::empty();

    session
        .write_options()
        .write(
            &mut buffer,
            sequential_integers().into_array().to_array_stream(),
        )
        .await?;

    Ok(())
}

#[tokio::test]
async fn configured_btrblocks_builder_uses_enabled_editions_in_either_order() -> VortexResult<()> {
    let session = baseline_core_session()?;
    let allowed: HashSet<_> = session
        .enabled_component_ids(ComponentKind::Array)
        .into_iter()
        .collect();
    let strategies = [
        WriteStrategyBuilder::default()
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default())
            .with_allow_encodings(allowed.clone())
            .build(),
        WriteStrategyBuilder::default()
            .with_allow_encodings(allowed)
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default())
            .build(),
    ];

    for strategy in strategies {
        let mut buffer = ByteBufferMut::empty();
        session
            .write_options()
            .with_strategy(strategy)
            .write(
                &mut buffer,
                sequential_integers().into_array().to_array_stream(),
            )
            .await?;
    }

    Ok(())
}

#[tokio::test]
async fn opaque_compressor_cannot_write_outside_enabled_editions() -> VortexResult<()> {
    let session = baseline_core_session()?;
    let allowed = session
        .enabled_component_ids(ComponentKind::Array)
        .into_iter()
        .collect();
    let strategy = WriteStrategyBuilder::default()
        .with_compressor(BtrBlocksCompressorBuilder::default().build())
        .with_allow_encodings(allowed)
        .build();
    let mut buffer = ByteBufferMut::empty();

    let result = session
        .write_options()
        .with_strategy(strategy)
        .write(
            &mut buffer,
            sequential_integers().into_array().to_array_stream(),
        )
        .await;
    let error = match result {
        Ok(_) => {
            return Err(vortex_err!(
                "the unrestricted opaque compressor wrote an encoding outside core@2025.05"
            ));
        }
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("normalize forbids encoding (vortex.sequence)"),
        "unexpected error: {message}"
    );

    Ok(())
}
