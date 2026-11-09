// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Primitive;
use vortex_array::assert_arrays_eq;
use vortex_array::session::ArraySessionExt;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_edition::ComponentKind;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_session::VortexSession;
use vortex_session::registry::Id;

use super::UNSTABLE;
use crate::VortexSessionDefault;

/// An external wire ID backed by the primitive implementation, registered only by this test.
#[derive(Debug)]
struct ExternalPlugin(Id);

impl ArrayPlugin for ExternalPlugin {
    fn id(&self) -> ArrayId {
        ArrayPlugin::id(&Primitive)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![self.0]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        vec![self.0]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        Ok(
            ArrayPlugin::serialize(&Primitive, array, session)?.map(|mut serialized| {
                serialized.serialized_id = self.0;
                serialized
            }),
        )
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        ArrayPlugin::deserialize(
            &Primitive,
            ArrayDeserialization {
                serialized_id: ArrayPlugin::id(&Primitive),
                ..parts
            },
            session,
        )
    }
}

#[tokio::test]
async fn external_format_requires_opt_in_and_round_trips_with_its_original_id() -> VortexResult<()>
{
    let session = VortexSession::default();
    let id = Id::from("external.experimental_primitive");
    session.arrays().register(ExternalPlugin(id));
    session.editions().declare_inclusion(EditionInclusion::new(
        ComponentKind::Array,
        &id,
        UNSTABLE,
    ))?;
    let array = buffer![1i32, 2, 3].into_array();
    let strategy = WriteStrategyBuilder::from_session(&session)
        .with_compressor(|array: &ArrayRef, _: &mut ExecutionCtx| Ok(array.clone()))
        .build();
    let mut rejected = ByteBufferMut::empty();
    let error = session
        .write_options()
        .with_strategy(Arc::clone(&strategy))
        .write(&mut rejected, array.clone().to_array_stream())
        .await
        .err()
        .ok_or_else(|| vortex_err!("experimental output was accepted without an opt-in"))?;
    assert!(
        error
            .to_string()
            .contains("Serialized array ID external.experimental_primitive not permitted"),
        "{error}"
    );

    session.enable_edition(UNSTABLE)?;
    let mut bytes = ByteBufferMut::empty();
    session
        .write_options()
        .with_strategy(strategy)
        .write(&mut bytes, array.clone().to_array_stream())
        .await?;

    // Reader permissions are independent: only the matching plugin is needed to read this file.
    let reader = VortexSession::default();
    reader.arrays().register(ExternalPlugin(id));
    let actual = reader
        .open_options()
        .open_buffer(bytes)?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;
    assert_arrays_eq!(array, actual, &mut reader.create_execution_ctx());
    Ok(())
}
