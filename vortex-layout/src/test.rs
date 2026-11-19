// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::session::LayoutSession;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| new_session().with_tokio());

pub fn new_session() -> VortexSession {
    VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
}

/// A child strategy that fails with [`FailingStrategy::MESSAGE`] without reading its stream, as a
/// writer that rejects its input does.
pub struct FailingStrategy;

impl FailingStrategy {
    pub const MESSAGE: &str = "child writer rejected its input";
}

#[async_trait::async_trait]
impl crate::LayoutStrategy for FailingStrategy {
    async fn write_stream(
        &self,
        _ctx: crate::LayoutWriterContext,
        _segment_sink: crate::segments::SegmentSinkRef,
        _stream: crate::sequence::SendableSequentialStream,
        _eof: crate::sequence::SequencePointer,
        _session: &VortexSession,
    ) -> VortexResult<crate::LayoutRef> {
        vortex_error::vortex_bail!("{}", Self::MESSAGE)
    }
}

/// A test extension type stored as `{a: i32, b: i64}`. It makes no claims about its storage, so
/// only structural operations and `ext_storage` apply to its values.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TestPoint;

impl TestPoint {
    /// The point dtype with the given top-level nullability.
    pub fn dtype(nullability: Nullability) -> ExtDTypeRef {
        ExtDType::<Self>::try_new(EmptyMetadata, nullability)
            .vortex_expect("TestPoint accepts every nullability")
            .erased()
    }
}

impl ExtVTable for TestPoint {
    type Metadata = EmptyMetadata;
    type StoredMetadata = EmptyMetadata;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("test.point");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        Ok(EmptyMetadata)
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I64, Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        )
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(storage_value)
    }
}
