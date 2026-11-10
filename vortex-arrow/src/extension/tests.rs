// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Int64Array;
use arrow_schema::DataType;
use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar::ScalarValue;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::*;
use crate::ArrowExportVTable;
use crate::ArrowImportVTable;
use crate::ArrowSessionExt;
use crate::ext_metadata::VORTEX_EXTENSION_METADATA_KEY;
use crate::ext_metadata::set_arrow_extension_name;

/// Exports a tagged field through its registered policy and checks its complete dtype on import.
#[track_caller]
fn round_trip_field<const PRESERVE_VORTEX: bool>(tag: u8, nullable: bool) -> VortexResult<Field> {
    let session = tagged_session::<PRESERVE_VORTEX>();
    let arrow = session.arrow();
    let dtype = DType::Extension(Tagged::<PRESERVE_VORTEX>::dtype(tag, nullable)?.erased());

    let field = arrow.to_arrow_field("tag", &dtype)?;
    assert_eq!(arrow.from_arrow_field(&field)?, dtype);

    Ok(field)
}

fn tagged_session<const PRESERVE_VORTEX: bool>() -> VortexSession {
    let session = VortexSession::empty();
    session.dtypes().register(Tagged::<PRESERVE_VORTEX>);
    session
        .arrow()
        .register_extension(Tagged::<PRESERVE_VORTEX>);

    session
}

fn raw_tagged_session() -> VortexSession {
    let session = VortexSession::empty();
    let plugin = Arc::new(RawTagged);
    let exporter = Arc::clone(&plugin);
    session.arrow().register_exporter(exporter);
    session.arrow().register_importer(plugin);

    session
}

/// The native field infers tag 255. Tag zero deliberately has an empty serialized payload.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct Tagged<const PRESERVE_VORTEX: bool>;

impl<const PRESERVE_VORTEX: bool> Tagged<PRESERVE_VORTEX> {
    fn dtype(tag: u8, nullable: bool) -> VortexResult<ExtDType<Self>> {
        ExtDType::try_with_vtable(Self, tag, DType::Primitive(PType::I64, nullable.into()))
    }
}

impl<const PRESERVE_VORTEX: bool> ExtVTable for Tagged<PRESERVE_VORTEX> {
    type Metadata = u8;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> Id {
        static ID: CachedId = CachedId::new("test.typed_tag");
        *ID
    }

    fn serialize_metadata(&self, tag: &u8) -> VortexResult<Vec<u8>> {
        Ok(if *tag == 0 { vec![] } else { vec![*tag] })
    }

    fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<u8> {
        match bytes {
            [] => Ok(0),
            [tag] => Ok(*tag),
            _ => vortex_bail!("tag must contain at most one byte"),
        }
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure_eq!(
            *ext_dtype.storage_dtype(),
            DType::Primitive(PType::I64, ext_dtype.storage_dtype().nullability())
        );

        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &ExtDType<Self>,
        value: &'a ScalarValue,
    ) -> VortexResult<&'a ScalarValue> {
        Ok(value)
    }
}

impl<const PRESERVE_VORTEX: bool> ArrowExtVTable for Tagged<PRESERVE_VORTEX> {
    fn metadata_policy() -> ArrowMetadataPolicy {
        if PRESERVE_VORTEX {
            ArrowMetadataPolicy::PreserveVortex
        } else {
            ArrowMetadataPolicy::ArrowNative
        }
    }

    fn arrow_ext_id(&self) -> Id {
        self.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        ext_dtype: &ExtDType<Self>,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let mut field = Field::new(
            name,
            DataType::Int64,
            ext_dtype.storage_dtype().is_nullable(),
        );
        set_arrow_extension_name(&mut field, self.arrow_ext_id().to_string());

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        _target: &Field,
        _options: &ArrowExportOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        Ok(ArrowExport::Unsupported(array))
    }

    fn from_arrow_field(
        field: &Field,
        _session: &ArrowSession,
    ) -> VortexResult<Option<ExtDType<Self>>> {
        Ok(Some(Self::dtype(255, field.is_nullable())?))
    }

    fn from_arrow_array(
        array: ArrowArrayRef,
        _field: &Field,
        _ext_dtype: &ExtDType<Self>,
        _session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        Ok(ArrowImport::Unsupported(array))
    }
}

/// Bypasses the typed adapter and deliberately drops the extension when importing arrays.
#[derive(Debug)]
struct RawTagged;

impl ArrowExportVTable for RawTagged {
    fn arrow_ext_id(&self) -> Id {
        Tagged::<false>.id()
    }

    fn vortex_id(&self) -> Id {
        Tagged::<false>.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        <Tagged<false> as ArrowExtVTable>::to_arrow_field(
            &Tagged,
            name,
            &dtype.as_extension().clone().downcast::<Tagged<false>>(),
            session,
        )
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        _target: &Field,
        _options: &ArrowExportOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        Ok(ArrowExport::Unsupported(array))
    }
}

impl ArrowImportVTable for RawTagged {
    fn arrow_ext_id(&self) -> Id {
        Tagged::<false>.id()
    }

    fn from_arrow_field(
        &self,
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        Ok(
            <Tagged<false> as ArrowExtVTable>::from_arrow_field(field, session)?
                .map(|dtype| DType::Extension(dtype.erased())),
        )
    }

    fn from_arrow_array(
        &self,
        _array: ArrowArrayRef,
        _field: &Field,
        _dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        Ok(ArrowImport::Imported(
            PrimitiveArray::from_iter([1i64]).into_array(),
        ))
    }
}

#[rstest]
#[case::native(false, 255, false, None)]
#[case::nullable_native(false, 255, true, None)]
#[case::already_exact(true, 255, false, None)]
#[case::empty_metadata(true, 0, true, Some(""))]
#[case::nonempty_metadata(true, 7, false, Some("Bw=="))]
fn policy_preserves_dtype(
    #[case] preserve_vortex: bool,
    #[case] tag: u8,
    #[case] nullable: bool,
    #[case] expected_payload: Option<&str>,
) -> VortexResult<()> {
    let field = if preserve_vortex {
        round_trip_field::<true>(tag, nullable)?
    } else {
        round_trip_field::<false>(tag, nullable)?
    };

    assert_eq!(
        field
            .metadata()
            .get(VORTEX_EXTENSION_METADATA_KEY)
            .map(String::as_str),
        expected_payload
    );

    Ok(())
}

#[test]
fn lossy_native_policy_is_rejected() -> VortexResult<()> {
    let error =
        round_trip_field::<false>(7, false).expect_err("the native field cannot express tag 7");

    assert!(error.to_string().contains("must preserve its Vortex dtype"));

    Ok(())
}

#[test]
fn raw_exporter_must_preserve_dtype() -> VortexResult<()> {
    let session = raw_tagged_session();
    let dtype = DType::Extension(Tagged::<false>::dtype(7, false)?.erased());
    let error = session
        .arrow()
        .to_arrow_field("tag", &dtype)
        .expect_err("the raw exporter's field loses tag 7");

    assert!(error.to_string().contains("must preserve its Vortex dtype"));

    Ok(())
}

#[test]
fn raw_importer_must_preserve_dtype() -> VortexResult<()> {
    let session = raw_tagged_session();
    let dtype = DType::Extension(Tagged::<false>::dtype(255, false)?.erased());
    let field = session.arrow().to_arrow_field("tag", &dtype)?;
    let array: ArrowArrayRef = Arc::new(Int64Array::from(vec![1]));
    let error = session
        .arrow()
        .from_arrow_array(array, &field)
        .expect_err("the raw importer discards the inferred extension dtype");

    assert!(
        error
            .to_string()
            .contains("must preserve the dtype inferred from its field")
    );

    Ok(())
}
