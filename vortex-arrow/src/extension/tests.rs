// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Int64Array;
use arrow_schema::DataType;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexError;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::*;
use crate::ArrowExportVTable;
use crate::ArrowImportVTable;
use crate::ArrowSessionExt;
use crate::ExecuteArrowExt;
use crate::ext_metadata::VORTEX_EXTENSION_METADATA_KEY;
use crate::ext_metadata::set_arrow_extension_name;

/// Checks a tagged field's complete dtype and round-trips its values through the registered policy.
#[track_caller]
fn round_trip_field<const PRESERVE_VORTEX: bool>(tag: u8, nullable: bool) -> VortexResult<Field> {
    let session = tagged_session::<PRESERVE_VORTEX>();
    let arrow = session.arrow();
    let dtype = DType::Extension(Tagged::<PRESERVE_VORTEX>::dtype(tag, nullable)?.erased());

    let field = arrow.to_arrow_field("tag", &dtype)?;
    assert_eq!(arrow.from_arrow_field(&field)?, dtype);

    let storage = if nullable {
        PrimitiveArray::from_option_iter([Some(1i64)]).into_array()
    } else {
        PrimitiveArray::from_iter([1i64]).into_array()
    };
    let expected = ExtensionArray::try_new(dtype.as_extension().clone(), storage)?.into_array();
    let mut ctx = session.create_execution_ctx();

    let array: ArrowArrayRef = Arc::new(Int64Array::from(vec![1]));
    let imported = arrow.from_arrow_array(array, &field)?;
    assert_arrays_eq!(imported, expected, &mut ctx);

    let exported = expected.clone().execute_arrow(Some(&field), &mut ctx)?;
    let round_tripped = arrow.from_arrow_array(exported, &field)?;
    assert_arrays_eq!(round_tripped, expected, &mut ctx);

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

/// The native field always carries the decimal text `255`, so it can express no other tag. This
/// text differs from the one-byte wire metadata of the same tag, as a real plugin's native metadata
/// can. Tag zero deliberately has an empty serialized payload.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct Tagged<const PRESERVE_VORTEX: bool>;

impl<const PRESERVE_VORTEX: bool> Tagged<PRESERVE_VORTEX> {
    fn dtype(tag: u8, nullable: bool) -> VortexResult<ExtDType<Self>> {
        ExtDType::try_with_vtable(Self, tag, nullable.into())
    }
}

impl<const PRESERVE_VORTEX: bool> ExtVTable for Tagged<PRESERVE_VORTEX> {
    type Metadata = u8;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> Id {
        static ID: CachedId = CachedId::new("test.typed_tag");
        *ID
    }

    fn serialize_metadata(&self, tag: &u8) -> VortexResult<Vec<u8>> {
        Ok(if *tag == 0 { vec![] } else { vec![*tag] })
    }

    fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::StoredMetadata> {
        match bytes {
            [] => Ok(0),
            [tag] => Ok(*tag),
            _ => vortex_bail!("tag must contain at most one byte"),
        }
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(metadata: &Self::Metadata) -> DType {
        // Tag 254 declares a different width to exercise metadata restoration mismatches.
        let ptype = if *metadata == 254 {
            PType::I32
        } else {
            PType::I64
        };
        DType::Primitive(ptype, false.into())
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
        field
            .metadata_mut()
            .insert(EXTENSION_TYPE_METADATA_KEY.to_owned(), "255".to_owned());

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
        let tag = field
            .metadata()
            .get(EXTENSION_TYPE_METADATA_KEY)
            .map_or(Ok(255), |tag| tag.parse::<u8>())
            .map_err(|error| VortexError::external(error).with_context("invalid native tag"))?;

        Ok(Some(Self::dtype(tag, field.is_nullable())?))
    }

    fn from_arrow_array(
        array: ArrowArrayRef,
        field: &Field,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        let storage = session.from_arrow_array(array, field.is_nullable())?;
        let imported = ExtensionArray::try_new(ext_dtype.clone().erased(), storage)?.into_array();

        Ok(ArrowImport::Imported(imported))
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

#[rstest]
// Native text "7" means tag 7, but its byte is the Vortex wire encoding of tag 55.
#[case::different_native_parameter(b'7', DataType::Int64, "7", "must describe")]
#[case::different_physical_storage(255, DataType::Int32, "255", "must have an Arrow type")]
fn declined_export_rejects_incompatible_target(
    #[case] tag: u8,
    #[case] data_type: DataType,
    #[case] native_metadata: &str,
    #[case] message: &str,
) -> VortexResult<()> {
    let session = tagged_session::<false>();
    let dtype = Tagged::<false>::dtype(tag, false)?.erased();
    let array = ExtensionArray::try_new(dtype, PrimitiveArray::from_iter([1i64]).into_array())?
        .into_array();
    let mut field = Field::new("tag", data_type, false);
    set_arrow_extension_name(&mut field, Tagged::<false>.id().to_string());
    field.metadata_mut().insert(
        EXTENSION_TYPE_METADATA_KEY.to_owned(),
        native_metadata.to_owned(),
    );

    let error = array
        .execute_arrow(Some(&field), &mut session.create_execution_ctx())
        .expect_err("native parameter and physical storage mismatches must be rejected");
    assert!(error.to_string().contains(message), "{error}");

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

#[test]
fn preserved_metadata_must_declare_inferred_storage() -> VortexResult<()> {
    let session = tagged_session::<true>();
    let mut field = round_trip_field::<true>(7, false)?;
    field
        .metadata_mut()
        .insert(VORTEX_EXTENSION_METADATA_KEY.to_owned(), "/g==".to_owned());
    let error = session
        .arrow()
        .from_arrow_field(&field)
        .expect_err("tag 254 declares i32 over inferred i64 storage");
    let message = error.to_string();
    assert!(
        message.contains("storage must be i32, got i64"),
        "{message}"
    );
    Ok(())
}
