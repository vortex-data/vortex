// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::tests_outside_test_module)]

//! Arrow export of an extension type without an Arrow plugin, which travels as its storage named
//! with its Vortex ID.

use std::sync::LazyLock;

use arrow_array::Array as ArrowArray;
use arrow_array::Int64Array;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar::ScalarValue;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::Datum;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// A session that can look up [`Tagged`] by its Vortex ID.
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    session.dtypes().register(Tagged);
    vortex_arrow::initialize(&session);
    session
});

static TAGGED_ID: CachedId = CachedId::new("test.tagged");

/// An extension type without an Arrow plugin. It accepts any storage, and its metadata is one tag
/// byte.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Tagged;

impl ExtVTable for Tagged {
    type Metadata = u8;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        *TAGGED_ID
    }

    fn serialize_metadata(&self, tag: &u8) -> VortexResult<Vec<u8>> {
        Ok(vec![*tag])
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<u8> {
        match metadata {
            [tag] => Ok(*tag),
            _ => vortex_bail!("Tagged metadata must be one byte, got {metadata:?}"),
        }
    }

    fn validate_dtype(_ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Ok(storage_value)
    }
}

fn tagged(tag: u8, storage_dtype: DType) -> VortexResult<ExtDTypeRef> {
    Ok(ExtDType::try_with_vtable(Tagged, tag, storage_dtype)?.erased())
}

/// A [`Tagged`] array over the nullable `i64` storage `[1, null, 3]`.
fn tagged_i64s(tag: u8) -> VortexResult<ArrayRef> {
    let storage = PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]).into_array();
    let ext_dtype = tagged(tag, storage.dtype().clone())?;
    Ok(ExtensionArray::try_new(ext_dtype, storage)?.into_array())
}

#[test]
fn export_field_holds_vortex_id_and_base64_metadata() -> VortexResult<()> {
    let field = SESSION
        .arrow()
        .to_arrow_field("t", tagged_i64s(0xff)?.dtype())?;

    // The metadata `[0xff]` is not UTF-8, so the field carries it in padded base64.
    let expected_field = Field::new("t", DataType::Int64, true).with_metadata(
        [
            (EXTENSION_TYPE_NAME_KEY.to_owned(), "test.tagged".to_owned()),
            (EXTENSION_TYPE_METADATA_KEY.to_owned(), "/w==".to_owned()),
        ]
        .into(),
    );
    assert_eq!(field, expected_field);

    Ok(())
}

/// An export to a target with the right Vortex ID but different metadata would give the values a
/// different meaning, so it must fail. A session that knows the Vortex ID compares dtypes, and any
/// other session compares the metadata bytes.
#[rstest]
#[case::registered(SESSION.arrow().clone(), "must name test.tagged[255]")]
#[case::unregistered(ArrowSession::default(), "must carry the metadata of test.tagged[255]")]
fn export_rejects_target_with_other_metadata(
    #[case] session: ArrowSession,
    #[case] message: &str,
) -> VortexResult<()> {
    let other_dtype = tagged(7, DType::Primitive(PType::I64, Nullability::Nullable))?;
    let target = session.to_arrow_field("t", &DType::Extension(other_dtype))?;

    let err = session
        .execute_arrow(
            tagged_i64s(0xff)?,
            Some(&target),
            &mut SESSION.create_execution_ctx(),
        )
        .expect_err("a target with other metadata must be rejected");
    assert!(err.to_string().contains(message), "unexpected error: {err}");

    Ok(())
}

/// An Arrow field has only one extension name, so the storage of an extension type without a plugin
/// must not have its own.
#[test]
fn export_rejects_storage_with_extension_name() -> VortexResult<()> {
    let inner = tagged(1, DType::Primitive(PType::I64, Nullability::NonNullable))?;
    let dtype = DType::Extension(tagged(2, DType::Extension(inner))?);

    let err = SESSION
        .arrow()
        .to_arrow_field("t", &dtype)
        .expect_err("storage with its own extension name must be rejected");
    assert!(
        err.to_string()
            .contains("must not be an Arrow extension type, got test.tagged"),
        "unexpected error: {err}"
    );

    Ok(())
}

/// An Arrow data type, an export without a target, and a [`Datum`] have no field to hold the
/// extension metadata, so they fail for a top-level type without a plugin.
#[test]
fn exports_without_a_field_reject_types_without_a_plugin() -> VortexResult<()> {
    let array = tagged_i64s(0xff)?;
    let message = "needs an extension type with an Arrow plugin";
    let mut ctx = SESSION.create_execution_ctx();

    let errors = [
        SESSION.arrow().to_arrow_datatype(array.dtype()).err(),
        SESSION
            .arrow()
            .execute_arrow(array.clone(), None, &mut ctx)
            .err(),
        Datum::try_new(&array, &mut ctx).err(),
    ];
    for err in errors {
        let err = err.expect("an export without a field must fail instead of losing the type");
        assert!(err.to_string().contains(message), "unexpected error: {err}");
    }

    Ok(())
}
