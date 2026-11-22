// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow plugin impls for the UUID extension type.
//!
//! UUIDs are a canonical Arrow extension type backed by `FixedSizeBinary[16]`. The Vortex side
//! stores them as `FixedSizeList<u8; 16>`, so the conversion is a zero-copy reinterpretation
//! of the byte buffer in both directions.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeBinaryArray;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt8Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use arrow_schema::extension::Uuid as ArrowUuid;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowExtVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowMetadataPolicy;
use vortex_arrow::ArrowSession;
use vortex_arrow::ExecuteArrowExt;
use vortex_arrow::level_validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::Uuid;
use crate::UuidMetadata;

/// The number of bytes in an Arrow `FixedSizeBinary` UUID value.
const UUID_BYTE_LEN: i32 = 16;

/// The cached Id of Arrow's canonical `arrow.uuid` extension type.
static ARROW_UUID: CachedId = CachedId::new(ArrowUuid::NAME);

impl ArrowExtVTable for Uuid {
    fn metadata_policy() -> ArrowMetadataPolicy {
        ArrowMetadataPolicy::PreserveVortex
    }

    fn arrow_ext_id(&self) -> Id {
        *ARROW_UUID
    }

    fn to_arrow_field(
        &self,
        name: &str,
        ext_dtype: &ExtDType<Self>,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let mut field = Field::new(
            name,
            DataType::FixedSizeBinary(UUID_BYTE_LEN),
            ext_dtype.storage_dtype().is_nullable(),
        );
        field
            .try_with_extension_type(ArrowUuid)
            .vortex_expect("UUID field was constructed with FixedSizeBinary(16)");

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        _target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        Ok(ArrowExport::Exported(try_fsl_to_fsb(array, options, ctx)?))
    }

    fn from_arrow_field(
        field: &Field,
        _session: &ArrowSession,
    ) -> VortexResult<Option<ExtDType<Self>>> {
        field.try_extension_type::<ArrowUuid>()?;

        Ok(Some(ExtDType::try_with_vtable(
            Uuid,
            UuidMetadata::default(),
            field.is_nullable().into(),
        )?))
    }

    fn from_arrow_array(
        array: ArrowArrayRef,
        _field: &Field,
        ext_dtype: &ExtDType<Self>,
        _session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        vortex_ensure!(
            matches!(array.data_type(), DataType::FixedSizeBinary(UUID_BYTE_LEN)),
            "Arrow UUID storage must be FixedSizeBinary({UUID_BYTE_LEN}), got {}",
            array.data_type()
        );

        let fsb = array.as_fixed_size_binary();
        let buffer = Buffer::from_arrow_buffer(fsb.values().clone(), Alignment::none());
        let u8_array = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_host(buffer),
            PType::U8,
            Validity::NonNullable,
        );
        // The Arrow bitmap belongs to UUID rows. The Vortex storage uses those rows as lists.
        let validity = level_validity(fsb.nulls(), ext_dtype.storage_dtype().is_nullable())?;

        let storage = FixedSizeListArray::new(
            u8_array.into_array(),
            fsb.value_length() as u32,
            validity,
            fsb.len(),
        )
        .into_array();
        Ok(ArrowImport::Imported(
            ExtensionArray::try_new(ext_dtype.clone().erased(), storage)?.into_array(),
        ))
    }
}

/// Reinterpret a Vortex UUID extension array's `FixedSizeList<u8; 16>` storage as an Arrow
/// `FixedSizeBinary[16]` array, sharing the underlying byte buffer.
fn try_fsl_to_fsb(
    array: ArrayRef,
    options: &ArrowExportOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let executed = array.execute::<ExtensionArray>(ctx)?;
    let storage = executed.storage_array().clone();
    let storage_arrow_type = DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::UInt8, false)),
        UUID_BYTE_LEN,
    );

    let storage_field = Field::new(
        String::new(),
        storage_arrow_type,
        storage.dtype().is_nullable(),
    );

    let arrow_storage = storage.execute_arrow_with_options(Some(&storage_field), options, ctx)?;

    let fsl = arrow_storage.as_fixed_size_list();
    let bytes = fsl
        .values()
        .as_primitive::<UInt8Type>()
        .values()
        .inner()
        .clone();

    Ok(Arc::new(FixedSizeBinaryArray::new(
        fsl.value_length(),
        bytes,
        fsl.nulls().cloned(),
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::DictionaryArray;
    use arrow_array::Int32Array;
    use arrow_array::MapArray as ArrowMapArray;
    use arrow_array::RecordBatch;
    use arrow_array::RecordBatchIterator;
    use arrow_array::StructArray as ArrowStructArray;
    use arrow_array::ffi_stream;
    use arrow_array::new_null_array;
    use arrow_array::types::Int32Type;
    use arrow_buffer::Buffer as ArrowBuffer;
    use arrow_buffer::NullBuffer;
    use arrow_buffer::OffsetBuffer;
    use arrow_buffer::ScalarBuffer;
    use arrow_schema::Fields;
    use arrow_schema::Schema;
    use arrow_schema::extension::EXTENSION_TYPE_METADATA_KEY;
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    use rstest::rstest;
    use uuid::Version;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::Struct;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::StructFields;
    use vortex_array::iter::ArrayIterator;
    use vortex_arrow::ArrowArrayStreamAdapter;
    use vortex_arrow::ArrowSessionExt;
    use vortex_session::VortexSession;

    use super::*;

    #[track_caller]
    fn check_dictionary_round_trip(nullable: bool) -> VortexResult<()> {
        let session = uuid_session();
        let arrow = session.arrow();
        let mut ctx = session.create_execution_ctx();
        let values = versioned_uuid_array(Some(Version::Random), nullable)?;
        let field = arrow.to_arrow_field("id", values.dtype())?;
        let field = field.clone().with_data_type(DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(field.data_type().clone()),
        ));
        let array =
            DictArray::try_new(PrimitiveArray::from_iter([2u32, 1, 0]).into_array(), values)?
                .into_array();

        let exported = array.clone().execute_arrow(Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());

        let imported = arrow.from_arrow_array(exported, &field)?;
        assert_arrays_eq!(imported, array, &mut ctx);

        Ok(())
    }

    /// A session with the UUID extension dtype and its Arrow plugins registered.
    fn uuid_session() -> VortexSession {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    }

    fn uuid_dtype(nullable: bool) -> DType {
        DType::Extension(
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), nullable.into())
                .expect("uuid ext dtype")
                .erased(),
        )
    }

    /// A session that only [`crate::initialize`] sets up imports an `arrow.uuid` field through the
    /// UUID plugin, and a field that names the Vortex ID of the UUID type through the dtype
    /// registry.
    #[test]
    fn uuid_session_imports_by_plugin_and_by_vortex_id() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();
        let dtype = uuid_dtype(false);

        let mut plugin_field = Field::new("id", DataType::FixedSizeBinary(16), false);
        plugin_field.try_with_extension_type(ArrowUuid)?;
        assert_eq!(session.from_arrow_field(&plugin_field)?, dtype);

        let vortex_id_field = session.to_arrow_storage_field("id", dtype.as_extension())?;
        assert_eq!(session.from_arrow_field(&vortex_id_field)?, dtype);

        let array = uuid_array(&session)?;
        let exported = array
            .clone()
            .execute_arrow(Some(&vortex_id_field), &mut ctx)?;
        assert_eq!(exported.data_type(), vortex_id_field.data_type());
        assert_arrays_eq!(
            session.from_arrow_array(exported, &vortex_id_field)?,
            array,
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn schema_roundtrip_preserves_map_uuid_fields() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let map = DType::map(
            uuid_dtype(false),
            uuid_dtype(true),
            true,
            Nullability::Nullable,
        )?;
        let dtype = DType::Struct(
            StructFields::from_iter([(FieldName::from("ids"), map)]),
            Nullability::NonNullable,
        );

        let schema = session.to_arrow_schema(&dtype)?;
        let field = schema.field(0);
        let DataType::Map(entries, keys_sorted) = field.data_type() else {
            panic!("expected Map, got {:?}", field.data_type());
        };
        assert!(*keys_sorted);
        assert_eq!(entries.name(), "entries");
        assert!(!entries.is_nullable());
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("expected map entries struct, got {:?}", entries.data_type());
        };
        assert!(fields[0].has_valid_extension_type::<ArrowUuid>());
        assert!(fields[1].has_valid_extension_type::<ArrowUuid>());
        assert!(!fields[0].is_nullable());
        assert!(fields[1].is_nullable());

        assert_eq!(session.from_arrow_schema(&schema)?, dtype);
        Ok(())
    }

    /// A field that names `arrow.uuid` but breaks its canonical spec fails to import instead of
    /// importing as plain storage.
    #[rstest]
    #[case::fixed_size_binary_8(DataType::FixedSizeBinary(8), "")]
    #[case::metadata(DataType::FixedSizeBinary(16), "v4")]
    fn from_arrow_field_rejects_invalid_uuid(#[case] data_type: DataType, #[case] metadata: &str) {
        let field = Field::new("id", data_type, false).with_metadata(
            [
                (
                    EXTENSION_TYPE_NAME_KEY.to_string(),
                    ArrowUuid::NAME.to_string(),
                ),
                (
                    EXTENSION_TYPE_METADATA_KEY.to_string(),
                    metadata.to_string(),
                ),
            ]
            .into(),
        );
        let err = uuid_session()
            .arrow()
            .from_arrow_field(&field)
            .expect_err("a field outside the arrow.uuid spec must fail to import");
        assert!(
            err.to_string().contains("as arrow.uuid"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn to_arrow_datatype_dispatches_plugins() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        assert_eq!(
            session.to_arrow_datatype(&uuid_dtype(false))?,
            DataType::FixedSizeBinary(16)
        );
        assert_eq!(
            session.to_arrow_datatype(&DType::Utf8(Nullability::Nullable))?,
            DataType::Utf8View
        );
        Ok(())
    }

    #[test]
    fn from_arrow_datatype_recurses_into_nested_extension_fields() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let mut elem = Field::new("item", DataType::FixedSizeBinary(16), false);
        elem.try_with_extension_type(ArrowUuid)?;
        let data_type = DataType::List(Arc::new(elem));

        let dtype = session.from_arrow_datatype(&data_type, Nullability::Nullable)?;
        assert_eq!(
            dtype,
            DType::List(Arc::new(uuid_dtype(false)), Nullability::Nullable)
        );
        Ok(())
    }

    /// Import an Arrow FixedSizeBinary UUID column as a Vortex extension array.
    fn uuid_array(session: &ArrowSession) -> VortexResult<ArrayRef> {
        let mut field = Field::new("id", DataType::FixedSizeBinary(16), false);
        field.try_with_extension_type(ArrowUuid)?;
        let arrow_array: ArrowArrayRef = Arc::new(FixedSizeBinaryArray::try_from_iter(
            [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
        )?);
        session.from_arrow_array(arrow_array, &field)
    }

    #[test]
    fn dictionary_preserves_outer_uuid_metadata() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let uuid_field =
            Field::new("id", DataType::FixedSizeBinary(16), false).with_extension_type(ArrowUuid);
        let values = FixedSizeBinaryArray::try_from_iter([[1u8; 16], [2u8; 16]].into_iter())?;
        let dict = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![1, 0, 1]),
            Arc::new(values),
        )?;
        let field = uuid_field.clone().with_data_type(dict.data_type().clone());
        let schema = Arc::new(Schema::new(vec![field]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(dict)])?;

        let array = session.from_arrow_record_batch(batch, &schema)?;
        assert_eq!(array.dtype(), &session.from_arrow_schema(&schema)?);
        let ids = array.as_::<Struct>().unmasked_field_by_name("id")?.clone();
        assert_eq!(ids.dtype(), &uuid_dtype(false));
        let exported = ids.execute_arrow(
            Some(&uuid_field),
            &mut vortex_session.create_execution_ctx(),
        )?;
        let expected =
            FixedSizeBinaryArray::try_from_iter([[2u8; 16], [1u8; 16], [2u8; 16]].into_iter())?;
        assert_eq!(exported.to_data(), expected.to_data());
        Ok(())
    }

    fn expected_uuids(bytes: [[u8; 16]; 2]) -> VortexResult<ExtensionArray> {
        let storage = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(bytes.into_iter().flatten()).into_array(),
            16,
            Validity::NonNullable,
            2,
        )?;
        ExtensionArray::try_new(
            uuid_dtype(false).as_extension().clone(),
            storage.into_array(),
        )
    }

    #[test]
    fn test_import_non_nullable_uuid_dictionary_without_values() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let field = Field::new("id", DataType::FixedSizeBinary(16), false)
            .with_extension_type(ArrowUuid)
            .with_data_type(DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(DataType::FixedSizeBinary(16)),
            ));
        let array = new_null_array(field.data_type(), 2);

        let expected = expected_uuids([[0u8; 16]; 2])?;
        let imported = vortex_session.arrow().from_arrow_array(array, &field)?;
        assert_arrays_eq!(
            imported,
            expected,
            &mut vortex_session.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_import_non_nullable_uuid_field_under_null_struct_row() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let field =
            Field::new("id", DataType::FixedSizeBinary(16), false).with_extension_type(ArrowUuid);
        let nulls = NullBuffer::from(vec![true, false]);
        let bytes = [[1u8; 16], [7u8; 16]];
        let uuids = FixedSizeBinaryArray::try_new(
            16,
            ArrowBuffer::from_iter(bytes.into_iter().flatten()),
            Some(nulls.clone()),
        )?;
        let rows =
            ArrowStructArray::try_new(vec![field].into(), vec![Arc::new(uuids)], Some(nulls))?;

        let expected = expected_uuids(bytes)?;
        let imported = vortex_session
            .arrow()
            .from_arrow_array(Arc::new(rows), true)?;
        let imported = imported.as_::<Struct>();
        assert_arrays_eq!(
            imported.unmasked_field(0),
            expected,
            &mut vortex_session.create_execution_ctx()
        );
        Ok(())
    }

    /// Importing by nullability instead of by [`Field`] synthesizes an anonymous field from the
    /// array's own data type, so extension metadata on *nested* fields still reaches its importer.
    /// A [`bool`] and the equivalent [`Nullability`] must agree.
    #[rstest]
    #[case(true)]
    #[case(false)]
    fn from_arrow_array_by_nullability(#[case] nullable: bool) -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();

        let mut uuid_field = Field::new("id", DataType::FixedSizeBinary(16), false);
        uuid_field.try_with_extension_type(ArrowUuid)?;
        let uuids: ArrowArrayRef = Arc::new(FixedSizeBinaryArray::try_from_iter(
            [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
        )?);
        let arrow_struct: ArrowArrayRef = Arc::new(ArrowStructArray::try_new(
            Fields::from(vec![uuid_field]),
            vec![uuids],
            None,
        )?);

        let array = session.from_arrow_array(ArrowArrayRef::clone(&arrow_struct), nullable)?;
        assert_eq!(array.dtype().nullability(), nullable.into());
        let DType::Struct(fields, _) = array.dtype() else {
            panic!("expected a Struct dtype, got {}", array.dtype());
        };
        assert!(
            fields
                .field_by_index(0)
                .vortex_expect("struct dtype has one field")
                .as_extension()
                .is::<Uuid>(),
            "expected the nested Uuid extension to survive, got {}",
            array.dtype()
        );

        // The `Nullability` form is equivalent to the `bool` form.
        let by_nullability =
            session.from_arrow_array(Arc::clone(&arrow_struct), Nullability::from(nullable))?;
        assert_eq!(by_nullability.dtype(), array.dtype());
        let mut ctx = vortex_session.create_execution_ctx();
        for imported in [array, by_nullability] {
            let exported = imported.execute_arrow(None, &mut ctx)?;
            assert_eq!(exported.to_data(), arrow_struct.to_data());
        }
        Ok(())
    }

    /// The adapter imports through the [`ArrowSession`], so a UUID column arrives as a Vortex
    /// extension array rather than its `FixedSizeList` storage.
    #[test]
    fn stream_preserves_extension_types() -> VortexResult<()> {
        let mut field = Field::new("id", DataType::FixedSizeBinary(16), false);
        field.try_with_extension_type(ArrowUuid)?;
        let schema = Arc::new(Schema::new(vec![field]));
        let ids = FixedSizeBinaryArray::try_from_iter(
            [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
        )?;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(ids)])?;

        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let stream = ffi_stream::ArrowArrayStreamReader::try_new(
            ffi_stream::FFI_ArrowArrayStream::new(Box::new(reader)),
        )?;

        let vortex_session = uuid_session();
        let mut adapter = ArrowArrayStreamAdapter::try_new(&vortex_session.arrow(), stream)?;

        let DType::Struct(fields, _) = adapter.dtype().clone() else {
            panic!("expected a struct dtype, got {}", adapter.dtype());
        };
        assert!(
            fields
                .field_by_index(0)
                .vortex_expect("one field")
                .as_extension()
                .is::<Uuid>()
        );

        let array = adapter.next().vortex_expect("one batch")?;
        assert_eq!(array.dtype(), adapter.dtype());
        assert!(
            array
                .as_::<Struct>()
                .unmasked_field(0)
                .dtype()
                .as_extension()
                .is::<Uuid>()
        );
        assert_eq!(array.len(), 2);
        let mut ctx = vortex_session.create_execution_ctx();
        let exported = array.execute_arrow(None, &mut ctx)?;
        let ids = exported.as_struct().column(0).as_fixed_size_binary();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids.null_count(), 0);
        assert_eq!(ids.value(0), b"0123456789abcdef");
        assert_eq!(ids.value(1), b"fedcba9876543210");
        assert!(adapter.next().is_none());

        Ok(())
    }

    #[test]
    fn map_roundtrip_preserves_nested_uuid_fields() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();

        let mut key_field = Field::new("key", DataType::FixedSizeBinary(16), false);
        key_field.try_with_extension_type(ArrowUuid)?;
        let mut value_field = Field::new("value", DataType::FixedSizeBinary(16), true);
        value_field.try_with_extension_type(ArrowUuid)?;
        let fields = Fields::from(vec![key_field, value_field]);
        let entries_field = Arc::new(Field::new_struct("entries", fields.clone(), false));
        let field = Field::new(
            "ids",
            DataType::Map(Arc::clone(&entries_field), true),
            false,
        );
        let keys = FixedSizeBinaryArray::try_from_iter(
            [
                b"0123456789abcdef".as_slice(),
                b"fedcba9876543210".as_slice(),
            ]
            .into_iter(),
        )?;
        let values = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            [Some(b"aaaaaaaaaaaaaaaa".as_slice()), None].into_iter(),
            16,
        )?;
        let entries =
            ArrowStructArray::try_new(fields, vec![Arc::new(keys), Arc::new(values)], None)?;
        let arrow = ArrowMapArray::try_new(
            entries_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 2])),
            entries,
            None,
            true,
        )?;

        let expected = arrow.to_data();
        let vortex = session.from_arrow_array(Arc::new(arrow), &field)?;
        let DType::Map(map_dtype, Nullability::NonNullable) = vortex.dtype() else {
            panic!("expected map dtype, got {}", vortex.dtype());
        };
        assert!(map_dtype.key_dtype().is_extension());
        assert!(map_dtype.value_dtype().is_extension());

        let exported = vortex.execute_arrow(Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());
        assert_eq!(exported.as_map().value_offsets(), &[0, 2]);
        assert_eq!(exported.to_data(), expected);

        Ok(())
    }

    /// Builds three UUIDs with valid version bits, with a null middle row when nullable.
    fn versioned_uuid_array(version: Option<Version>, nullable: bool) -> VortexResult<ArrayRef> {
        let dtype =
            ExtDType::try_with_vtable(Uuid, UuidMetadata { version }, nullable.into())?.erased();

        let bytes = match version {
            Some(Version::Max) => [u8::MAX; 16],
            Some(Version::Nil) | None => [0; 16],
            Some(version) => uuid::Builder::from_bytes([0; 16])
                .with_version(version)
                .with_variant(uuid::Variant::RFC4122)
                .into_uuid()
                .into_bytes(),
        };

        let validity = if nullable {
            Validity::from_iter([
                true,  // First UUID.
                false, // Null middle UUID.
                true,  // Last UUID.
            ])
        } else {
            Validity::NonNullable
        };
        let storage = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter((0..3).flat_map(|_| bytes)).into_array(),
            16,
            validity,
            3,
        )?;

        Ok(ExtensionArray::try_new(dtype, storage.into_array())?.into_array())
    }

    #[rstest]
    #[case::non_nullable(false)]
    #[case::nullable(true)]
    fn dictionary_uuid_round_trip(#[case] nullable: bool) -> VortexResult<()> {
        check_dictionary_round_trip(nullable)
    }

    #[test]
    fn mismatched_metadata_is_rejected() -> VortexResult<()> {
        let session = uuid_session();
        let mut ctx = session.create_execution_ctx();
        let array = uuid_array(&session.arrow())?;
        let target_array = versioned_uuid_array(Some(Version::Random), false)?;
        let field = session.arrow().to_arrow_field("id", target_array.dtype())?;

        assert!(array.execute_arrow(Some(&field), &mut ctx).is_err());

        Ok(())
    }
}
