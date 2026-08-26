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
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowImportVTable;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::has_valid_extension_type;
use vortex_arrow::nulls;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::Uuid;
use crate::UuidMetadata;

/// The number of bytes in an Arrow `FixedSizeBinary` UUID value.
const UUID_BYTE_LEN: i32 = 16;

/// The cached Id of Arrow's canonical `arrow.uuid` extension type.
static ARROW_UUID: CachedId = CachedId::new(ArrowUuid::NAME);

impl ArrowExportVTable for Uuid {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_UUID
    }

    fn vortex_id(&self) -> Id {
        Uuid.id()
    }

    // Encode all of these.
    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let mut field = Field::new(
            name.to_string(),
            DataType::FixedSizeBinary(UUID_BYTE_LEN),
            dtype.is_nullable(),
        );
        field
            .try_with_extension_type(ArrowUuid)
            .vortex_expect("FixedSizeBinary[16] is correct type for ArrowUuid");
        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        _target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_uuid = array
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<Uuid>())
            .unwrap_or(false);
        if !is_uuid {
            return Ok(ArrowExport::Unsupported(array));
        }
        Ok(ArrowExport::Exported(try_fsl_to_fsb(array, ctx)?))
    }
}

impl ArrowImportVTable for Uuid {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_UUID
    }

    fn from_arrow_field(
        &self,
        field: &Field,
        _session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        if !has_valid_extension_type::<ArrowUuid>(field) {
            return Ok(None);
        }

        let storage_dtype = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            UUID_BYTE_LEN as u32,
            field.is_nullable().into(),
        );

        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage_dtype)?.erased(),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        _field: &Field,
        dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        let DType::Extension(dtype) = dtype else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !matches!(array.data_type(), DataType::FixedSizeBinary(UUID_BYTE_LEN))
            || !dtype.is::<Uuid>()
        {
            return Ok(ArrowImport::Unsupported(array));
        }

        let fsb = array.as_fixed_size_binary();
        let buffer = Buffer::from_arrow_buffer(fsb.values().clone(), Alignment::none());
        let u8_array = PrimitiveArray::from_buffer_handle(
            BufferHandle::new_host(buffer),
            PType::U8,
            Validity::NonNullable,
        );
        let validity = nulls(fsb.nulls(), dtype.is_nullable())?;

        let storage = FixedSizeListArray::new(
            u8_array.into_array(),
            fsb.value_length() as u32,
            validity,
            fsb.len(),
        )
        .into_array();
        Ok(ArrowImport::Imported(
            ExtensionArray::new(dtype.clone(), storage).into_array(),
        ))
    }
}

/// Reinterpret a Vortex UUID extension array's `FixedSizeList<u8; 16>` storage as an Arrow
/// `FixedSizeBinary[16]` array, sharing the underlying byte buffer.
fn try_fsl_to_fsb(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrowArrayRef> {
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

    let session = ctx.session().clone();
    let arrow_storage = session
        .arrow()
        .execute_arrow(storage, Some(&storage_field), ctx)?;

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
    use arrow_array::ListArray as ArrowListArray;
    use arrow_array::MapArray as ArrowMapArray;
    use arrow_array::RecordBatch;
    use arrow_array::RecordBatchIterator;
    use arrow_array::RunArray;
    use arrow_array::StructArray as ArrowStructArray;
    use arrow_array::ffi_stream;
    use arrow_array::types::Int32Type;
    use arrow_buffer::OffsetBuffer;
    use arrow_buffer::ScalarBuffer;
    use arrow_schema::Fields;
    use arrow_schema::Schema;
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::Struct;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::StructFields;
    use vortex_array::iter::ArrayIterator;
    use vortex_arrow::ArrowArrayStreamAdapter;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;

    /// A session with the UUID extension dtype and its Arrow plugins registered.
    fn uuid_session() -> VortexSession {
        let session = array_session();
        crate::initialize(&session);
        session
    }

    fn uuid_dtype(nullable: bool) -> DType {
        let storage = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            16,
            nullable.into(),
        );
        DType::Extension(
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage)
                .expect("uuid ext dtype")
                .erased(),
        )
    }

    #[test]
    fn to_arrow_field_top_level_uuid_carries_extension_metadata() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let field = session.to_arrow_field("id", &uuid_dtype(false))?;
        assert!(has_valid_extension_type::<ArrowUuid>(&field));
        Ok(())
    }

    #[test]
    fn to_arrow_field_struct_with_nested_uuid_preserves_metadata() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let dtype = DType::Struct(
            StructFields::from_iter([(FieldName::from("id"), uuid_dtype(false))]),
            Nullability::NonNullable,
        );
        let field = session.to_arrow_field("row", &dtype)?;
        let DataType::Struct(inner) = field.data_type() else {
            panic!("expected Struct, got {:?}", field.data_type());
        };
        assert_eq!(inner.len(), 1);
        assert_eq!(inner[0].data_type(), &DataType::FixedSizeBinary(16));
        assert!(has_valid_extension_type::<ArrowUuid>(&inner[0]));
        Ok(())
    }

    #[test]
    fn to_arrow_field_list_of_uuid_preserves_metadata() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let dtype = DType::List(Arc::new(uuid_dtype(true)), Nullability::NonNullable);
        let field = session.to_arrow_field("ids", &dtype)?;
        let DataType::List(elem) = field.data_type() else {
            panic!("expected List, got {:?}", field.data_type());
        };
        assert!(has_valid_extension_type::<ArrowUuid>(elem));
        Ok(())
    }

    #[test]
    fn to_arrow_field_fixed_size_list_of_uuid_preserves_metadata() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let dtype = DType::FixedSizeList(Arc::new(uuid_dtype(false)), 3, Nullability::NonNullable);
        let field = session.to_arrow_field("triple", &dtype)?;
        let DataType::FixedSizeList(elem, size) = field.data_type() else {
            panic!("expected FixedSizeList, got {:?}", field.data_type());
        };
        assert_eq!(*size, 3);
        assert!(has_valid_extension_type::<ArrowUuid>(elem));
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
        assert!(has_valid_extension_type::<ArrowUuid>(&fields[0]));
        assert!(has_valid_extension_type::<ArrowUuid>(&fields[1]));
        assert!(!fields[0].is_nullable());
        assert!(fields[1].is_nullable());

        assert_eq!(session.from_arrow_schema(&schema)?, dtype);
        Ok(())
    }

    #[test]
    fn to_arrow_schema_struct_of_struct_uuid() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let inner = DType::Struct(
            StructFields::from_iter([(FieldName::from("id"), uuid_dtype(true))]),
            Nullability::NonNullable,
        );
        let outer = DType::Struct(
            StructFields::from_iter([(FieldName::from("payload"), inner)]),
            Nullability::NonNullable,
        );
        let schema = session.to_arrow_schema(&outer)?;
        let payload = schema.field(0);
        let DataType::Struct(inner_fields) = payload.data_type() else {
            panic!("expected Struct, got {:?}", payload.data_type());
        };
        assert!(has_valid_extension_type::<ArrowUuid>(&inner_fields[0]));
        Ok(())
    }

    #[test]
    fn from_arrow_field_recurses_into_nested_uuid() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let mut elem = Field::new("item", DataType::FixedSizeBinary(16), false);
        elem.try_with_extension_type(ArrowUuid)?;
        let outer = Field::new("ids", DataType::List(Arc::new(elem)), false);

        let dtype = session.from_arrow_field(&outer)?;
        let DType::List(inner_dt, _) = dtype else {
            panic!("expected List dtype, got {dtype}");
        };
        assert!(
            matches!(inner_dt.as_ref(), DType::Extension(ext) if ext.id() == Uuid.id()),
            "expected Uuid extension element, got {inner_dt}",
        );
        Ok(())
    }

    #[test]
    fn schema_roundtrip_preserves_nested_uuid() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();
        let dtype = DType::Struct(
            StructFields::from_iter([
                (FieldName::from("id"), uuid_dtype(false)),
                (
                    FieldName::from("ids"),
                    DType::List(Arc::new(uuid_dtype(true)), Nullability::NonNullable),
                ),
            ]),
            Nullability::NonNullable,
        );
        let schema = session.to_arrow_schema(&dtype)?;
        let roundtripped = session.from_arrow_schema(&schema)?;
        assert_eq!(roundtripped, dtype);
        Ok(())
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
        let DType::List(inner_dt, Nullability::Nullable) = dtype else {
            panic!("expected nullable List dtype, got {dtype}");
        };
        assert!(
            matches!(inner_dt.as_ref(), DType::Extension(ext) if ext.id() == Uuid.id()),
            "expected Uuid extension element, got {inner_dt}",
        );
        Ok(())
    }

    #[test]
    fn execute_arrow_target_none_preserves_top_level_uuid_metadata() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();

        let mut field = Field::new("id", DataType::FixedSizeBinary(16), false);
        field.try_with_extension_type(ArrowUuid)?;
        let arrow_array: ArrowArrayRef = Arc::new(FixedSizeBinaryArray::try_from_iter(
            [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
        )?);

        let vortex_array = session.from_arrow_array(arrow_array, &field)?;

        let vortex_ext = vortex_array.dtype().as_extension();
        assert!(vortex_ext.is::<Uuid>());

        let exported = session.execute_arrow(vortex_array, None, &mut ctx)?;
        assert_eq!(exported.data_type(), &DataType::FixedSizeBinary(16));
        let fsb = exported.as_fixed_size_binary();
        assert_eq!(fsb.len(), 2);
        assert_eq!(fsb.value(0), b"0123456789abcdef");
        assert_eq!(fsb.value(1), b"fedcba9876543210");
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

    /// Exporting a struct that contains an extension column with no target field must still route
    /// the column through its export plugin *and* re-attach the Arrow extension metadata to the
    /// inferred child field.
    #[test]
    fn execute_arrow_target_none_preserves_nested_uuid_metadata() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();

        let uuids = uuid_array(&session)?;
        let struct_array = StructArray::try_new(
            FieldNames::from(["id"]),
            vec![uuids],
            2,
            Validity::NonNullable,
        )?
        .into_array();

        let exported = session.execute_arrow(struct_array, None, &mut ctx)?;
        let DataType::Struct(fields) = exported.data_type() else {
            panic!("expected Struct, got {:?}", exported.data_type());
        };
        assert_eq!(fields[0].data_type(), &DataType::FixedSizeBinary(16));
        assert!(has_valid_extension_type::<ArrowUuid>(&fields[0]));

        let uuids = exported.as_struct().column(0).as_fixed_size_binary();
        assert_eq!(uuids.value(0), b"0123456789abcdef");
        assert_eq!(uuids.value(1), b"fedcba9876543210");
        Ok(())
    }

    /// Exporting a list of extension elements with no target field must infer an element field that
    /// still carries the Arrow extension metadata.
    #[test]
    fn execute_arrow_target_none_preserves_list_element_uuid_metadata() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();

        let list = ListArray::try_new(
            uuid_array(&session)?,
            PrimitiveArray::from_iter([0i32, 1, 2]).into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let exported = session.execute_arrow(list, None, &mut ctx)?;
        let DataType::List(elem) = exported.data_type() else {
            panic!("expected List, got {:?}", exported.data_type());
        };
        assert_eq!(elem.data_type(), &DataType::FixedSizeBinary(16));
        assert!(has_valid_extension_type::<ArrowUuid>(elem));

        let uuids = exported.as_list::<i32>().values().as_fixed_size_binary();
        assert_eq!(uuids.value(0), b"0123456789abcdef");
        assert_eq!(uuids.value(1), b"fedcba9876543210");
        Ok(())
    }

    /// An Arrow run-end array whose values field carries extension metadata must import as that
    /// extension, through both the dtype and the array conversion.
    #[test]
    fn run_end_recurses_into_extension_values() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let mut ctx = vortex_session.create_execution_ctx();
        let session = vortex_session.arrow();

        let mut values_field = Field::new("values", DataType::FixedSizeBinary(16), false);
        values_field.try_with_extension_type(ArrowUuid)?;
        let field = Field::new(
            "id",
            DataType::RunEndEncoded(
                Arc::new(Field::new("run_ends", DataType::Int32, false)),
                Arc::new(values_field),
            ),
            false,
        );

        let dtype = session.from_arrow_field(&field)?;
        assert!(
            dtype.as_extension().is::<Uuid>(),
            "expected a Uuid extension dtype, got {dtype}"
        );

        let values = FixedSizeBinaryArray::try_from_iter(
            [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
        )?;
        let run_array: ArrowArrayRef = Arc::new(RunArray::<Int32Type>::try_new(
            &Int32Array::from(vec![2i32, 5]),
            &values,
        )?);

        let vortex_array = session.from_arrow_array(run_array, &field)?;
        assert_eq!(vortex_array.len(), 5);
        // The array conversion must agree with the dtype conversion.
        assert_eq!(vortex_array.dtype(), &dtype);

        // And the values must round-trip back out through the export plugin.
        let exported = session.execute_arrow(vortex_array, Some(&field), &mut ctx)?;
        assert_eq!(exported.len(), 5);
        let ree = exported
            .as_any()
            .downcast_ref::<RunArray<Int32Type>>()
            .ok_or_else(|| {
                vortex_err!(
                    "expected an Int32 run-end array, got {}",
                    exported.data_type()
                )
            })?;
        let values = ree.values().as_fixed_size_binary();
        assert_eq!(values.value(0), b"0123456789abcdef");
        assert_eq!(values.value(1), b"fedcba9876543210");
        Ok(())
    }

    /// An Arrow dictionary array cannot carry extension metadata on the values themselves, but
    /// fields nested inside the values data type can. Both the dtype and the array conversion
    /// must dispatch those through their importer, and must agree with each other.
    #[test]
    fn dictionary_recurses_into_nested_extension_values() -> VortexResult<()> {
        let vortex_session = uuid_session();
        let session = vortex_session.arrow();

        let mut elem = Field::new("item", DataType::FixedSizeBinary(16), false);
        elem.try_with_extension_type(ArrowUuid)?;

        let uuids = FixedSizeBinaryArray::try_from_iter(
            [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
        )?;
        let values = ArrowListArray::try_new(
            Arc::new(elem),
            OffsetBuffer::new(vec![0, 1, 2].into()),
            Arc::new(uuids),
            None,
        )?;
        let dict: ArrowArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(vec![0, 1, 0]),
            Arc::new(values),
        )?);
        let field = Field::new("ids", dict.data_type().clone(), false);

        let dtype = session.from_arrow_field(&field)?;
        let DType::List(elem_dt, _) = &dtype else {
            panic!("expected a List dtype, got {dtype}");
        };
        assert!(
            elem_dt.as_extension().is::<Uuid>(),
            "expected a Uuid extension element, got {elem_dt}"
        );

        let array = session.from_arrow_array(dict, &field)?;
        assert_eq!(array.len(), 3);
        // The array conversion must agree with the dtype conversion.
        assert_eq!(array.dtype(), &dtype);

        // Arrow's dictionary values data type does carry the nested element field, so the
        // extension survives the export as well.
        let mut ctx = vortex_session.create_execution_ctx();
        let exported = session.execute_arrow(array, Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());
        let values = exported.as_any_dictionary().values().as_list::<i32>();
        let uuids = values.values().as_fixed_size_binary();
        assert_eq!(uuids.value(0), b"0123456789abcdef");
        assert_eq!(uuids.value(1), b"fedcba9876543210");
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
        let by_nullability = session.from_arrow_array(arrow_struct, Nullability::from(nullable))?;
        assert_eq!(by_nullability.dtype(), array.dtype());
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

        let vortex = session.from_arrow_array(Arc::new(arrow), &field)?;
        let DType::Map(map_dtype, Nullability::NonNullable) = vortex.dtype() else {
            panic!("expected map dtype, got {}", vortex.dtype());
        };
        assert!(map_dtype.key_dtype().is_extension());
        assert!(map_dtype.value_dtype().is_extension());

        let exported = session.execute_arrow(vortex, Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());
        assert_eq!(exported.as_map().value_offsets(), &[0, 2]);

        Ok(())
    }
}
