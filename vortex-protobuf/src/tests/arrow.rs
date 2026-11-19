// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow export and import of protobuf arrays.

use arrow_schema::DataType;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_arrow::ext_metadata::decode_arrow_ext_metadata;
use vortex_error::VortexResult;

use super::SESSION;
use super::all_types_dtype;
use crate::Protobuf;
use crate::ProtobufDecoder;

#[rstest]
fn test_arrow_round_trip(
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = all_types_dtype(nullability)?;
    let messages = super::codec::sample_messages(dtype.metadata().message_descriptor())?;
    let mut decoder = ProtobufDecoder::new(&dtype);
    for message in &messages {
        decoder.append_message(message)?;
    }
    if nullability.is_nullable() {
        decoder.append_null()?;
    }
    let array = decoder.finish()?;

    let field = SESSION.arrow().to_arrow_field("message", array.dtype())?;
    assert_eq!(
        field
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .map(String::as_str),
        Some(Protobuf::ID)
    );
    assert_eq!(
        decode_arrow_ext_metadata(&field)?,
        dtype.serialize_metadata()?
    );
    assert!(matches!(field.data_type(), DataType::Struct(_)));
    assert_eq!(field.is_nullable(), nullability.is_nullable());
    assert_eq!(
        SESSION.arrow().from_arrow_field(&field)?,
        DType::Extension(dtype.erased())
    );

    let exported = array.clone().execute_arrow(Some(&field), &mut ctx)?;
    let imported = SESSION.arrow().from_arrow_array(exported, &field)?;
    assert_eq!(imported.dtype(), array.dtype());
    assert_arrays_eq!(imported, array, &mut ctx);
    Ok(())
}
