use std::sync::Arc;

use arrow_array::{ArrayRef, GenericBinaryArray, GenericStringArray, OffsetSizeTrait};
use arrow_schema::DataType;
use vortex_dtype::{DType, NativePType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::compute::{ToArrowFn, try_cast};
use crate::{Array, ToCanonical};

impl ToArrowFn<&VarBinArray> for VarBinEncoding {
    fn preferred_arrow_data_type(&self, array: &VarBinArray) -> VortexResult<Option<DataType>> {
        let offsets_ptype = PType::try_from(array.offsets().dtype())?;
        Ok(Some(match array.dtype() {
            DType::Utf8(_) => match offsets_ptype {
                PType::I64 | PType::U64 => DataType::LargeUtf8,
                _ => DataType::Utf8,
            },
            DType::Binary(_) => match offsets_ptype {
                PType::I64 | PType::U64 => DataType::LargeBinary,
                _ => DataType::Binary,
            },
            _ => vortex_bail!("Unsupported DType"),
        }))
    }

    fn to_arrow(
        &self,
        array: &VarBinArray,
        data_type: &DataType,
    ) -> VortexResult<Option<ArrayRef>> {
        let array_ref = match data_type {
            DataType::BinaryView | DataType::FixedSizeBinary(_) | DataType::Utf8View => {
                // TODO(ngates): we should support converting VarBin into these Arrow arrays.
                return Ok(None);
            }
            DataType::Binary | DataType::Utf8 => {
                // These are both supported with a zero-copy cast, see below
                varbin_to_arrow::<i32>(array)
            }
            DataType::LargeBinary | DataType::LargeUtf8 => {
                // These are both supported with a zero-copy cast, see below
                varbin_to_arrow::<i64>(array)
            }
            _ => {
                // Everything else is unsupported
                vortex_bail!("Unsupported data type: {data_type}")
            }
        }?;

        Ok(Some(if array_ref.data_type() != data_type {
            arrow_cast::cast(array_ref.as_ref(), data_type)?
        } else {
            array_ref
        }))
    }
}

/// Convert the array to Arrow variable length binary array type.
pub(crate) fn varbin_to_arrow<O: NativePType + OffsetSizeTrait>(
    varbin_array: &VarBinArray,
) -> VortexResult<ArrayRef> {
    let offsets = try_cast(
        varbin_array.offsets(),
        &DType::Primitive(O::PTYPE, Nullability::NonNullable),
    )?
    .to_primitive()
    .map_err(|err| err.with_context("Failed to canonicalize offsets"))?;

    let nulls = varbin_array.validity_mask()?.to_null_buffer();
    let data = varbin_array.bytes().clone();

    // Switch on DType.
    Ok(match varbin_array.dtype() {
        DType::Binary(_) => Arc::new(unsafe {
            GenericBinaryArray::new_unchecked(
                offsets.buffer::<O>().into_arrow_offset_buffer(),
                data.into_arrow_buffer(),
                nulls,
            )
        }),
        DType::Utf8(_) => Arc::new(unsafe {
            GenericStringArray::new_unchecked(
                offsets.buffer::<O>().into_arrow_offset_buffer(),
                data.into_arrow_buffer(),
                nulls,
            )
        }),
        _ => vortex_bail!(
            "expected utf8 or binary instead of {}",
            varbin_array.dtype()
        ),
    })
}
