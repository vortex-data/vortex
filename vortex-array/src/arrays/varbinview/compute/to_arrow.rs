use arrow_array::ArrayRef;
use arrow_cast::cast;
use arrow_schema::DataType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::{varbinview_as_arrow, VarBinViewArray, VarBinViewEncoding};
use crate::compute::ToArrowFn;

impl ToArrowFn<VarBinViewArray> for VarBinViewEncoding {
    fn to_arrow(
        &self,
        array: &VarBinViewArray,
        data_type: &DataType,
    ) -> VortexResult<Option<ArrayRef>> {
        match data_type {
            DataType::Binary
            | DataType::FixedSizeBinary(_)
            | DataType::LargeBinary
            | DataType::Utf8
            | DataType::LargeUtf8 => {
                // TODO(ngates): we should support converting VarBinView into these Arrow arrays.
            }
            DataType::BinaryView | DataType::Utf8View => {
                // These are both supported with a zero-copy cast, see below
            }
            _ => {
                // Everything else is unsupported
                vortex_bail!("Unsupported data type: {data_type}")
            }
        }

        let arrow_arr = varbinview_as_arrow(array);
        Ok(Some(if arrow_arr.data_type() != data_type {
            cast(arrow_arr.as_ref(), data_type)?
        } else {
            arrow_arr
        }))
    }
}
