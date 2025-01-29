use arrow_array::{BinaryArray, StringArray};
use arrow_ord::cmp;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::array::{VarBinArray, VarBinEncoding};
use crate::arrow::{from_arrow_array_with_len, Datum};
use crate::compute::{CompareFn, Operator};
use crate::{ArrayData, IntoArrayData};

// This implementation exists so we can have custom translation of RHS to arrow that's not the same as IntoCanonical
impl CompareFn<VarBinArray> for VarBinEncoding {
    fn compare(
        &self,
        lhs: &VarBinArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        if let Some(rhs_const) = rhs.as_constant() {
            let nullable = lhs.dtype().is_nullable() || rhs_const.dtype().is_nullable();
            let len = lhs.len();

            let lhs = Datum::try_new(lhs.clone().into_array())?;

            // TODO(robert): Handle LargeString/Binary arrays
            let arrow_rhs: &dyn arrow_array::Datum = match rhs_const.dtype() {
                DType::Utf8(_) => &rhs_const
                    .as_utf8()
                    .value()
                    .map(StringArray::new_scalar)
                    .unwrap_or_else(|| arrow_array::Scalar::new(StringArray::new_null(1))),
                DType::Binary(_) => &rhs_const
                    .as_binary()
                    .value()
                    .map(BinaryArray::new_scalar)
                    .unwrap_or_else(|| arrow_array::Scalar::new(BinaryArray::new_null(1))),
                _ => vortex_bail!(
                    "VarBin array RHS can only be Utf8 or Binary, given {}",
                    rhs_const.dtype()
                ),
            };

            let array = match operator {
                Operator::Eq => cmp::eq(&lhs, arrow_rhs),
                Operator::NotEq => cmp::neq(&lhs, arrow_rhs),
                Operator::Gt => cmp::gt(&lhs, arrow_rhs),
                Operator::Gte => cmp::gt_eq(&lhs, arrow_rhs),
                Operator::Lt => cmp::lt(&lhs, arrow_rhs),
                Operator::Lte => cmp::lt_eq(&lhs, arrow_rhs),
            }
            .map_err(|err| vortex_err!("Failed to compare VarBin array: {}", err))?;

            Ok(Some(from_arrow_array_with_len(&array, len, nullable)?))
        } else {
            Ok(None)
        }
    }
}
