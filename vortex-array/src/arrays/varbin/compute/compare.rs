use arrow_array::{BinaryArray, StringArray};
use arrow_buffer::BooleanBuffer;
use arrow_ord::cmp;
use itertools::Itertools;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::{vortex_bail, vortex_err, VortexExpect as _, VortexResult};

use crate::arrays::{BoolArray, PrimitiveArray, VarBinArray, VarBinEncoding};
use crate::arrow::{from_arrow_array_with_len, Datum};
use crate::compute::{compare_lengths_to_empty, CompareFn, Operator};
use crate::variants::PrimitiveArrayTrait as _;
use crate::{Array, IntoArray, IntoCanonical};

// This implementation exists so we can have custom translation of RHS to arrow that's not the same as IntoCanonical
impl CompareFn<VarBinArray> for VarBinEncoding {
    fn compare(
        &self,
        lhs: &VarBinArray,
        rhs: &Array,
        operator: Operator,
    ) -> VortexResult<Option<Array>> {
        if let Some(rhs_const) = rhs.as_constant() {
            let nullable = lhs.dtype().is_nullable() || rhs_const.dtype().is_nullable();
            let len = lhs.len();

            let rhs_is_empty = match rhs_const.dtype() {
                DType::Binary(_) => rhs_const
                    .as_binary()
                    .is_empty()
                    .vortex_expect("RHS should not be null"),
                DType::Utf8(_) => rhs_const
                    .as_utf8()
                    .is_empty()
                    .vortex_expect("RHS should not be null"),
                _ => vortex_bail!("VarBinArray can only have type of Binary or Utf8"),
            };

            if rhs_is_empty {
                let buffer = match operator {
                    // Every possible value is gte ""
                    Operator::Gte => BooleanBuffer::new_set(len),
                    // No value is lt ""
                    Operator::Lt => BooleanBuffer::new_unset(len),
                    _ => {
                        let lhs_offsets = lhs.offsets().into_canonical()?.into_primitive()?;
                        match_each_native_ptype!(lhs_offsets.ptype(), |$P| {
                            compare_offsets_to_empty::<$P>(lhs_offsets, operator)
                        })
                    }
                };

                return Ok(Some(
                    BoolArray::try_new(buffer, lhs.validity())?.into_array(),
                ));
            }

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

fn compare_offsets_to_empty<P: NativePType>(
    offsets: PrimitiveArray,
    operator: Operator,
) -> BooleanBuffer {
    let lengths_iter = offsets
        .as_slice::<P>()
        .iter()
        .tuple_windows()
        .map(|(&s, &e)| e - s);
    compare_lengths_to_empty(lengths_iter, operator)
}
