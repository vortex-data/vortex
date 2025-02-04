use arrow_array::{BinaryArray, StringArray};
use arrow_buffer::BooleanBuffer;
use arrow_ord::cmp;
use itertools::Itertools;
use num_traits::Zero;
use vortex_dtype::{match_each_native_ptype, DType, NativePType};
use vortex_error::{vortex_bail, vortex_err, VortexResult};

use crate::array::{BoolArray, PrimitiveArray, VarBinArray, VarBinEncoding};
use crate::arrow::{from_arrow_array_with_len, Datum};
use crate::compute::{CompareFn, Operator};
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
                DType::Utf8(_) => {
                    let v = rhs_const.as_utf8().value();
                    v.is_some_and(|v| v.is_empty())
                }
                DType::Binary(_) => {
                    let v = rhs_const.as_binary().value();
                    v.is_some_and(|v| v.is_empty())
                }
                _ => vortex_bail!(
                    "VarBin array RHS can only be Utf8 or Binary, given {}",
                    rhs_const.dtype()
                ),
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
                            compare_to_empty::<$P>(lhs_offsets, operator)
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

/// All comparison can be expressed in terms of equality. "" is the absolute min of possible value.
const fn cmp_to_empty<T: PartialEq + PartialOrd + Zero>(op: Operator) -> fn(T) -> bool {
    match op {
        Operator::Eq | Operator::Lte => |v| v == T::zero(),
        Operator::NotEq | Operator::Gt => |v| v != T::zero(),
        Operator::Gte => |_| true,
        Operator::Lt => |_| false,
    }
}

fn compare_to_empty<T>(array: PrimitiveArray, op: Operator) -> BooleanBuffer
where
    T: NativePType,
{
    let cmp_fn = cmp_to_empty::<T>(op);
    let slice = array.as_slice::<T>();
    slice
        .iter()
        .tuple_windows()
        .map(|(&s, &e)| cmp_fn(e - s))
        .collect::<BooleanBuffer>()
}
