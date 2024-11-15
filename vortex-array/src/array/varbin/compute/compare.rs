use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::ByteArrayType;
use arrow_array::{Array as ArrowArray, Datum, GenericByteArray};
use arrow_ord::cmp;
use arrow_schema::DataType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::varbin::arrow::{varbin_datum, varbin_to_arrow};
use crate::array::{ConstantArray, VarBinArray};
use crate::arrow::FromArrowArray;
use crate::compute::{MaybeCompareFn, Operator};
use crate::ArrayData;

impl MaybeCompareFn for VarBinArray {
    fn maybe_compare(
        &self,
        other: &ArrayData,
        operator: Operator,
    ) -> Option<VortexResult<ArrayData>> {
        if let Ok(rhs_const) = ConstantArray::try_from(other) {
            Some(compare_constant(self, &rhs_const, operator))
        } else {
            None
        }
    }
}

fn compare_constant(
    lhs: &VarBinArray,
    rhs: &ConstantArray,
    operator: Operator,
) -> VortexResult<ArrayData> {
    // Compare using the arrow kernels directly.
    let arrow_lhs = varbin_to_arrow(lhs)?;
    let constant = varbin_datum(rhs.owned_scalar())?;

    match arrow_lhs.data_type() {
        DataType::Binary => {
            compare_constant_arrow(arrow_lhs.as_binary::<i32>(), constant, operator)
        }
        DataType::LargeBinary => {
            compare_constant_arrow(arrow_lhs.as_binary::<i64>(), constant, operator)
        }
        DataType::Utf8 => compare_constant_arrow(arrow_lhs.as_string::<i32>(), constant, operator),
        DataType::LargeUtf8 => {
            compare_constant_arrow(arrow_lhs.as_string::<i64>(), constant, operator)
        }
        _ => {
            vortex_bail!("Cannot compare VarBinArray with non-binary type");
        }
    }
}

fn compare_constant_arrow<T: ByteArrayType>(
    lhs: &GenericByteArray<T>,
    rhs: Arc<dyn Datum>,
    operator: Operator,
) -> VortexResult<ArrayData> {
    let rhs = rhs.as_ref();
    let array = match operator {
        Operator::Eq => cmp::eq(lhs, rhs)?,
        Operator::NotEq => cmp::neq(lhs, rhs)?,
        Operator::Gt => cmp::gt(lhs, rhs)?,
        Operator::Gte => cmp::gt_eq(lhs, rhs)?,
        Operator::Lt => cmp::lt(lhs, rhs)?,
        Operator::Lte => cmp::lt_eq(lhs, rhs)?,
    };
    Ok(ArrayData::from_arrow(&array, true))
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};
    use vortex_scalar::Scalar;

    use super::*;
    use crate::array::builder::VarBinBuilder;
    use crate::IntoArrayVariant;

    #[test]
    fn basic_test() {
        let mut builder = VarBinBuilder::<i32>::new();
        for v in [
            b"one".as_slice(),
            b"two".as_slice(),
            b"three".as_slice(),
            b"four".as_slice(),
            b"five".as_slice(),
            b"six".as_slice(),
        ] {
            builder.push_value(v);
        }

        let arr = builder.finish(DType::Utf8(Nullability::Nullable));

        let s = Scalar::utf8("seven".to_string(), Nullability::Nullable);

        let constant_array = ConstantArray::new(s, arr.len());

        let r = compare_constant(&arr, &constant_array, Operator::Eq)
            .unwrap()
            .into_bool()
            .unwrap();

        for v in r.boolean_buffer().iter() {
            assert!(!v);
        }
    }
}
