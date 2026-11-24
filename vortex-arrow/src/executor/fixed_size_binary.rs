// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::FixedSizeBinaryArray as ArrowFixedSizeBinaryArray;
use arrow_buffer::NullBuffer;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::FixedSizeBinaryArray;
use vortex_array::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::null_buffer::to_null_buffer;

/// Exports a fixed-size binary array as an Arrow `FixedSizeBinary(byte_width)` array, sharing its
/// values buffer.
pub(super) fn to_arrow_fixed_size_binary(
    array: ArrayRef,
    byte_width: i32,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    vortex_ensure!(
        array.dtype().is_fixed_size_binary(),
        "Cannot convert Vortex array with dtype {} to an Arrow FixedSizeBinary({byte_width}) array",
        array.dtype(),
    );
    let array = array.execute::<FixedSizeBinaryArray>(ctx)?;
    vortex_ensure_eq!(
        i64::from(array.byte_width()),
        i64::from(byte_width),
        "Cannot export {} as Arrow FixedSizeBinary({byte_width})",
        array.dtype(),
    );

    let len = array.len();
    let nulls = to_null_buffer(array.validity()?.execute_mask(len, ctx)?);
    // Arrow takes the length of a zero-width array from its null buffer.
    let nulls = match (byte_width, nulls) {
        (0, None) => Some(NullBuffer::new_valid(len)),
        (_, nulls) => nulls,
    };
    let values = array.values().into_arrow_buffer();

    Ok(Arc::new(
        ArrowFixedSizeBinaryArray::try_new(byte_width, values, nulls)
            .map_err(|err| vortex_err!("Invalid Arrow FixedSizeBinary export: {err}"))?,
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use arrow_array::Array as ArrowArray;
    use arrow_array::FixedSizeBinaryArray as ArrowFixedSizeBinaryArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FixedSizeBinaryArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ExecuteArrowExt;
    use crate::convert::from_arrow_dyn;
    use crate::dtype::from_arrow_data_type;
    use crate::dtype::to_data_type_naive;
    use crate::scalar::ToArrowDatum;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::non_nullable(Nullability::NonNullable)]
    #[case::nullable(Nullability::Nullable)]
    fn dtype_round_trip(#[case] nullability: Nullability) -> VortexResult<()> {
        let dtype = DType::FixedSizeBinary(16, nullability);
        let data_type = to_data_type_naive(&dtype)?;
        assert_eq!(data_type, DataType::FixedSizeBinary(16));
        assert_eq!(from_arrow_data_type(&data_type, nullability)?, dtype);
        Ok(())
    }

    #[test]
    fn import_shares_values_and_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arrow = ArrowFixedSizeBinaryArray::try_from_sparse_iter_with_size(
            [Some(b"abc"), None, Some(b"xyz")].into_iter(),
            3,
        )?;
        let imported = from_arrow_dyn(&arrow, true)?;
        let expected =
            FixedSizeBinaryArray::from_option_iter([Some(b"abc"), None, Some(b"xyz")], 3);
        assert_arrays_eq!(imported, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn import_sliced_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arrow =
            ArrowFixedSizeBinaryArray::try_from_iter([b"aa", b"bb", b"cc", b"dd"].into_iter())?
                .slice(1, 2);
        let imported = from_arrow_dyn(&arrow, false)?;
        let expected = FixedSizeBinaryArray::from_iter([b"bb", b"cc"], 2);
        assert_arrays_eq!(imported, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::non_nullable(FixedSizeBinaryArray::from_iter([b"ab", b"cd", b"ef"], 2))]
    #[case::nullable(FixedSizeBinaryArray::from_option_iter([Some(b"ab"), None, Some(b"ef")], 2))]
    #[case::zero_width(FixedSizeBinaryArray::from_iter([b""; 4], 0))]
    #[case::zero_width_nullable(FixedSizeBinaryArray::from_option_iter(
        [Some(b""), None, Some(b"")],
        0,
    ))]
    fn export_round_trip(#[case] array: FixedSizeBinaryArray) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let nullable = array.dtype().is_nullable();
        let arrow = array.clone().into_array().execute_arrow(None, &mut ctx)?;
        assert_eq!(arrow.len(), array.len());
        assert!(matches!(arrow.data_type(), DataType::FixedSizeBinary(_)));

        let round_trip = from_arrow_dyn(arrow.as_ref(), nullable)?;
        assert_arrays_eq!(round_trip, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn export_to_another_width_fails() {
        let mut ctx = SESSION.create_execution_ctx();
        let field = Field::new("value", DataType::FixedSizeBinary(3), false);
        let result = FixedSizeBinaryArray::from_iter([b"ab"], 2)
            .into_array()
            .execute_arrow(Some(&field), &mut ctx);
        assert!(result.is_err());
    }

    #[rstest]
    #[case::value(Scalar::fixed_size_binary(b"abc".to_vec(), Nullability::NonNullable))]
    #[case::null(Scalar::null(DType::FixedSizeBinary(3, Nullability::Nullable)))]
    fn scalar_datum(#[case] scalar: Scalar) -> VortexResult<()> {
        let datum = scalar.to_arrow_datum()?;
        let (array, is_scalar) = datum.get();
        assert!(is_scalar);
        assert_eq!(array.data_type(), &DataType::FixedSizeBinary(3));
        assert_eq!(array.is_null(0), scalar.is_null());
        Ok(())
    }
}
