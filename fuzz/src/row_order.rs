// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::sync::Arc;

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_row::RowEncoder;
use vortex_row::RowSortField;

use crate::SESSION;
use crate::error::Backtrace;
use crate::error::VortexFuzzError;
use crate::error::VortexFuzzResult;

const MAX_COLUMNS: usize = 4;
const MAX_ROWS_PER_SIDE: usize = 32;
const MAX_NESTING_DEPTH: u8 = 2;
const MAX_STRUCT_FIELDS: usize = 3;
const MAX_FIXED_SIZE_LIST_LEN: u32 = 3;

#[derive(Debug)]
pub struct FuzzRowOrder {
    left_cols: Vec<ArrayRef>,
    right_cols: Vec<ArrayRef>,
    sort_fields: Vec<RowSortField>,
}

impl<'a> Arbitrary<'a> for FuzzRowOrder {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let column_count = u.int_in_range(1..=MAX_COLUMNS)?;
        let left_len = u.int_in_range(1..=MAX_ROWS_PER_SIDE)?;
        let right_len = u.int_in_range(1..=MAX_ROWS_PER_SIDE)?;

        let mut left_cols = Vec::with_capacity(column_count);
        let mut right_cols = Vec::with_capacity(column_count);
        let mut sort_fields = Vec::with_capacity(column_count);

        for _ in 0..column_count {
            let dtype = random_supported_dtype(u, MAX_NESTING_DEPTH)?;
            left_cols.push(random_array(u, dtype.clone(), left_len)?);
            right_cols.push(random_array(u, dtype, right_len)?);
            sort_fields.push(RowSortField::new(u.arbitrary()?, u.arbitrary()?));
        }

        Ok(Self {
            left_cols,
            right_cols,
            sort_fields,
        })
    }
}

#[expect(clippy::result_large_err)]
pub fn run_row_order_fuzz(fuzz: FuzzRowOrder) -> VortexFuzzResult<bool> {
    run_row_order_fuzz_inner(fuzz)
        .map_err(|err| VortexFuzzError::VortexError(err, Backtrace::capture()))
}

fn run_row_order_fuzz_inner(fuzz: FuzzRowOrder) -> VortexResult<bool> {
    let FuzzRowOrder {
        left_cols,
        right_cols,
        sort_fields,
    } = fuzz;

    let mut ctx = SESSION.create_execution_ctx();
    let encoder = RowEncoder::new(sort_fields.iter().copied());
    let left_rows = collect_row_bytes(&encoder.encode(&left_cols, &mut ctx)?, &mut ctx)?;
    let right_rows = collect_row_bytes(&encoder.encode(&right_cols, &mut ctx)?, &mut ctx)?;

    for (left_idx, left_bytes) in left_rows.iter().enumerate() {
        for (right_idx, right_bytes) in right_rows.iter().enumerate() {
            let array_order = compare_rows(
                &left_cols,
                left_idx,
                &right_cols,
                right_idx,
                &sort_fields,
                &mut ctx,
            )?;
            let row_order = left_bytes.cmp(right_bytes);
            if array_order != row_order {
                vortex_bail!(
                    "row-order mismatch comparing left row {} to right row {}: \
                     array order {:?}, row-byte order {:?}, dtypes {:?}, sort fields {:?}, \
                     left bytes {:?}, right bytes {:?}",
                    left_idx,
                    right_idx,
                    array_order,
                    row_order,
                    left_cols.iter().map(|col| col.dtype()).collect::<Vec<_>>(),
                    sort_fields,
                    left_bytes,
                    right_bytes
                );
            }
        }
    }

    Ok(true)
}

fn collect_row_bytes(
    encoded: &vortex_array::arrays::ListViewArray,
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<Vec<Vec<u8>>> {
    (0..encoded.len())
        .map(|row_idx| {
            let row = encoded.list_elements_at(row_idx)?;
            let row = row.execute::<PrimitiveArray>(ctx)?;
            Ok(row.as_slice::<u8>().to_vec())
        })
        .collect()
}

fn compare_rows(
    left_cols: &[ArrayRef],
    left_idx: usize,
    right_cols: &[ArrayRef],
    right_idx: usize,
    sort_fields: &[RowSortField],
    ctx: &mut vortex_array::ExecutionCtx,
) -> VortexResult<Ordering> {
    for ((left_col, right_col), field) in left_cols.iter().zip(right_cols).zip(sort_fields) {
        let left = left_col.execute_scalar(left_idx, ctx)?;
        let right = right_col.execute_scalar(right_idx, ctx)?;
        match compare_scalar(&left, &right, *field)? {
            Ordering::Equal => {}
            ordering => return Ok(ordering),
        }
    }

    Ok(Ordering::Equal)
}

fn compare_scalar(left: &Scalar, right: &Scalar, field: RowSortField) -> VortexResult<Ordering> {
    if !left.dtype().eq_ignore_nullability(right.dtype()) {
        vortex_bail!(
            "cannot compare row scalars with different dtypes: {} vs {}",
            left.dtype(),
            right.dtype()
        );
    }

    compare_scalar_values(left.dtype(), left.value(), right.value(), field)
}

fn compare_scalar_values(
    dtype: &DType,
    left: Option<&ScalarValue>,
    right: Option<&ScalarValue>,
    field: RowSortField,
) -> VortexResult<Ordering> {
    let (Some(left), Some(right)) = (left, right) else {
        return Ok(compare_nulls(left.is_none(), right.is_none(), field));
    };

    match dtype {
        DType::Null => Ok(Ordering::Equal),
        DType::Struct(fields, _) => compare_struct_values(fields, left, right, field),
        DType::FixedSizeList(element_dtype, list_size, _) => {
            compare_fixed_size_list_values(element_dtype, *list_size, left, right, field)
        }
        DType::List(..) | DType::Variant(_) | DType::Union(_) | DType::Extension(_) => {
            vortex_bail!("row-order fuzzer generated unsupported dtype: {dtype}")
        }
        _ => compare_leaf_values(dtype, left, right, field),
    }
}

fn compare_nulls(left_is_null: bool, right_is_null: bool, field: RowSortField) -> Ordering {
    match (left_is_null, right_is_null) {
        (true, true) | (false, false) => Ordering::Equal,
        (true, false) => {
            if field.nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (false, true) => {
            if field.nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
    }
}

fn compare_struct_values(
    fields: &StructFields,
    left: &ScalarValue,
    right: &ScalarValue,
    field: RowSortField,
) -> VortexResult<Ordering> {
    let (ScalarValue::Tuple(left_fields), ScalarValue::Tuple(right_fields)) = (left, right) else {
        vortex_bail!("struct dtype expected tuple scalar values");
    };
    if left_fields.len() != fields.nfields() || right_fields.len() != fields.nfields() {
        vortex_bail!(
            "struct scalar field count mismatch: expected {}, got {} and {}",
            fields.nfields(),
            left_fields.len(),
            right_fields.len()
        );
    }

    for ((field_dtype, left_value), right_value) in
        fields.fields().zip(left_fields).zip(right_fields)
    {
        match compare_scalar_values(
            &field_dtype,
            left_value.as_ref(),
            right_value.as_ref(),
            field,
        )? {
            Ordering::Equal => {}
            ordering => return Ok(ordering),
        }
    }

    Ok(Ordering::Equal)
}

fn compare_fixed_size_list_values(
    element_dtype: &DType,
    list_size: u32,
    left: &ScalarValue,
    right: &ScalarValue,
    field: RowSortField,
) -> VortexResult<Ordering> {
    let (ScalarValue::Tuple(left_elements), ScalarValue::Tuple(right_elements)) = (left, right)
    else {
        vortex_bail!("fixed-size list dtype expected tuple scalar values");
    };
    let expected_len = list_size as usize;
    if left_elements.len() != expected_len || right_elements.len() != expected_len {
        vortex_bail!(
            "fixed-size list scalar length mismatch: expected {}, got {} and {}",
            expected_len,
            left_elements.len(),
            right_elements.len()
        );
    }

    for (left_value, right_value) in left_elements.iter().zip(right_elements) {
        match compare_scalar_values(
            element_dtype,
            left_value.as_ref(),
            right_value.as_ref(),
            field,
        )? {
            Ordering::Equal => {}
            ordering => return Ok(ordering),
        }
    }

    Ok(Ordering::Equal)
}

fn compare_leaf_values(
    dtype: &DType,
    left: &ScalarValue,
    right: &ScalarValue,
    field: RowSortField,
) -> VortexResult<Ordering> {
    let left = Scalar::try_new(dtype.clone(), Some(left.clone()))?;
    let right = Scalar::try_new(dtype.clone(), Some(right.clone()))?;
    let ordering = left.partial_cmp(&right).ok_or_else(|| {
        vortex_err!(
            "scalar comparison returned None for matching row-order dtype {}",
            dtype
        )
    })?;

    Ok(if field.descending {
        ordering.reverse()
    } else {
        ordering
    })
}

fn random_array(u: &mut Unstructured<'_>, dtype: DType, len: usize) -> Result<ArrayRef> {
    Ok(ArbitraryArray::arbitrary_with_config(
        u,
        &ArbitraryArrayConfig {
            dtype: Some(dtype),
            len: len..=len,
        },
    )?
    .0)
}

fn random_supported_dtype(u: &mut Unstructured<'_>, depth: u8) -> Result<DType> {
    let max_kind = if depth == 0 { 5 } else { 7 };
    Ok(match u.int_in_range(0..=max_kind)? {
        0 => DType::Null,
        1 => DType::Bool(u.arbitrary()?),
        2 => DType::Primitive(PType::arbitrary(u)?, u.arbitrary()?),
        3 => DType::Decimal(random_supported_decimal_dtype(u)?, u.arbitrary()?),
        4 => DType::Utf8(u.arbitrary()?),
        5 => DType::Binary(u.arbitrary()?),
        6 => DType::Struct(
            random_supported_struct_fields(u, depth - 1)?,
            u.arbitrary()?,
        ),
        7 => DType::FixedSizeList(
            Arc::new(random_supported_dtype(u, depth - 1)?),
            u.int_in_range(0..=MAX_FIXED_SIZE_LIST_LEN)?,
            u.arbitrary()?,
        ),
        _ => unreachable!("dtype kind range is bounded"),
    })
}

fn random_supported_decimal_dtype(u: &mut Unstructured<'_>) -> Result<DecimalDType> {
    let precision = u.int_in_range(1..=38)?;
    let scale = u.int_in_range(-18..=precision as i8)?;
    Ok(DecimalDType::new(precision, scale))
}

fn random_supported_struct_fields(u: &mut Unstructured<'_>, depth: u8) -> Result<StructFields> {
    let field_count = u.int_in_range(0..=MAX_STRUCT_FIELDS)?;
    let names = (0..field_count)
        .map(|idx| FieldName::from(format!("f{idx}")))
        .collect::<Vec<_>>();
    let dtypes = (0..field_count)
        .map(|_| random_supported_dtype(u, depth))
        .collect::<Result<Vec<_>>>()?;

    Ok(StructFields::new(FieldNames::from(names), dtypes))
}
