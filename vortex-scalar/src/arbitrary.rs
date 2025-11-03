// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arbitrary scalar value generation.
//!
//! This module provides functions to generate arbitrary scalar values of various data types.
//! It is used by the fuzzer to test the correctness of the scalar value implementation.

use std::iter;
use std::sync::Arc;

use arbitrary::{Result, Unstructured};
use vortex_buffer::{BufferString, ByteBuffer};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, DecimalDType, NativeDecimalType, PType};

use crate::{
    DecimalValue, InnerScalarValue, PValue, Scalar, ScalarValue, match_each_decimal_value_type,
};

/// Generate an arbitrary scalar value of the given data type.
pub fn random_scalar(u: &mut Unstructured, dtype: &DType) -> Result<Scalar> {
    Ok(Scalar::new(dtype.clone(), random_scalar_value(u, dtype)?))
}

fn random_scalar_value(u: &mut Unstructured, dtype: &DType) -> Result<ScalarValue> {
    match dtype {
        DType::Null => Ok(ScalarValue(InnerScalarValue::Null)),
        DType::Bool(_) => Ok(ScalarValue(InnerScalarValue::Bool(u.arbitrary()?))),
        DType::Primitive(p, _) => Ok(ScalarValue(InnerScalarValue::Primitive(random_pvalue(
            u, p,
        )?))),
        DType::Decimal(decimal_type, _) => random_decimal(u, decimal_type),
        DType::Utf8(_) => Ok(ScalarValue(InnerScalarValue::BufferString(Arc::new(
            BufferString::from(u.arbitrary::<String>()?),
        )))),
        DType::Binary(_) => Ok(ScalarValue(InnerScalarValue::Buffer(Arc::new(
            ByteBuffer::from(u.arbitrary::<Vec<u8>>()?),
        )))),
        DType::Struct(sdt, _) => Ok(ScalarValue(InnerScalarValue::List(
            sdt.fields()
                .map(|d| random_scalar_value(u, &d))
                .collect::<Result<Vec<_>>>()?
                .into(),
        ))),
        DType::List(edt, _) => Ok(ScalarValue(InnerScalarValue::List(
            iter::from_fn(|| {
                // Creates `Some(_)` with 1/4 probability.
                u.arbitrary()
                    .unwrap_or(false)
                    .then(|| random_scalar_value(u, edt))
            })
            .collect::<Result<Vec<_>>>()?
            .into(),
        ))),
        DType::FixedSizeList(edt, size, _) => Ok(ScalarValue(InnerScalarValue::List(
            (0..*size)
                .map(|_| random_scalar_value(u, edt))
                .collect::<Result<Vec<_>>>()?
                .into(),
        ))),
        DType::Extension(..) => {
            unreachable!("Can't yet generate arbitrary scalars for ext dtype")
        }
    }
}

fn random_pvalue(u: &mut Unstructured, ptype: &PType) -> Result<PValue> {
    Ok(match ptype {
        PType::U8 => PValue::U8(u.arbitrary()?),
        PType::U16 => PValue::U16(u.arbitrary()?),
        PType::U32 => PValue::U32(u.arbitrary()?),
        PType::U64 => PValue::U64(u.arbitrary()?),
        PType::I8 => PValue::I8(u.arbitrary()?),
        PType::I16 => PValue::I16(u.arbitrary()?),
        PType::I32 => PValue::I32(u.arbitrary()?),
        PType::I64 => PValue::I64(u.arbitrary()?),
        PType::F16 => PValue::F16(f16::from_bits(u.arbitrary()?)),
        PType::F32 => PValue::F32(u.arbitrary()?),
        PType::F64 => PValue::F64(u.arbitrary()?),
    })
}

/// Generate an arbitrary decimal scalar confined to the given bounds of precision and scale.
pub fn random_decimal(u: &mut Unstructured, decimal_type: &DecimalDType) -> Result<ScalarValue> {
    let precision = decimal_type.precision();
    let value = match_each_decimal_value_type!(
        DecimalType::smallest_decimal_value_type(decimal_type),
        |D| {
            DecimalValue::from(u.int_in_range(
                D::MIN_BY_PRECISION[precision as usize]..=D::MAX_BY_PRECISION[precision as usize],
            )?)
        }
    );

    Ok(ScalarValue(InnerScalarValue::Decimal(value)))
}
