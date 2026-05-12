// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arbitrary scalar value generation.
//!
//! This module provides functions to generate arbitrary scalar values of various data types.
//! It is used by the fuzzer to test the correctness of the scalar value implementation.

use std::iter;

use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// Generates an arbitrary [`Scalar`] of the given [`DType`].
///
/// # Errors
///
/// Returns an error if the underlying arbitrary generation fails.
pub fn random_scalar(u: &mut Unstructured, dtype: &DType) -> Result<Scalar> {
    // For nullable types, return null ~25% of the time. This is just to make sure we don't generate
    // too few nulls.
    if dtype.is_nullable() && u.ratio(1, 4)? {
        return Ok(Scalar::null(dtype.clone()));
    }

    Ok(match dtype {
        DType::Null => Scalar::null(dtype.clone()),
        DType::Bool(_) => Scalar::try_new(dtype.clone(), Some(ScalarValue::Bool(u.arbitrary()?)))
            .vortex_expect("unable to construct random `Scalar`_"),
        DType::Primitive(p, _) => Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Primitive(random_pvalue(u, p)?)),
        )
        .vortex_expect("unable to construct random `Scalar`_"),
        DType::Decimal(decimal_type, _) => {
            Scalar::try_new(dtype.clone(), Some(random_decimal(u, decimal_type)?))
                .vortex_expect("unable to construct random `Scalar`_")
        }
        DType::Utf8(_) => Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Utf8(BufferString::from(
                u.arbitrary::<String>()?,
            ))),
        )
        .vortex_expect("unable to construct random `Scalar`_"),
        DType::Binary(_) => Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Binary(ByteBuffer::from(
                u.arbitrary::<Vec<u8>>()?,
            ))),
        )
        .vortex_expect("unable to construct random `Scalar`_"),
        DType::Struct(sdt, _) => Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Tuple(
                sdt.fields()
                    .map(|d| random_scalar(u, &d).map(|s| s.into_value()))
                    .collect::<Result<Vec<_>>>()?,
            )),
        )
        .vortex_expect("unable to construct random `Scalar`_"),
        DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
        DType::List(edt, _) => Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Tuple(
                iter::from_fn(|| {
                    // Generate elements with 1/4 probability.
                    u.arbitrary()
                        .unwrap_or(false)
                        .then(|| random_scalar(u, edt).map(|s| s.into_value()))
                })
                .collect::<Result<Vec<_>>>()?,
            )),
        )
        .vortex_expect("unable to construct random `Scalar`_"),
        DType::FixedSizeList(edt, size, _) => Scalar::try_new(
            dtype.clone(),
            Some(ScalarValue::Tuple(
                (0..*size)
                    .map(|_| random_scalar(u, edt).map(|s| s.into_value()))
                    .collect::<Result<Vec<_>>>()?,
            )),
        )
        .vortex_expect("unable to construct random `Scalar`_"),
        DType::Extension(..) => {
            unreachable!("Can't yet generate arbitrary scalars for ext dtype")
        }
        DType::Variant(_) => todo!(),
    })
}

/// Generates an arbitrary [`PValue`] for the given [`PType`].
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

/// Generates an arbitrary decimal scalar confined to the given bounds of precision and scale.
///
/// # Errors
///
/// Returns an error if the underlying arbitrary generation fails.
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

    Ok(ScalarValue::Decimal(value))
}
