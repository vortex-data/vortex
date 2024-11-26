use std::iter;

use arbitrary::{Result, Unstructured};
use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, PType};

use crate::{InnerScalarValue, PValue, Scalar, ScalarValue};

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
        DType::Utf8(_) => Ok(ScalarValue(InnerScalarValue::BufferString(
            BufferString::from(u.arbitrary::<String>()?),
        ))),
        DType::Binary(_) => Ok(ScalarValue(InnerScalarValue::Buffer(Buffer::from(
            u.arbitrary::<Vec<u8>>()?,
        )))),
        DType::Struct(sdt, _) => Ok(ScalarValue(InnerScalarValue::List(
            sdt.dtypes()
                .iter()
                .map(|d| random_scalar_value(u, d).map(|x| x.0))
                .collect::<Result<Vec<_>>>()?
                .into(),
        ))),
        DType::List(edt, _) => Ok(ScalarValue(InnerScalarValue::List(
            iter::from_fn(|| {
                u.arbitrary()
                    .unwrap_or(false)
                    .then(|| random_scalar_value(u, edt).map(|x| x.0))
            })
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
