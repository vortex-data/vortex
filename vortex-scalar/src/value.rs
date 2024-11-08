use std::fmt::{Display, Write};
use std::sync::Arc;

use half::f16;
use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};

use crate::pvalue::PValue;

/// Represents the internal data of a scalar value. Must be interpreted by wrapping
/// up with a DType to make a Scalar.
///
/// Note that these values can be deserialized from JSON or other formats. So a PValue may not
/// have the correct width for what the DType expects. This means primitive values must be
/// cast on-read.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ScalarValue {
    Bool(bool),
    Primitive(PValue),
    Buffer(Buffer),
    BufferString(BufferString),
    List(Arc<[ScalarValue]>),
    // It's significant that Null is last in this list. As a result generated PartialOrd sorts Scalar
    // values such that Nulls are last (greatest)
    Null,
}

fn to_hex(slice: &[u8]) -> Result<String, std::fmt::Error> {
    let mut output = String::new();
    for byte in slice {
        write!(output, "{:02x}", byte)?;
    }
    Ok(output)
}

impl Display for ScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarValue::Bool(b) => write!(f, "{}", b),
            ScalarValue::Primitive(pvalue) => write!(f, "{}", pvalue),
            ScalarValue::Buffer(buf) => {
                if buf.len() > 10 {
                    write!(
                        f,
                        "{}..{}",
                        to_hex(&buf[0..5])?,
                        to_hex(&buf[buf.len() - 5..buf.len()])?,
                    )
                } else {
                    write!(f, "{}", to_hex(buf.as_slice())?)
                }
            }
            ScalarValue::BufferString(bufstr) => {
                if bufstr.len() > 10 {
                    write!(
                        f,
                        "{}..{}",
                        &bufstr.as_str()[0..5],
                        &bufstr.as_str()[bufstr.len() - 5..bufstr.len()],
                    )
                } else {
                    write!(f, "{}", bufstr.as_str())
                }
            }
            ScalarValue::List(_) => todo!(),
            ScalarValue::Null => write!(f, "null"),
        }
    }
}

impl ScalarValue {
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    pub fn is_instance_of(&self, dtype: &DType) -> bool {
        match (self, dtype) {
            (ScalarValue::Bool(_), DType::Bool(_)) => true,
            (ScalarValue::Primitive(pvalue), DType::Primitive(ptype, _)) => {
                pvalue.is_instance_of(ptype)
            }
            (ScalarValue::Buffer(_), DType::Binary(_)) => true,
            (ScalarValue::BufferString(_), DType::Utf8(_)) => true,
            (ScalarValue::List(values), DType::List(dtype, _)) => {
                values.iter().all(|v| v.is_instance_of(dtype))
            }
            (ScalarValue::List(values), DType::Struct(structdt, _)) => values
                .iter()
                .zip(structdt.dtypes().to_vec())
                .all(|(v, dt)| v.is_instance_of(&dt)),
            (ScalarValue::Null, dtype) => dtype.is_nullable(),
            (_, DType::Extension(ext_dtype)) => self.is_instance_of(ext_dtype.storage_dtype()),
            _ => false,
        }
    }

    pub fn as_null(&self) -> VortexResult<()> {
        match self {
            Self::Null => Ok(()),
            _ => Err(vortex_err!("Expected a Null scalar, found {:?}", self)),
        }
    }

    pub fn as_bool(&self) -> VortexResult<Option<bool>> {
        match self {
            Self::Null => Ok(None),
            Self::Bool(b) => Ok(Some(*b)),
            _ => Err(vortex_err!("Expected a bool scalar, found {:?}", self)),
        }
    }

    pub fn as_pvalue(&self) -> VortexResult<Option<PValue>> {
        match self {
            Self::Null => Ok(None),
            Self::Primitive(p) => Ok(Some(*p)),
            _ => Err(vortex_err!("Expected a primitive scalar, found {:?}", self)),
        }
    }

    pub fn as_buffer(&self) -> VortexResult<Option<Buffer>> {
        match self {
            Self::Null => Ok(None),
            Self::Buffer(b) => Ok(Some(b.clone())),
            _ => Err(vortex_err!("Expected a binary scalar, found {:?}", self)),
        }
    }

    pub fn as_buffer_string(&self) -> VortexResult<Option<BufferString>> {
        match self {
            Self::Null => Ok(None),
            Self::Buffer(b) => Ok(Some(BufferString::try_from(b.clone())?)),
            Self::BufferString(b) => Ok(Some(b.clone())),
            _ => Err(vortex_err!("Expected a string scalar, found {:?}", self)),
        }
    }

    pub fn as_list(&self) -> VortexResult<Option<&Arc<[Self]>>> {
        match self {
            Self::Null => Ok(None),
            Self::List(l) => Ok(Some(l)),
            _ => Err(vortex_err!("Expected a list scalar, found {:?}", self)),
        }
    }
}

impl From<usize> for ScalarValue {
    fn from(value: usize) -> Self {
        ScalarValue::Primitive(PValue::from(value))
    }
}

impl From<String> for ScalarValue {
    fn from(value: String) -> Self {
        ScalarValue::BufferString(BufferString::from(value))
    }
}

impl From<BufferString> for ScalarValue {
    fn from(value: BufferString) -> Self {
        ScalarValue::BufferString(value)
    }
}

impl From<bytes::Bytes> for ScalarValue {
    fn from(value: bytes::Bytes) -> Self {
        ScalarValue::Buffer(Buffer::from(value))
    }
}

impl From<Buffer> for ScalarValue {
    fn from(value: Buffer) -> Self {
        ScalarValue::Buffer(value)
    }
}

impl<T> From<Option<T>> for ScalarValue
where
    ScalarValue: From<T>,
{
    fn from(value: Option<T>) -> Self {
        match value {
            None => ScalarValue::Null,
            Some(value) => ScalarValue::from(value),
        }
    }
}

macro_rules! from_vec_for_scalar_value {
    ($T:ty) => {
        impl From<Vec<$T>> for ScalarValue {
            fn from(value: Vec<$T>) -> Self {
                ScalarValue::List(
                    value
                        .into_iter()
                        .map(ScalarValue::from)
                        .collect::<Vec<_>>()
                        .into(),
                )
            }
        }
    };
}

// no From<Vec<u8>> because it could either be a List or a Buffer
from_vec_for_scalar_value!(u16);
from_vec_for_scalar_value!(u32);
from_vec_for_scalar_value!(u64);
from_vec_for_scalar_value!(usize);
from_vec_for_scalar_value!(i8);
from_vec_for_scalar_value!(i16);
from_vec_for_scalar_value!(i32);
from_vec_for_scalar_value!(i64);
from_vec_for_scalar_value!(f16);
from_vec_for_scalar_value!(f32);
from_vec_for_scalar_value!(f64);
from_vec_for_scalar_value!(String);
from_vec_for_scalar_value!(BufferString);
from_vec_for_scalar_value!(bytes::Bytes);
from_vec_for_scalar_value!(Buffer);

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability, PType, StructDType};

    use crate::{PValue, ScalarValue};

    #[test]
    pub fn test_is_instance_of_bool() {
        assert!(ScalarValue::Bool(true).is_instance_of(&DType::Bool(Nullability::Nullable)));
        assert!(ScalarValue::Bool(true).is_instance_of(&DType::Bool(Nullability::NonNullable)));
        assert!(ScalarValue::Bool(false).is_instance_of(&DType::Bool(Nullability::Nullable)));
        assert!(ScalarValue::Bool(false).is_instance_of(&DType::Bool(Nullability::NonNullable)));
    }

    #[test]
    pub fn test_is_instance_of_primitive() {
        assert!(ScalarValue::Primitive(PValue::F64(0.0))
            .is_instance_of(&DType::Primitive(PType::F64, Nullability::NonNullable)));
    }

    #[test]
    pub fn test_is_instance_of_list_and_struct() {
        let tbool = DType::Bool(Nullability::NonNullable);
        let tboolnull = DType::Bool(Nullability::Nullable);
        let tnull = DType::Null;

        let bool_null = ScalarValue::List(vec![ScalarValue::Bool(true), ScalarValue::Null].into());
        let bool_bool =
            ScalarValue::List(vec![ScalarValue::Bool(true), ScalarValue::Bool(false)].into());

        fn tlist(element: &DType) -> DType {
            DType::List(element.clone().into(), Nullability::NonNullable)
        }

        assert!(bool_null.is_instance_of(&tlist(&tboolnull)));
        assert!(!bool_null.is_instance_of(&tlist(&tbool)));
        assert!(bool_bool.is_instance_of(&tlist(&tbool)));
        assert!(bool_bool.is_instance_of(&tlist(&tbool)));

        fn tstruct(left: &DType, right: &DType) -> DType {
            DType::Struct(
                StructDType::new(
                    vec!["left".into(), "right".into()].into(),
                    vec![left.clone(), right.clone()],
                ),
                Nullability::NonNullable,
            )
        }

        assert!(bool_null.is_instance_of(&tstruct(&tboolnull, &tboolnull)));
        assert!(bool_null.is_instance_of(&tstruct(&tbool, &tboolnull)));
        assert!(!bool_null.is_instance_of(&tstruct(&tboolnull, &tbool)));
        assert!(!bool_null.is_instance_of(&tstruct(&tbool, &tbool)));

        assert!(bool_null.is_instance_of(&tstruct(&tbool, &tnull)));
        assert!(!bool_null.is_instance_of(&tstruct(&tnull, &tbool)));

        assert!(bool_bool.is_instance_of(&tstruct(&tboolnull, &tboolnull)));
        assert!(bool_bool.is_instance_of(&tstruct(&tbool, &tboolnull)));
        assert!(bool_bool.is_instance_of(&tstruct(&tboolnull, &tbool)));
        assert!(bool_bool.is_instance_of(&tstruct(&tbool, &tbool)));

        assert!(!bool_bool.is_instance_of(&tstruct(&tbool, &tnull)));
        assert!(!bool_bool.is_instance_of(&tstruct(&tnull, &tbool)));
    }

    #[test]
    pub fn test_is_instance_of_null() {
        assert!(ScalarValue::Null.is_instance_of(&DType::Bool(Nullability::Nullable)));
        assert!(!ScalarValue::Null.is_instance_of(&DType::Bool(Nullability::NonNullable)));

        assert!(
            ScalarValue::Null.is_instance_of(&DType::Primitive(PType::U8, Nullability::Nullable))
        );
        assert!(ScalarValue::Null.is_instance_of(&DType::Utf8(Nullability::Nullable)));
        assert!(ScalarValue::Null.is_instance_of(&DType::Binary(Nullability::Nullable)));
        assert!(ScalarValue::Null.is_instance_of(&DType::Struct(
            StructDType::new([].into(), [].into()),
            Nullability::Nullable,
        )));
        assert!(ScalarValue::Null.is_instance_of(&DType::List(
            DType::Utf8(Nullability::NonNullable).into(),
            Nullability::Nullable
        )));
        assert!(ScalarValue::Null.is_instance_of(&DType::Null));
    }
}
