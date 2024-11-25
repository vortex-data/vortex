use std::fmt::{Display, Write};
use std::sync::Arc;

use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexResult};

use crate::pvalue::PValue;

/// Represents the internal data of a scalar value. Must be interpreted by wrapping
/// up with a DType to make a Scalar.
///
/// Note that these values can be deserialized from JSON or other formats. So a PValue may not
/// have the correct width for what the DType expects. Primitive values should therefore be
/// read using [PrimitiveScalar] which will handle the conversion.
///
/// For this reason, [`PartialEq`] and [`PartialOrd`] are only defined over [`Scalar`] and not
/// [`ScalarValue`].
#[derive(Debug, Clone)]
pub struct ScalarValue(pub(crate) InnerScalarValue);

#[derive(Debug, Clone)]
pub enum InnerScalarValue {
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
        write!(f, "{}", self.0)
    }
}

impl Display for InnerScalarValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{}", b),
            Self::Primitive(pvalue) => write!(f, "{}", pvalue),
            Self::Buffer(buf) => {
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
            Self::BufferString(bufstr) => {
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
            Self::List(_) => todo!(),
            Self::Null => write!(f, "null"),
        }
    }
}

impl ScalarValue {
    pub(crate) fn is_null(&self) -> bool {
        matches!(self.0, InnerScalarValue::Null)
    }

    pub fn is_instance_of(&self, dtype: &DType) -> bool {
        match (&self.0, dtype) {
            (InnerScalarValue::Bool(_), DType::Bool(_)) => true,
            (InnerScalarValue::Primitive(pvalue), DType::Primitive(ptype, _)) => {
                pvalue.is_instance_of(ptype)
            }
            (InnerScalarValue::Buffer(_), DType::Binary(_)) => true,
            (InnerScalarValue::BufferString(_), DType::Utf8(_)) => true,
            (InnerScalarValue::List(values), DType::List(dtype, _)) => {
                values.iter().all(|v| v.is_instance_of(dtype))
            }
            (InnerScalarValue::List(values), DType::Struct(structdt, _)) => values
                .iter()
                .zip(structdt.dtypes().to_vec())
                .all(|(v, dt)| v.is_instance_of(&dt)),
            (InnerScalarValue::Null, dtype) => dtype.is_nullable(),
            (_, DType::Extension(ext_dtype)) => self.is_instance_of(ext_dtype.storage_dtype()),
            _ => false,
        }
    }

    pub(crate) fn as_null(&self) -> VortexResult<()> {
        match self.0 {
            InnerScalarValue::Null => Ok(()),
            _ => Err(vortex_err!("Expected a Null scalar, found {:?}", self.0)),
        }
    }

    pub(crate) fn as_bool(&self) -> VortexResult<Option<bool>> {
        match &self.0 {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::Bool(b) => Ok(Some(*b)),
            _ => Err(vortex_err!("Expected a bool scalar, found {:?}", self.0)),
        }
    }

    /// FIXME(ngates): PValues are such a footgun... we should probably remove this.
    ///  But the other accessors can sometimes be useful? e.g. as_buffer. But maybe we just force
    ///  the user to switch over Utf8 and Binary and use the correct Scalar wrapper?
    pub(crate) fn as_pvalue(&self) -> VortexResult<Option<PValue>> {
        match &self.0 {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::Primitive(p) => Ok(Some(*p)),
            _ => Err(vortex_err!(
                "Expected a primitive scalar, found {:?}",
                self.0
            )),
        }
    }

    pub(crate) fn as_buffer(&self) -> VortexResult<Option<Buffer>> {
        match &self.0 {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::Buffer(b) => Ok(Some(b.clone())),
            _ => Err(vortex_err!("Expected a binary scalar, found {:?}", self.0)),
        }
    }

    pub(crate) fn as_buffer_string(&self) -> VortexResult<Option<BufferString>> {
        match &self.0 {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::Buffer(b) => Ok(Some(BufferString::try_from(b.clone())?)),
            InnerScalarValue::BufferString(b) => Ok(Some(b.clone())),
            _ => Err(vortex_err!("Expected a string scalar, found {:?}", self.0)),
        }
    }

    pub(crate) fn as_list(&self) -> VortexResult<Option<&Arc<[Self]>>> {
        match &self.0 {
            InnerScalarValue::Null => Ok(None),
            InnerScalarValue::List(l) => Ok(Some(l)),
            _ => Err(vortex_err!("Expected a list scalar, found {:?}", self.0)),
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability, PType, StructDType};

    use crate::{InnerScalarValue, PValue, ScalarValue};

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

        let bool_null = ScalarValue::List(
            vec![ScalarValue::Bool(true), ScalarValue(InnerScalarValue::Null)].into(),
        );
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
        assert!(
            ScalarValue(InnerScalarValue::Null).is_instance_of(&DType::Bool(Nullability::Nullable))
        );
        assert!(!ScalarValue(InnerScalarValue::Null)
            .is_instance_of(&DType::Bool(Nullability::NonNullable)));

        assert!(ScalarValue(InnerScalarValue::Null)
            .is_instance_of(&DType::Primitive(PType::U8, Nullability::Nullable)));
        assert!(
            ScalarValue(InnerScalarValue::Null).is_instance_of(&DType::Utf8(Nullability::Nullable))
        );
        assert!(ScalarValue(InnerScalarValue::Null)
            .is_instance_of(&DType::Binary(Nullability::Nullable)));
        assert!(
            ScalarValue(InnerScalarValue::Null).is_instance_of(&DType::Struct(
                StructDType::new([].into(), [].into()),
                Nullability::Nullable,
            ))
        );
        assert!(
            ScalarValue(InnerScalarValue::Null).is_instance_of(&DType::List(
                DType::Utf8(Nullability::NonNullable).into(),
                Nullability::Nullable
            ))
        );
        assert!(ScalarValue(InnerScalarValue::Null).is_instance_of(&DType::Null));
    }
}
