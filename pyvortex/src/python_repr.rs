use std::convert::AsRef;
use std::fmt::{Display, Formatter};

use itertools::Itertools;
use vortex::dtype::{DType, ExtID, ExtMetadata, Nullability, PType};

pub trait PythonRepr {
    fn python_repr(&self) -> impl Display;
}

struct DTypePythonRepr<'a>(&'a DType);

impl PythonRepr for DType {
    fn python_repr(&self) -> impl Display {
        DTypePythonRepr(self)
    }
}

impl Display for DTypePythonRepr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let DTypePythonRepr(dtype) = self;
        match dtype {
            DType::Null => write!(f, "null()"),
            DType::Bool(n) => write!(f, "bool({})", n.python_repr()),
            DType::Primitive(ptype, n) => match ptype {
                PType::U8 | PType::U16 | PType::U32 | PType::U64 => {
                    write!(f, "uint({}, {})", ptype.bit_width(), n.python_repr())
                }
                PType::I8 | PType::I16 | PType::I32 | PType::I64 => {
                    write!(f, "int({}, {})", ptype.bit_width(), n.python_repr())
                }
                PType::F16 | PType::F32 | PType::F64 => {
                    write!(f, "float({}, {})", ptype.bit_width(), n.python_repr())
                }
            },
            DType::Utf8(n) => write!(f, "utf8({})", n.python_repr()),
            DType::Binary(n) => write!(f, "binary({})", n.python_repr()),
            DType::Struct(st, n) => write!(
                f,
                "struct({{{}}}, {})",
                st.names()
                    .iter()
                    .zip(st.dtypes().iter())
                    .map(|(n, dt)| format!("\"{}\": {}", n, dt.python_repr()))
                    .join(", "),
                n.python_repr()
            ),
            DType::List(edt, n) => write!(f, "list({}, {})", edt.python_repr(), n.python_repr()),
            DType::Extension(ext) => {
                write!(
                    f,
                    "ext(\"{}\", {}, ",
                    ext.id().python_repr(),
                    ext.storage_dtype().python_repr()
                )?;
                match ext.metadata() {
                    None => write!(f, "None")?,
                    Some(metadata) => write!(f, "{}", metadata.python_repr())?,
                };
                write!(f, ")")
            }
        }
    }
}

struct NullabilityPythonRepr<'a>(&'a Nullability);

impl PythonRepr for Nullability {
    fn python_repr(&self) -> impl Display {
        NullabilityPythonRepr(self)
    }
}

impl Display for NullabilityPythonRepr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let NullabilityPythonRepr(x) = self;
        match x {
            Nullability::NonNullable => write!(f, "False"),
            Nullability::Nullable => write!(f, "True"),
        }
    }
}

struct ExtMetadataPythonRepr<'a>(&'a ExtMetadata);

impl PythonRepr for ExtMetadata {
    fn python_repr(&self) -> impl Display {
        ExtMetadataPythonRepr(self)
    }
}

impl Display for ExtMetadataPythonRepr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ExtMetadataPythonRepr(metadata) = self;
        write!(f, "\"{}\"", metadata.as_ref().escape_ascii())
    }
}

struct ExtIDPythonRepr<'a>(&'a ExtID);

impl PythonRepr for ExtID {
    fn python_repr(&self) -> impl Display {
        ExtIDPythonRepr(self)
    }
}

impl Display for ExtIDPythonRepr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ExtIDPythonRepr(ext_id) = self;
        write!(f, "\"{}\"", ext_id.as_ref().escape_default())
    }
}
