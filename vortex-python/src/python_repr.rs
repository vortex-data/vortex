// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::convert::AsRef;
use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtId;

pub trait PythonRepr {
    fn python_repr(&self) -> impl Display;
}

struct DTypePythonRepr<'a>(&'a DType);

impl PythonRepr for DType {
    fn python_repr(&self) -> impl Display {
        DTypePythonRepr(self)
    }
}

// TODO(connor): We should probably just use the `Display` impl on `DType`.
impl Display for DTypePythonRepr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let DTypePythonRepr(dtype) = self;
        match dtype {
            DType::Null => write!(f, "null()"),
            DType::Bool(n) => write!(f, "bool(nullable={})", n.python_repr()),
            DType::Primitive(ptype, n) => match ptype {
                PType::U8 | PType::U16 | PType::U32 | PType::U64 => {
                    write!(
                        f,
                        "uint({}, nullable={})",
                        ptype.bit_width(),
                        n.python_repr()
                    )
                }
                PType::I8 | PType::I16 | PType::I32 | PType::I64 => {
                    write!(
                        f,
                        "int({}, nullable={})",
                        ptype.bit_width(),
                        n.python_repr()
                    )
                }
                PType::F16 | PType::F32 | PType::F64 => {
                    write!(
                        f,
                        "float({}, nullable={})",
                        ptype.bit_width(),
                        n.python_repr()
                    )
                }
            },
            DType::Decimal(decimal_type, n) => {
                write!(
                    f,
                    "decimal(precision={}, scale={}, nullable={})",
                    decimal_type.precision(),
                    decimal_type.scale(),
                    n.python_repr()
                )
            }
            DType::Utf8(n) => write!(f, "utf8(nullable={})", n.python_repr()),
            DType::Binary(n) => write!(f, "binary(nullable={})", n.python_repr()),
            DType::Struct(st, n) => write!(
                f,
                "struct({{{}}}, nullable={})",
                st.names()
                    .iter()
                    .zip(st.fields())
                    .map(|(n, dt)| format!("\"{}\": {}", n, dt.python_repr()))
                    .join(", "),
                n.python_repr()
            ),
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::List(edt, n) => write!(
                f,
                "list({}, nullable={})",
                edt.python_repr(),
                n.python_repr()
            ),
            DType::FixedSizeList(edt, size, n) => write!(
                f,
                "fixed_size_list({}, {}, nullable={})",
                edt.python_repr(),
                size,
                n.python_repr()
            ),
            DType::Extension(ext) => {
                write!(
                    f,
                    "ext({}, {}",
                    ext.id().python_repr(),
                    ext.storage_dtype().python_repr()
                )?;
                let opts = ext.display_metadata().to_string();
                if !opts.is_empty() {
                    write!(f, ", {}", opts)?
                }
                write!(f, ")")
            }
            DType::Variant(_) => write!(f, "variant()"),
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

struct ExtIdPythonRepr<'a>(&'a ExtId);

impl PythonRepr for ExtId {
    fn python_repr(&self) -> impl Display {
        ExtIdPythonRepr(self)
    }
}

impl Display for ExtIdPythonRepr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ExtIdPythonRepr(ext_id) = self;
        write!(f, "\"{}\"", ext_id.as_ref().escape_default())
    }
}
