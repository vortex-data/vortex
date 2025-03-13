use duckdb::vtab::Value;
use vortex_dtype::{DType, PType};
use vortex_error::VortexExpect;
use vortex_scalar::{BoolScalar, PrimitiveScalar, Scalar};

pub trait ToDuckDBScalar {
    fn to_duckdb_scalar(&self) -> Value;
}

impl ToDuckDBScalar for Scalar {
    fn to_duckdb_scalar(&self) -> Value {
        match self.dtype() {
            DType::Null => todo!(),
            DType::Bool(_) => self.as_bool().to_duckdb_scalar(),
            DType::Primitive(..) => prim_to_duckdb_scalar(self.as_primitive()),
            DType::Utf8(_)
            | DType::Binary(_)
            | DType::Struct(..)
            | DType::List(..)
            | DType::Extension(_) => todo!(),
        }
    }
}

fn prim_to_duckdb_scalar(scalar: PrimitiveScalar) -> Value {
    match scalar.ptype() {
        PType::U8 | PType::U16 | PType::U32 | PType::U64 | PType::I8 | PType::I16 => todo!(),
        PType::I32 => Value::from(scalar.as_::<i32>().vortex_expect("is i32")),
        PType::I64 => Value::from(scalar.as_::<i64>().vortex_expect("is i64")),
        PType::F16 | PType::F32 | PType::F64 => todo!(),
    }
}

impl ToDuckDBScalar for BoolScalar<'_> {
    fn to_duckdb_scalar(&self) -> Value {
        Value::from(self.value())
    }
}
