use duckdb::core::Value;
use vortex_dtype::{DType, match_each_native_ptype};
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
    match_each_native_ptype!(scalar.ptype(), |$P| {
        Value::from(scalar.as_::<u64>().vortex_expect("ptype value mismatch"))
    })
}

impl ToDuckDBScalar for BoolScalar<'_> {
    fn to_duckdb_scalar(&self) -> Value {
        Value::from(self.value())
    }
}
