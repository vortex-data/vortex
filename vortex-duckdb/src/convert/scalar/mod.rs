use duckdb::core::Value;
use vortex_dtype::datetime::{TemporalMetadata, TimeUnit};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, PType, match_each_native_simd_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{BoolScalar, ExtScalar, PrimitiveScalar, Scalar};

pub trait ToDuckDBScalar {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value>;
}

impl ToDuckDBScalar for Scalar {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        match self.dtype() {
            DType::Null => Ok(Value::null()),
            DType::Bool(_) => self.as_bool().try_to_duckdb_scalar(),
            DType::Primitive(..) => self.as_primitive().try_to_duckdb_scalar(),
            DType::Extension(..) => self.as_extension().try_to_duckdb_scalar(),
            DType::Utf8(_) | DType::Binary(_) | DType::Struct(..) | DType::List(..) => todo!(),
        }
    }
}

// fn prim_to_duckdb_scalar(scalar: PrimitiveScalar) -> Value {}
impl ToDuckDBScalar for PrimitiveScalar<'_> {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        if self.ptype() == PType::F16 {
            return Ok(Value::from(
                self.as_::<f16>()
                    .vortex_expect("check ptyped")
                    .map(|f| f.to_f32()),
            ));
        }
        match_each_native_simd_ptype!(self.ptype(), |$P| {
            Ok(Value::from(self.as_::<$P>().vortex_expect("ptype value mismatch")))
        })
    }
}

impl ToDuckDBScalar for BoolScalar<'_> {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        Ok(Value::from(self.value()))
    }
}

impl ToDuckDBScalar for ExtScalar<'_> {
    fn try_to_duckdb_scalar(&self) -> VortexResult<Value> {
        let time = TemporalMetadata::try_from(self.ext_dtype())?;
        match time {
            TemporalMetadata::Time(_) => todo!(),
            TemporalMetadata::Date(unit) => match unit {
                TimeUnit::D => Ok(self
                    .storage()
                    .as_primitive_opt()
                    .ok_or_else(|| {
                        vortex_err!("temporal types must be backed by primitive scalars")
                    })?
                    .as_::<i32>()?
                    .map(Value::date_from_day_count)
                    .unwrap_or_else(Value::null)),
                _ => vortex_bail!("cannot have TimeUnit {unit}, so represent a day"),
            },
            TemporalMetadata::Timestamp(..) => todo!(),
        }
    }
}
