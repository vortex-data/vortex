use std::collections::HashSet;

use itertools::Itertools as _;
use vortex_dtype::{match_each_native_ptype, DType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_scalar::{Scalar, ScalarValue};

use crate::array::builder::VarBinBuilder;
use crate::array::{BoolArray, NullArray, PrimitiveArray};
use crate::{Array, IntoArray as _};

impl Array {
    pub fn from_scalar_values(dtype: DType, values: Vec<ScalarValue>) -> VortexResult<Array> {
        let mismatched_values = values
            .iter()
            .filter(|v| !v.is_instance_of(&dtype))
            .collect_vec();
        if !mismatched_values.is_empty() {
            let mismatch_str = mismatched_values.iter().map(|v| v.to_string()).join(", ");
            vortex_bail!("Expected all scalars to be of type {dtype}; found {mismatch_str}");
        }

        Ok(match dtype {
            DType::Bool(_) => BoolArray::from_iter(values.into_iter().map(|s| {
                s.as_bool()
                    .vortex_expect("Expected ScalarValue to be a bool")
            }))
            .into_array(),
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |$P| {
                    PrimitiveArray::from_nullable_vec(values.iter().map(|s| {
                        s.as_pvalue()
                            .vortex_expect("Expected ScalarValue to be a primitive")
                            .map(|p| p.as_primitive::<$P>().vortex_expect("Expected ScalarValue to be a primitive"))
                    }).collect_vec())
                    .into_array()
                })
            }
            DType::Binary(_) => {
                let mut builder = VarBinBuilder::<u64>::with_capacity(values.len());
                for value in values {
                    let buf = value.as_buffer()?;
                    builder.push(buf.as_ref().map(|b| b.as_slice()));
                }
                builder.finish(dtype.clone()).into_array()
            }
            DType::Utf8(_) => {
                let mut builder = VarBinBuilder::<u64>::with_capacity(values.len());
                for value in values {
                    let buf_str = value.as_buffer_string()?;
                    builder.push(buf_str.as_ref().map(|b| b.as_bytes()));
                }
                builder.finish(dtype.clone()).into_array()
            }
            DType::List(..) => vortex_bail!("Cannot convert ScalarValues to ListArray"),
            DType::Struct(..) => vortex_bail!("Cannot convert ScalarValues to StructArray"),
            DType::Null => NullArray::new(values.len()).into_array(),
            DType::Extension(..) => vortex_bail!("Cannot convert ScalarValues to ExtensionArray"),
        })
    }

    pub fn from_scalars(scalars: &[Scalar]) -> VortexResult<Array> {
        if scalars.is_empty() {
            vortex_bail!("Cannot convert empty Vec<Scalar> to canonical");
        }

        let scalar = scalars[0].clone();
        let dtype = scalar.dtype();

        let mismatched_types: HashSet<&DType> = scalars
            .iter()
            .skip(1)
            .filter(|s| s.dtype() != dtype)
            .map(|s| s.dtype())
            .collect();
        if !mismatched_types.is_empty() {
            let mismatch_str = mismatched_types.iter().map(|t| t.to_string()).join(", ");
            vortex_bail!("Expected all scalars to be of type {dtype}; also found {mismatch_str}");
        }

        let scalar_values = scalars.iter().map(|s| s.value().clone()).collect_vec();
        Array::from_scalar_values(dtype.clone(), scalar_values)
    }
}

#[cfg(test)]
mod test {}
