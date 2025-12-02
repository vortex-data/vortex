// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::FieldPath;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_proto::expr as pb;
use vortex_vector::Datum;
use vortex_vector::ScalarOps;
use vortex_vector::VectorOps;

use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::functions::ArgName;
use crate::expr::functions::Arity;
use crate::expr::functions::ExecutionArgs;
use crate::expr::functions::FunctionId;
use crate::expr::functions::VTable;
use crate::expr::stats::Stat;

pub struct GetItemFn;
impl VTable for GetItemFn {
    type Options = FieldName;

    fn id(&self) -> FunctionId {
        FunctionId::from("vortex.get_item")
    }

    fn serialize(&self, field_name: &FieldName) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::GetItemOpts {
                path: field_name.to_string(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, bytes: &[u8]) -> VortexResult<Self::Options> {
        let opts = pb::GetItemOpts::decode(bytes)?;
        Ok(FieldName::from(opts.path))
    }

    fn arity(&self, _field_name: &FieldName) -> Arity {
        Arity::Exact(1)
    }

    fn arg_name(&self, _field_name: &FieldName, _arg_idx: usize) -> ArgName {
        ArgName::from("input")
    }

    fn stat_expression(
        &self,
        field_name: &FieldName,
        _expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        // TODO(ngates): I think we can do better here and support stats over nested fields.
        //  It would be nice if delegating to our child would return a struct of statistics
        //  matching the nested DType such that we can write:
        //    `get_item(expr.child(0).stat_expression(...), expr.data().field_name())`

        // TODO(ngates): this is a bug whereby we may return stats for a nested field of the same
        //  name as a field in the root struct. This should be resolved with upcoming change to
        //  falsify expressions, but for now I'm preserving the existing buggy behavior.
        catalog.stats_ref(&FieldPath::from_name(field_name.clone()), stat)
    }

    fn return_dtype(&self, field_name: &FieldName, arg_types: &[DType]) -> VortexResult<DType> {
        let struct_dtype = &arg_types[0];
        let field_dtype = struct_dtype
            .as_struct_fields_opt()
            .and_then(|st| st.field(field_name))
            .ok_or_else(|| {
                vortex_err!("Couldn't find the {} field in the input scope", field_name)
            })?;

        // Match here to avoid cloning the dtype if nullability doesn't need to change
        if matches!(
            (struct_dtype.nullability(), field_dtype.nullability()),
            (Nullability::Nullable, Nullability::NonNullable)
        ) {
            return Ok(field_dtype.with_nullability(Nullability::Nullable));
        }

        Ok(field_dtype)
    }

    fn execute(&self, field_name: &FieldName, args: &ExecutionArgs) -> VortexResult<Datum> {
        let struct_dtype = args
            .input_type(0)
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Expected struct dtype for child of GetItem expression"))?;
        let field_idx = struct_dtype
            .find(field_name)
            .ok_or_else(|| vortex_err!("Field {} not found in struct dtype", field_name))?;

        match args.input_datums(0) {
            Datum::Scalar(s) => {
                let mut field = s.as_struct().field(field_idx);
                field.mask_validity(s.is_valid());
                Ok(Datum::Scalar(field))
            }
            Datum::Vector(v) => {
                let mut field = v.as_struct().fields()[field_idx].clone();
                field.mask_validity(v.validity());
                Ok(Datum::Vector(field))
            }
        }
    }
}
