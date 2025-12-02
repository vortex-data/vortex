// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod array;

use prost::Message;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_proto::expr as pb;
use vortex_vector::Datum;

use crate::expr::Expression;
use crate::expr::StatsCatalog;
use crate::expr::functions::ArgName;
use crate::expr::functions::Arity;
use crate::expr::functions::ExecutionArgs;
use crate::expr::functions::FunctionId;
use crate::expr::functions::NullHandling;
use crate::expr::functions::VTable;
use crate::expr::stats::Stat;
use crate::scalar_fns::ExprBuiltins;

pub struct CastFn;
impl VTable for CastFn {
    type Options = DType;

    fn id(&self) -> FunctionId {
        FunctionId::from("vortex.cast")
    }

    fn serialize(&self, target_dtype: &DType) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::CastOpts {
                target: Some(target_dtype.into()),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, bytes: &[u8]) -> VortexResult<DType> {
        pb::CastOpts::decode(bytes)?
            .target
            .as_ref()
            .ok_or_else(|| vortex_err!("Missing target dtype in Cast expression"))?
            .try_into()
    }

    fn arity(&self, _options: &DType) -> Arity {
        Arity::Exact(1)
    }

    fn null_handling(&self, _options: &DType) -> NullHandling {
        NullHandling::Propagate
    }

    fn arg_name(&self, _options: &DType, arg_idx: usize) -> ArgName {
        match arg_idx {
            0 => ArgName::from("input"),
            _ => vortex_panic!("Invalid argument index {}", arg_idx),
        }
    }

    fn stat_expression(
        &self,
        target_dtype: &DType,
        expr: &Expression,
        stat: Stat,
        catalog: &dyn StatsCatalog,
    ) -> Option<Expression> {
        match stat {
            Stat::IsConstant
            | Stat::IsSorted
            | Stat::IsStrictSorted
            | Stat::NaNCount
            | Stat::Sum
            | Stat::UncompressedSizeInBytes => expr.child(0).stat_expression(stat, catalog),
            Stat::Max | Stat::Min => {
                // We cast min/max to the new type
                expr.child(0).stat_expression(stat, catalog).map(|x| {
                    x.cast(target_dtype.clone())
                        .vortex_expect("Failed to cast stat expression")
                })
            }
            Stat::NullCount => {
                // if !expr.data().is_nullable() {
                // NOTE(ngates): we should decide on the semantics here. In theory, the null
                //  count of something cast to non-nullable will be zero. But if we return
                //  that we know this to be zero, then a pruning predicate may eliminate data
                //  that would otherwise have caused the cast to error.
                // return Some(lit(0u64));
                // }
                None
            }
        }
    }

    fn return_dtype(&self, target_dtype: &DType, _arg_types: &[DType]) -> VortexResult<DType> {
        Ok(target_dtype.clone())
    }

    fn execute(&self, target_dtype: &DType, args: &ExecutionArgs) -> VortexResult<Datum> {
        let datum = args.input_datums(0);
        vortex_compute::cast::Cast::cast(datum, target_dtype)
    }
}
