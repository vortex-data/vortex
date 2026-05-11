// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::ToPrimitive as _;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::ScalarFn;
use crate::arrays::VarBin;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrays::varbin::builder::VarBinBuilder;
use crate::executor::ExecutionCtx;
use crate::kernel::ExecuteParentKernel;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar_fn::fns::substring::Substring;
use crate::scalar_fn::fns::substring::parse_byte_range;

#[derive(Default, Debug)]
pub(crate) struct SubstringVarBin;

impl ExecuteParentKernel<VarBin> for SubstringVarBin {
    type Parent = ExactScalarFn<Substring>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, VarBin>,
        parent: ScalarFnArrayView<'_, Substring>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let children = scalar_fn_array.children();

        let (byte_start, byte_length) = parse_byte_range(&children[1], children.get(2))?;

        let len = array.as_ref().len();
        let dtype = array.dtype().clone();
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let data = array.bytes();
        let validity = array.varbin_validity();

        vortex_ensure!(
            data.len() <= u32::MAX as usize,
            "Substring: input data exceeds 4 GiB"
        );

        let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
        let mut builder = VarBinBuilder::<u32>::with_capacity(len);

        match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
            let offsets_slice = offsets.as_slice::<O>();
            for i in 0..len {
                if validity.is_null(i).unwrap_or(false) {
                    builder.append_null();
                    continue;
                }
                let elem_start = offsets_slice[i]
                    .to_usize()
                    .vortex_expect("offset fits in usize");
                let elem_end = offsets_slice[i + 1]
                    .to_usize()
                    .vortex_expect("offset fits in usize");
                let elem_len = elem_end - elem_start;
                let sub_start = elem_start + byte_start.min(elem_len);
                let sub_end = byte_length
                    .map(|l| (sub_start + l).min(elem_end))
                    .unwrap_or(elem_end);
                builder.append_value(&data[sub_start..sub_end]);
            }
        });

        Ok(Some(builder.finish(dtype).into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::substr;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_start() -> VortexResult<()> {
        let arr = VarBinArray::from(vec!["hello", "world"]).into_array();
        let result = arr
            .apply(&substr(root(), lit(2i64), None))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(result, VarBinViewArray::from_iter_str(["ello", "orld"]));
        Ok(())
    }

    #[test]
    fn test_start_length() -> VortexResult<()> {
        let arr = VarBinArray::from(vec!["hello", "world"]).into_array();
        let result = arr
            .apply(&substr(root(), lit(2i64), Some(lit(3i64))))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(result, VarBinViewArray::from_iter_str(["ell", "orl"]));
        Ok(())
    }

    #[test]
    fn test_null_values() -> VortexResult<()> {
        let arr = VarBinArray::from(vec![Some("hello"), None, Some("world")]).into_array();
        let result = arr
            .apply(&substr(root(), lit(2i64), Some(lit(3i64))))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(
            result,
            VarBinViewArray::from_iter_nullable_str([Some("ell"), None, Some("orl")])
        );
        Ok(())
    }

    #[test]
    fn test_start_beyond_length() -> VortexResult<()> {
        let arr = VarBinArray::from(vec!["hi"]).into_array();
        let result = arr
            .apply(&substr(root(), lit(10i64), None))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(result, VarBinViewArray::from_iter_str([""]));
        Ok(())
    }
}
