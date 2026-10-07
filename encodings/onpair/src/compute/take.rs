// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::OnPair;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;

impl TakeExecute for OnPair {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Dense dictionary expansion should decode each unique value once, then gather views.
        // Taking the compressed token stream would instead decode repeated values repeatedly.
        if indices.len() > array.len() / 4 {
            return Ok(None);
        }
        // The list take gathers token ranges in output order, including duplicates, before any
        // string decoding. Its zero-length ranges for null indices match the zero lengths below.
        // SAFETY: OnPair stores the same elements/offsets representation as a non-nullable list
        // of tokens. Outer string validity is taken separately below.
        let codes = unsafe {
            ListArray::new_unchecked(
                array.codes().clone(),
                array.codes_offsets().clone(),
                Validity::NonNullable,
            )
        };
        let taken_codes = <List as TakeExecute>::take(codes.as_view(), indices, ctx)?
            .vortex_expect("List take always returns Some");
        let taken_codes = taken_codes
            .as_opt::<List>()
            .vortex_expect("List take returns a List");
        let lengths = array.uncompressed_lengths();
        let lengths = lengths
            .take(indices.clone())?
            .fill_null(Scalar::zero_value(lengths.dtype()))?;
        let validity = array.array_validity().take(indices)?;

        // SAFETY: list take preserves token codes and rebuilds valid offsets. Lengths and outer
        // validity use the same indices, and both dictionary children remain unchanged.
        Ok(Some(
            unsafe {
                OnPair::new_unchecked(
                    array.dtype().union_nullability(indices.dtype().nullability()),
                    array.data().clone(),
                    array.dict_offsets().clone(),
                    taken_codes.elements().clone(),
                    taken_codes.offsets().clone(),
                    lengths,
                    validity,
                )
            }
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::SharedArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::shared::SharedArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::VarBinBuilder;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;

    use super::*;
    use crate::DEFAULT_CONFIG;
    use crate::compress::onpair_compress;

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn compressed_take_conformance(#[case] dtype: DType) -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let first = match dtype {
            DType::Binary(_) => b"long binary value with \xff\x00".as_slice(),
            _ => "long first value with 数据".as_bytes(),
        };
        let values = [
            Some(first),
            Some(b"".as_slice()),
            None,
            Some(b"short".as_slice()),
            Some(b"last long value".as_slice()),
        ];
        let input = VarBinArray::from_iter((0..64).map(|row| values[row % values.len()]), dtype);
        let encoded = onpair_compress(input.as_ref(), DEFAULT_CONFIG, &mut ctx)?;
        test_take_conformance(&encoded, &mut ctx);
        test_take_conformance(&encoded.slice(1..5)?, &mut ctx);
        Ok(())
    }

    #[test]
    fn sparse_shared_dictionary_takes_before_decoding() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let dtype = DType::Utf8(Nullability::Nullable);
        let input = VarBinArray::from_iter(
            (0..1024).map(|row| (row != 7).then(|| format!("long dictionary value {row}"))),
            dtype.clone(),
        );
        let encoded = onpair_compress(input.as_ref(), DEFAULT_CONFIG, &mut ctx)?;
        let encoded = encoded
            .as_opt::<OnPair>()
            .vortex_expect("input compresses with OnPair");
        let indices =
            PrimitiveArray::from_option_iter([Some(1023u32), None, Some(7), Some(0), Some(1023)])
                .into_array();
        let taken = <OnPair as TakeExecute>::take(encoded, &indices, &mut ctx)?
            .vortex_expect("OnPair take returns Some");
        let compressed = taken
            .as_opt::<OnPair>()
            .vortex_expect("take preserves OnPair");
        assert!(compressed.codes().len() < encoded.codes().len());
        assert_arrays_eq!(taken, input.take(indices.clone())?, &mut ctx);

        let shared = SharedArray::new(encoded.array().clone());
        let selected = shared.clone().into_array().take(indices)?;
        let mut builder = VarBinBuilder::<i32>::new_in(dtype.clone(), ctx.allocator());
        builder.append_value("prefix");
        selected.append_to_builder(&mut builder, &mut ctx)?;
        assert!(shared.current_array_ref().as_opt::<OnPair>().is_some());
        let expected = VarBinViewArray::from_iter(
            [
                Some("prefix"),
                Some("long dictionary value 1023"),
                None,
                None,
                Some("long dictionary value 0"),
                Some("long dictionary value 1023"),
            ],
            dtype,
        );
        assert_arrays_eq!(builder.finish_into_varbin(), expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn batched_shared_dictionary_keeps_decoded_cache() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let dtype = DType::Utf8(Nullability::Nullable);
        let input = VarBinArray::from_iter(
            (0..64).map(|row| Some(format!("long dictionary value {row}"))),
            dtype.clone(),
        );
        let encoded = onpair_compress(input.as_ref(), DEFAULT_CONFIG, &mut ctx)?;
        let shared = SharedArray::new(encoded);
        let indices = PrimitiveArray::from_iter(0u32..8).into_array();
        let selected = shared.clone().into_array().take(indices.clone())?;
        let mut builder = VarBinBuilder::<i32>::new_in(dtype, ctx.allocator());
        selected.append_to_builder(&mut builder, &mut ctx)?;
        assert!(shared.current_array_ref().as_opt::<OnPair>().is_none());
        assert_arrays_eq!(builder.finish_into_varbin(), input.take(indices)?, &mut ctx);
        Ok(())
    }
}
