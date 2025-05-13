use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{ChunkedArray, ChunkedVTable};
use crate::compute::{ComputeFn, InvocationArgs, Output};
use crate::vtable::ComputeVTable;
use crate::{Array, IntoArray};

impl ComputeVTable<ChunkedVTable> for ChunkedVTable {
    fn invoke(
        array: &ChunkedArray,
        compute_fn: &ComputeFn,
        args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        if compute_fn.is_elementwise() {
            return invoke_elementwise(array, compute_fn, args);
        }
        Ok(None)
    }
}

/// Invoke an element-wise compute function over a chunked array.
fn invoke_elementwise(
    array: &ChunkedArray,
    compute_fn: &ComputeFn,
    args: &InvocationArgs,
) -> VortexResult<Option<Output>> {
    assert!(
        compute_fn.is_elementwise(),
        "Expected elementwise compute function"
    );
    assert!(
        !args.inputs.is_empty(),
        "Elementwise compute function requires at least one input"
    );

    // If not all inputs are arrays, then we pass.
    if args.inputs.iter().any(|a| a.array().is_none()) {
        return Ok(None);
    }

    let mut idx = 0;
    let mut chunks = Vec::with_capacity(array.nchunks());
    let mut inputs = Vec::with_capacity(args.inputs.len());

    for chunk in array.non_empty_chunks() {
        inputs.clear();
        inputs.push(chunk.clone());
        for i in 1..args.inputs.len() {
            let input = args.inputs[i].array().vortex_expect("checked already");
            let sliced = input.slice(idx, idx + chunk.len())?;
            inputs.push(sliced);
        }

        // TODO(ngates): we might want to make invocation args not hold references?
        let input_refs = inputs.iter().map(|a| a.as_ref().into()).collect::<Vec<_>>();

        // Delegate the compute kernel to the chunk.
        let result = compute_fn
            .invoke(&InvocationArgs {
                inputs: &input_refs,
                options: args.options,
            })?
            .unwrap_array()?;

        chunks.push(result);
        idx += chunk.len();
    }

    let return_dtype = compute_fn.return_dtype(args)?;
    Ok(Some(
        ChunkedArray::try_new(chunks, return_dtype)?
            .into_array()
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DType, Nullability};

    use crate::arrays::{BoolArray, BooleanBuffer, ChunkedArray};
    use crate::canonical::ToCanonical;
    use crate::compute::{BooleanOperator, boolean};

    #[test]
    fn test_bin_bool_chunked() {
        let arr0 = BoolArray::from_iter(vec![true, false]).to_array();
        let arr1 = BoolArray::from_iter(vec![false, false, true]).to_array();
        let chunked1 =
            ChunkedArray::try_new(vec![arr0, arr1], DType::Bool(Nullability::NonNullable)).unwrap();

        let arr2 = BoolArray::from_iter(vec![Some(false), Some(true)]).to_array();
        let arr3 = BoolArray::from_iter(vec![Some(false), None, Some(false)]).to_array();
        let chunked2 =
            ChunkedArray::try_new(vec![arr2, arr3], DType::Bool(Nullability::Nullable)).unwrap();

        let result = boolean(chunked1.as_ref(), chunked2.as_ref(), BooleanOperator::Or)
            .unwrap()
            .to_bool()
            .unwrap();
        assert_eq!(
            result.boolean_buffer(),
            &BooleanBuffer::from_iter([true, true, false, false, true])
        );
    }
}
