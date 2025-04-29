use vortex_error::{VortexExpect, VortexResult};

use crate::Array;
use crate::arrays::ChunkedArray;
use crate::compute::{ComputeFn, InvocationArgs, Output, slice};

impl ChunkedArray {
    /// Invoke an element-wise compute function over a chunked array.
    pub(in crate::arrays::chunked) fn invoke_elementwise(
        &self,
        compute_fn: &ComputeFn,
        args: &InvocationArgs,
    ) -> VortexResult<Option<Output>> {
        assert!(compute_fn.is_elementwise());

        // We know the first input argument is ourselves, if there is one other input argument
        // then we can delegate the compute function. Otherwise, we return None.
        if args.inputs.len() != 2 || args.inputs[1].array().is_none() {
            return Ok(None);
        }
        let rhs = args.inputs[1].array().vortex_expect("checked already");

        let mut idx = 0;
        let mut chunks = Vec::with_capacity(self.nchunks());

        for chunk in self.non_empty_chunks() {
            let sliced = slice(rhs, idx, idx + chunk.len())?;

            // Delegate the compute kernel to the chunk.
            let result = compute_fn
                .invoke(&InvocationArgs {
                    inputs: &[chunk.as_ref().into(), sliced.as_ref().into()],
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
}
