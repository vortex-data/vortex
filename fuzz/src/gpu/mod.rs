// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! GPU fuzzer module for testing CUDA decompression.
//!
//! This module generates arbitrary instances of GPU-supported compressed encodings,
//! then verifies that GPU decompression produces the same results as CPU decompression.

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ArbitraryDictArray;
use vortex_dtype::Nullability;
use vortex_dtype::PType;

use crate::error::Backtrace;
use crate::error::VortexFuzzError;
use crate::error::VortexFuzzResult;

/// Which GPU-supported encoding to generate.
#[derive(Debug, Clone, Copy)]
pub enum GpuEncodingKind {
    /// Dictionary encoding with GPU take support.
    Dict,
}

impl<'a> Arbitrary<'a> for GpuEncodingKind {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        // Currently only Dict is supported
        match u.int_in_range(0..=0)? {
            0 => Ok(GpuEncodingKind::Dict),
            _ => unreachable!(),
        }
    }
}

/// Input for the GPU decompression fuzzer.
#[derive(Debug)]
pub struct FuzzCompressGpu {
    pub array: ArrayRef,
}

impl<'a> Arbitrary<'a> for FuzzCompressGpu {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let kind: GpuEncodingKind = u.arbitrary()?;

        let array = match kind {
            GpuEncodingKind::Dict => {
                // Dict already has Arbitrary support, use primitive values for GPU compatibility
                let dtype = arbitrary_gpu_primitive_dtype(u)?;
                ArbitraryDictArray::with_dtype(u, &dtype, None)?
                    .0
                    .into_array()
            }
        };

        Ok(FuzzCompressGpu { array })
    }
}

/// Generate a random primitive DType suitable for GPU operations.
fn arbitrary_gpu_primitive_dtype(u: &mut Unstructured) -> Result<vortex_dtype::DType> {
    let nullability: Nullability = u.arbitrary()?;
    let ptype = match u.int_in_range(0..=9)? {
        0 => PType::U8,
        1 => PType::U16,
        2 => PType::U32,
        3 => PType::U64,
        4 => PType::I8,
        5 => PType::I16,
        6 => PType::I32,
        7 => PType::I64,
        8 => PType::F32,
        9 => PType::F64,
        _ => unreachable!(),
    };
    Ok(vortex_dtype::DType::Primitive(ptype, nullability))
}

/// Run the GPU decompression fuzzer.
///
/// This function:
/// 1. Decompresses the array on CPU (reference)
/// 2. Decompresses the array on GPU
/// 3. Copies GPU result back to host using `CanonicalCudaExt::to_host`
/// 4. Compares the results
///
/// Returns:
/// - `Ok(true)` - test passed, keep in corpus
/// - `Ok(false)` - test skipped (e.g., no CUDA), reject from corpus
/// - `Err(_)` - a bug was found
#[allow(clippy::result_large_err)]
pub async fn run_compress_gpu(fuzz: FuzzCompressGpu) -> VortexFuzzResult<bool> {
    use vortex::VortexSessionDefault;
    use vortex::session::VortexSession;
    use vortex_cuda::CanonicalCudaExt;
    use vortex_cuda::CudaSession;
    use vortex_cuda::executor::CudaArrayExt;
    use vortex_error::VortexExpect;

    // Runtime check - skip if CUDA is not available
    if !vortex_cuda::cuda_available() {
        return Ok(false);
    }

    let FuzzCompressGpu { array } = fuzz;

    // Store original properties for error reporting
    let original_len = array.len();

    // 1. CPU decompression (reference)
    let cpu_canonical = match array.to_canonical() {
        Ok(c) => c,
        Err(e) => {
            return Err(VortexFuzzError::VortexError(e, Backtrace::capture()));
        }
    };

    // 2. Create CUDA execution context
    let session = VortexSession::default();

    let mut cuda_ctx =
        CudaSession::create_execution_ctx(&session).vortex_expect("cannot create session");

    // 3. GPU decompression
    let gpu_canonical = match array.clone().execute_cuda(&mut cuda_ctx).await {
        Ok(c) => c,
        Err(e) => {
            return Err(VortexFuzzError::VortexError(e, Backtrace::capture()));
        }
    };

    // 4. Copy GPU result back to host using CanonicalCudaExt
    let gpu_host_canonical = match gpu_canonical.to_host().await {
        Ok(c) => c,
        Err(e) => {
            return Err(VortexFuzzError::VortexError(e, Backtrace::capture()));
        }
    };

    // 5. Compare canonicals
    let cpu_array = cpu_canonical.into_array();
    let gpu_array = gpu_host_canonical.into_array();

    // Verify dtype is preserved
    if cpu_array.dtype() != gpu_array.dtype() {
        return Err(VortexFuzzError::DTypeMismatch(
            cpu_array,
            gpu_array,
            0,
            Backtrace::capture(),
        ));
    }

    // Verify length is preserved
    if original_len != gpu_array.len() {
        return Err(VortexFuzzError::LengthMismatch(
            original_len,
            gpu_array.len(),
            array,
            gpu_array,
            0,
            Backtrace::capture(),
        ));
    }

    // Compare element by element
    for i in 0..original_len {
        let cpu_scalar = cpu_array
            .scalar_at(i)
            .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?;
        let gpu_scalar = gpu_array
            .scalar_at(i)
            .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?;

        if cpu_scalar != gpu_scalar {
            return Err(VortexFuzzError::ArrayNotEqual(
                cpu_scalar,
                gpu_scalar,
                i,
                cpu_array,
                gpu_array,
                0,
                Backtrace::capture(),
            ));
        }
    }

    Ok(true)
}
