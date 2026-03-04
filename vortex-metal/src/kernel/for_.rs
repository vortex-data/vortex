// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::PrimitiveArrayParts;
use vortex::array::match_each_native_simd_ptype;
use vortex::dtype::NativePType;
use vortex::encodings::fastlanes::FoRArray;
use vortex::encodings::fastlanes::FoRVTable;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::MetalArrayExt;
use crate::MetalBufferExt;
use crate::MetalExecute;
use crate::MetalExecutionCtx;

/// Metal decoder for frame-of-reference.
#[derive(Debug)]
pub(crate) struct FoRExecutor;

impl FoRExecutor {
    fn try_specialize(array: ArrayRef) -> Option<FoRArray> {
        array.try_into::<FoRVTable>().ok()
    }
}

impl MetalExecute for FoRExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    fn execute(&self, array: ArrayRef, ctx: &mut MetalExecutionCtx) -> VortexResult<Canonical> {
        let array = Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected FoRArray"))?;

        match_each_native_simd_ptype!(array.ptype(), |P| { decode_for::<P>(array, ctx) })
    }
}

#[instrument(skip_all)]
fn decode_for<P>(array: FoRArray, ctx: &mut MetalExecutionCtx) -> VortexResult<Canonical>
where
    P: NativePType + Send + Sync + 'static,
{
    let array_len = array.encoded().len();
    vortex_ensure!(array_len > 0, "FoR encoded array must not be empty");

    let reference: P = array
        .reference_scalar()
        .as_primitive()
        .as_::<P>()
        .vortex_expect("Cannot have a null reference");

    // Execute child and ensure on device
    let canonical = array.encoded().clone().execute_metal(ctx)?;
    let primitive = canonical.into_primitive();
    let PrimitiveArrayParts {
        buffer, validity, ..
    } = primitive.into_parts();

    let device_buffer = ctx.ensure_on_device(buffer)?;

    // Get the Metal buffer
    let metal_buffer = device_buffer.metal_buffer()?;

    // Load kernel function
    let kernel_name = format!("for_{}", P::PTYPE.to_string().to_lowercase());
    let pipeline = ctx.load_pipeline("for", &kernel_name)?;

    // Prepare constant data
    let reference_bytes = bytemuck_ref_to_bytes(&reference);
    let array_len_u64 = array_len as u64;
    let len_bytes = array_len_u64.to_le_bytes();

    // Dispatch the kernel
    ctx.dispatch_kernel(
        &pipeline,
        &[(metal_buffer.metal_buffer(), metal_buffer.offset())],
        &[(reference_bytes, 1), (&len_bytes, 2)],
        array_len,
    )?;

    // Commit and wait for completion
    ctx.commit_and_wait()?;

    // Build result - in-place reuses the same buffer
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        device_buffer,
        P::PTYPE,
        validity,
    )))
}

/// Convert a reference to a NativePType to a byte slice.
fn bytemuck_ref_to_bytes<P: NativePType>(val: &P) -> &[u8] {
    // SAFETY: All NativePType types are Plain Old Data and can be safely
    // reinterpreted as bytes.
    unsafe { std::slice::from_raw_parts(std::ptr::from_ref(val).cast::<u8>(), size_of::<P>()) }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity::NonNullable;
    use vortex::buffer::Buffer;
    use vortex::dtype::NativePType;
    use vortex::encodings::fastlanes::FoRArray;
    use vortex::error::VortexResult;
    use vortex::scalar::Scalar;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalMetalExt;
    use crate::MetalSession;

    fn make_for_array<T: NativePType + Into<Scalar>>(input_data: Vec<T>, reference: T) -> FoRArray {
        #[allow(clippy::unwrap_used)]
        FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data), NonNullable).into_array(),
            reference.into(),
        )
        .unwrap()
    }

    #[rstest]
    #[case::u8(make_for_array((0..2050).map(|i| (i % 246) as u8).collect(), 10u8))]
    #[case::u16(make_for_array((0..2050).map(|i| (i % 2050) as u16).collect(), 1000u16))]
    #[case::u32(make_for_array((0..2050).map(|i| (i % 2050) as u32).collect(), 100000u32))]
    #[case::u64(make_for_array((0..2050).map(|i| (i % 2050) as u64).collect(), 1000000u64))]
    #[test]
    fn test_metal_for_decompression(#[case] for_array: FoRArray) -> VortexResult<()> {
        let session = MetalSession::new()?;
        let mut ctx = session.create_execution_ctx(&VortexSession::empty())?;

        let cpu_result = for_array.to_canonical()?;

        let gpu_result = FoRExecutor
            .execute(for_array.to_array(), &mut ctx)?
            .into_host()?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }

    #[rstest]
    #[case::i8(make_for_array((0i8..100i8).cycle().take(2050).map(|i| i - 50).collect(), 10i8))]
    #[case::i16(make_for_array((0i16..2050i16).map(|i| i - 1000).collect(), 1000i16))]
    #[case::i32(make_for_array((0i32..2050i32).map(|i| i - 1000).collect(), 100000i32))]
    #[case::i64(make_for_array((0i64..2050i64).map(|i| i - 1000).collect(), 1000000i64))]
    #[test]
    fn test_metal_for_signed_decompression(#[case] for_array: FoRArray) -> VortexResult<()> {
        let session = MetalSession::new()?;
        let mut ctx = session.create_execution_ctx(&VortexSession::empty())?;

        let cpu_result = for_array.to_canonical()?;

        let gpu_result = FoRExecutor
            .execute(for_array.to_array(), &mut ctx)?
            .into_host()?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }
}
