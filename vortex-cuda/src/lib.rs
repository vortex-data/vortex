// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA support for Vortex arrays.

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use tracing::info;

pub mod arrow;
mod canonical;
mod cub;
mod device_buffer;
mod device_read_at;
pub mod dynamic_dispatch;
pub mod executor;
pub mod hybrid_dispatch;
mod kernel;
pub mod layout;
mod pinned;
mod pooled_read_at;
mod session;
mod stream;
mod stream_pool;

pub use arrow::ArrowDeviceArrayWithSchema;
pub use arrow::DeviceArrayExt;
pub use arrow::DeviceArrayStreamExt;
pub use arrow::ExportDeviceArray;
pub use canonical::CanonicalCudaExt;
pub use device_buffer::CudaBufferExt;
pub use device_buffer::CudaDeviceBuffer;
pub use device_read_at::CopyDeviceReadAt;
pub use executor::CudaDispatchMode;
pub use executor::CudaExecutionCtx;
pub use executor::CudaKernelEvents;
use kernel::ALPExecutor;
use kernel::BitPackedExecutor;
use kernel::ConstantNumericExecutor;
use kernel::DateTimePartsExecutor;
use kernel::DecimalBytePartsExecutor;
pub use kernel::DefaultLaunchStrategy;
use kernel::DictExecutor;
use kernel::FSSTExecutor;
use kernel::FilterExecutor;
use kernel::FoRExecutor;
pub use kernel::LaunchStrategy;
use kernel::RunEndExecutor;
use kernel::SharedExecutor;
pub use kernel::TracingLaunchStrategy;
use kernel::ZigZagExecutor;
#[cfg(feature = "unstable_encodings")]
use kernel::ZstdBuffersExecutor;
use kernel::ZstdExecutor;
pub use kernel::ZstdKernelPrep;
pub use kernel::zstd_kernel_prepare;
pub use pinned::PinnedByteBufferPool;
pub use pinned::PinnedPoolStats;
pub use pinned::PooledPinnedBuffer;
pub use pooled_read_at::PooledByteBufferReadAt;
pub use pooled_read_at::PooledFileReadAt;
pub use pooled_read_at::PooledObjectStoreReadAt;
pub use session::CudaSession;
pub use session::CudaSessionExt;
pub use stream::VortexCudaStream;
pub use stream_pool::VortexCudaStreamPool;
use vortex::array::ArrayVTable;
use vortex::array::arrays::Constant;
use vortex::array::arrays::Dict;
use vortex::array::arrays::Filter;
use vortex::array::arrays::Shared;
use vortex::array::arrays::Slice;
use vortex::encodings::alp::ALP;
use vortex::encodings::datetime_parts::DateTimeParts;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fsst::FSST;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::sequence::Sequence;
use vortex::encodings::zigzag::ZigZag;
use vortex::encodings::zstd::Zstd;
#[cfg(feature = "unstable_encodings")]
use vortex::encodings::zstd::ZstdBuffers;
#[cfg(test)]
use vortex_cuda_macros::test;
pub use vortex_nvcomp as nvcomp;

use crate::kernel::SequenceExecutor;
use crate::kernel::SliceExecutor;

/// Checks if a CUDA driver and at least one CUDA device are available.
///
/// cudarc loads `libcuda` lazily and panics if the driver library is absent, so we first probe
/// for it with cudarc's own `is_culib_present`. Creating the context then fails gracefully with
/// `Err`, rather than panicking, when the driver is present but no usable device is.
pub fn cuda_available() -> bool {
    // SAFETY: `is_culib_present` only tries to dlopen the driver library to test for its
    // presence; it upholds no further invariants.
    let driver_present = unsafe { sys::is_culib_present() };
    driver_present && CudaContext::new(0).is_ok()
}

/// Registers CUDA kernels.
pub fn initialize_cuda(session: &CudaSession) {
    info!("Registering CUDA kernels");
    session.register_kernel(ALP.id(), &ALPExecutor);
    session.register_kernel(BitPacked.id(), &BitPackedExecutor);
    session.register_kernel(Constant.id(), &ConstantNumericExecutor);
    session.register_kernel(DateTimeParts.id(), &DateTimePartsExecutor);
    session.register_kernel(DecimalByteParts.id(), &DecimalBytePartsExecutor);
    session.register_kernel(Dict.id(), &DictExecutor);
    session.register_kernel(Shared.id(), &SharedExecutor);
    session.register_kernel(FoR.id(), &FoRExecutor);
    session.register_kernel(FSST.id(), &FSSTExecutor);
    session.register_kernel(RunEnd.id(), &RunEndExecutor);
    session.register_kernel(Sequence.id(), &SequenceExecutor);
    session.register_kernel(ZigZag.id(), &ZigZagExecutor);
    session.register_kernel(Zstd.id(), &ZstdExecutor);
    #[cfg(feature = "unstable_encodings")]
    session.register_kernel(ZstdBuffers.id(), &ZstdBuffersExecutor);

    // Operation kernels
    session.register_kernel(Filter.id(), &FilterExecutor);
    session.register_kernel(Slice.id(), &SliceExecutor);
}

/// Builds a fresh [`VortexSession`](vortex::session::VortexSession) with all array session
/// variables plus a default [`CudaSession`], for use in CUDA tests and benchmarks.
///
/// Each call returns an independent session with its own CUDA context and stream pool, matching
/// the per-test isolation that lazily-initialized sessions previously provided.
///
/// # Panics
///
/// Panics if CUDA device 0 cannot be initialized (the same contract as [`CudaSession::default`]).
#[cfg(any(test, feature = "_test-harness"))]
pub fn cuda_session() -> vortex::session::VortexSession {
    vortex::array::array_session().with::<CudaSession>()
}
