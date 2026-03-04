// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA support for Vortex arrays.

use std::process::Command;

use tracing::info;

pub mod arrow;
mod canonical;
mod device_buffer;
mod device_read_at;
pub mod dynamic_dispatch;
pub mod executor;
mod kernel;
pub mod layout;
mod pinned;
mod pooled_read_at;
mod session;
mod stream;
mod stream_pool;

pub use arrow::ExportDeviceArray;
pub use canonical::CanonicalCudaExt;
pub use device_buffer::CudaBufferExt;
pub use device_buffer::CudaDeviceBuffer;
pub use device_read_at::CopyDeviceReadAt;
pub use executor::CudaExecutionCtx;
pub use executor::CudaKernelEvents;
use kernel::ALPExecutor;
use kernel::BitPackedExecutor;
use kernel::ConstantNumericExecutor;
use kernel::DateTimePartsExecutor;
use kernel::DecimalBytePartsExecutor;
pub use kernel::DefaultLaunchStrategy;
use kernel::DictExecutor;
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
pub use kernel::transpose_patches;
pub use kernel::zstd_kernel_prepare;
pub use pinned::PinnedByteBufferPool;
pub use pinned::PinnedPoolStats;
pub use pinned::PooledPinnedBuffer;
pub use pooled_read_at::PooledFileReadAt;
pub use pooled_read_at::PooledObjectStoreReadAt;
pub use session::CudaSession;
pub use session::CudaSessionExt;
pub use stream::VortexCudaStream;
pub use stream_pool::VortexCudaStreamPool;
use vortex::array::arrays::ConstantVTable;
use vortex::array::arrays::DictVTable;
use vortex::array::arrays::FilterVTable;
use vortex::array::arrays::SharedVTable;
use vortex::array::arrays::SliceVTable;
use vortex::encodings::alp::ALPVTable;
use vortex::encodings::datetime_parts::DateTimePartsVTable;
use vortex::encodings::decimal_byte_parts::DecimalBytePartsVTable;
use vortex::encodings::fastlanes::BitPackedVTable;
use vortex::encodings::fastlanes::FoRVTable;
use vortex::encodings::runend::RunEndVTable;
use vortex::encodings::sequence::SequenceVTable;
use vortex::encodings::zigzag::ZigZagVTable;
#[cfg(feature = "unstable_encodings")]
use vortex::encodings::zstd::ZstdBuffersVTable;
use vortex::encodings::zstd::ZstdVTable;
pub use vortex_nvcomp as nvcomp;

use crate::kernel::SequenceExecutor;
use crate::kernel::SliceExecutor;

/// Checks if CUDA is available on the system by looking for nvcc.
pub fn cuda_available() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Registers CUDA kernels.
pub fn initialize_cuda(session: &CudaSession) {
    info!("Registering CUDA kernels");
    session.register_kernel(ALPVTable::ID, &ALPExecutor);
    session.register_kernel(BitPackedVTable::ID, &BitPackedExecutor);
    session.register_kernel(ConstantVTable::ID, &ConstantNumericExecutor);
    session.register_kernel(DateTimePartsVTable::ID, &DateTimePartsExecutor);
    session.register_kernel(DecimalBytePartsVTable::ID, &DecimalBytePartsExecutor);
    session.register_kernel(DictVTable::ID, &DictExecutor);
    session.register_kernel(SharedVTable::ID, &SharedExecutor);
    session.register_kernel(FoRVTable::ID, &FoRExecutor);
    session.register_kernel(RunEndVTable::ID, &RunEndExecutor);
    session.register_kernel(SequenceVTable::ID, &SequenceExecutor);
    session.register_kernel(ZigZagVTable::ID, &ZigZagExecutor);
    session.register_kernel(ZstdVTable::ID, &ZstdExecutor);
    #[cfg(feature = "unstable_encodings")]
    session.register_kernel(ZstdBuffersVTable::ID, &ZstdBuffersExecutor);

    // Operation kernels
    session.register_kernel(FilterVTable::ID, &FilterExecutor);
    session.register_kernel(SliceVTable::ID, &SliceExecutor);
}
