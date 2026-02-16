// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA support for Vortex arrays.

use std::process::Command;

use tracing::info;

pub mod arrow;
mod canonical;
mod device_buffer;
pub mod dynamic_dispatch;
pub mod executor;
mod host_to_device_allocator;
mod kernel;
pub mod layout;
mod session;
mod stream;
mod stream_pool;

pub use arrow::ExportDeviceArray;
pub use canonical::CanonicalCudaExt;
pub use device_buffer::CudaBufferExt;
pub use device_buffer::CudaDeviceBuffer;
pub use executor::CudaExecutionCtx;
pub use executor::CudaKernelEvents;
pub use host_to_device_allocator::CopyDeviceReadAt;
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
pub use kernel::zstd_kernel_prepare;
pub use session::CudaSession;
pub use session::CudaSessionExt;
pub use stream_pool::VortexCudaStreamPool;
use vortex_alp::ALPVTable;
use vortex_array::arrays::ConstantVTable;
use vortex_array::arrays::DictVTable;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::SharedVTable;
use vortex_array::arrays::SliceVTable;
use vortex_datetime_parts::DateTimePartsVTable;
use vortex_decimal_byte_parts::DecimalBytePartsVTable;
use vortex_fastlanes::BitPackedVTable;
use vortex_fastlanes::FoRVTable;
pub use vortex_nvcomp as nvcomp;
use vortex_runend::RunEndVTable;
use vortex_sequence::SequenceVTable;
use vortex_zigzag::ZigZagVTable;
#[cfg(feature = "unstable_encodings")]
use vortex_zstd::ZstdBuffersVTable;
use vortex_zstd::ZstdVTable;

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
