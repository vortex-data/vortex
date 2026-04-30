// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::use_debug)]

mod array;
pub mod compress;
pub mod error;
pub mod fsst_like;

// File module only available for native builds (requires vortex-file which uses tokio)
#[cfg(not(target_arch = "wasm32"))]
pub mod file;

// GPU fuzzer module (only available when cuda feature is enabled)
#[cfg(feature = "cuda")]
pub mod gpu;
pub use array::Action;
pub use array::CompressorStrategy;
pub use array::ExpectedValue;
pub use array::FuzzArrayAction;
pub use array::run_fuzz_action;
pub use array::sort_canonical_array;
pub use compress::FuzzCompressRoundtrip;
pub use compress::run_compress_roundtrip;
#[cfg(not(target_arch = "wasm32"))]
pub use file::FuzzFileAction;
pub use fsst_like::FuzzFsstLike;
pub use fsst_like::run_fsst_like_fuzz;
#[cfg(feature = "cuda")]
pub use gpu::FuzzCompressGpu;
#[cfg(feature = "cuda")]
pub use gpu::run_compress_gpu;

pub const FUZZ_ARRAY_MAX_LEN: usize = 16_384;

// Runtime initialization - platform-specific
#[cfg(not(target_arch = "wasm32"))]
mod native_runtime {
    use std::sync::LazyLock;

    use vortex::VortexSessionDefault;
    use vortex_io::runtime::BlockingRuntime;
    use vortex_io::runtime::current::CurrentThreadRuntime;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    pub static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);
    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        #[allow(unused_mut)]
        let mut session = VortexSession::default().with_handle(RUNTIME.handle());
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        // Even if the CUDA feature is enabled we need to check at
        // runtime whether CUDA is available in the current environment.
        if vortex_cuda::cuda_available() {
            use vortex_cuda::CudaSessionExt;
            session = session.with::<vortex_cuda::CudaSession>();
            vortex_cuda::initialize_cuda(&session.cuda_session());
        }
        session
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub use native_runtime::RUNTIME;
#[cfg(not(target_arch = "wasm32"))]
pub use native_runtime::SESSION;

// target_os = "unknown" matches wasm32-unknown-unknown (browser), excluding WASI targets
// where wasm-bindgen's JS interop is not available.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
mod wasm_runtime {
    use std::sync::LazyLock;

    use vortex::VortexSessionDefault;
    use vortex_io::runtime::wasm::WasmRuntime;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    pub static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::default().with_handle(WasmRuntime::handle()));
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use wasm_runtime::SESSION;

/// WASI targets get a session without an async runtime handle since wasm-bindgen
/// is not available. The fuzz workloads are synchronous, so this is sufficient.
#[cfg(all(target_arch = "wasm32", not(target_os = "unknown")))]
mod wasi_runtime {
    use std::sync::LazyLock;

    use vortex::VortexSessionDefault;
    use vortex_session::VortexSession;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);
}

#[cfg(all(target_arch = "wasm32", not(target_os = "unknown")))]
pub use wasi_runtime::SESSION;
