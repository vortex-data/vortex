use std::path::PathBuf;

/// <a href="https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/
/// nvenc-video-encoder-api-prog-guide/index.html#basic-encoding-flow"></a>
#[cfg(unix)]
const NVENC_LIB: (&str, &str) = ("nvidia-encode", "libnvidia-encode.so");
#[cfg(windows)]
const NVENC_LIB: (&str, &str) = ("nvencodeapi", "nvEncodeAPI.lib");

/// <a href="https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/
/// nvdec-video-decoder-api-prog-guide/index.html#
/// using-nvidia-video-decoder-nvdecode-api"></a>
#[cfg(unix)]
const NVDEC_LIB: (&str, &str) = ("nvcuvid", "libnvcuvid.so");
#[cfg(windows)]
const NVDEC_LIB: (&str, &str) = ("nvcuvid", "nvcuvid.lib");

/// Environment variables which might specify path to the libraries.
///
/// - <https://github.com/coreylowman/cudarc/blob/main/build.rs>
/// - <https://github.com/rust-av/nvidia-video-codec-rs/blob/master/nvidia-video-codec-sys/build.rs>
const ENVIRONMENT_VARIABLES: [&str; 5] = [
    "CUDA_PATH",
    "CUDA_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR",
    "NVIDIA_VIDEO_CODEC_SDK_PATH",
    "NVIDIA_VIDEO_CODEC_INCLUDE_PATH",
];

/// Candidate paths which do not require an environment variable.
///
/// - <https://github.com/coreylowman/cudarc/blob/main/build.rs>
/// - <https://github.com/ViliamVadocz/nvidia-video-codec-sdk/issues/13>
const ROOT_CANDIDATES: [&str; 7] = [
    "/usr",
    "/usr/local/cuda",
    "/opt/cuda",
    "/usr/lib/cuda",
    "C:/Program Files/NVIDIA GPU Computing Toolkit",
    "C:/CUDA",
    "/usr/include/nvidia-sdk",
];

const LIBRARY_CANDIDATES: [&str; 11] = [
    "",
    "lib",
    "lib/x64",
    "lib/Win32",
    "lib/x86_64",
    "lib/x86_64-linux-gnu",
    "lib64",
    "lib64/stubs",
    "targets/x86_64-linux",
    "targets/x86_64-linux/lib",
    "targets/x86_64-linux/lib/stubs",
];

fn main() {
    if cfg!(feature = "ci-check") {
        return;
    }
    rerun_if_changed();

    // Link to libraries if found. On systems without NVIDIA SDK (e.g. Mac dev
    // machines), we skip linking — the crate will still compile but panic at
    // runtime when NVENC functions are called.
    if let Some(path) = library_candidates().next() {
        println!("cargo:rustc-link-search={}", path.display());
        println!("cargo:rustc-link-lib={}", NVENC_LIB.0);
        println!("cargo:rustc-link-lib={}", NVDEC_LIB.0);
    } else {
        // No NVIDIA libs found — emit a warning but don't fail the build.
        println!(
            "cargo:warning=NVIDIA Video Codec SDK libraries not found. NVENC/NVDEC will not be \
             available at runtime."
        );
    }
}

/// Rerun the build script if any of the listed environment variables changes.
fn rerun_if_changed() {
    for var in ENVIRONMENT_VARIABLES {
        println!("cargo:rerun-if-env-changed={var}");
    }
}

/// Look for directories which contain the library files.
fn library_candidates() -> impl Iterator<Item = PathBuf> {
    ENVIRONMENT_VARIABLES
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .chain(ROOT_CANDIDATES.into_iter().map(Into::into))
        .flat_map(|root| {
            let root = PathBuf::from(root);
            LIBRARY_CANDIDATES
                .into_iter()
                .map(move |lib_path| root.join(lib_path))
                .filter(|path| path.join(NVENC_LIB.1).is_file() && path.join(NVDEC_LIB.1).is_file())
        })
}
