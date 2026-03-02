# Powershell script
bindgen `
    --allowlist-type cudaVideo.* `
    --allowlist-type cuvid.* `
    --allowlist-type CUVID.* `
    --allowlist-function cuvid.* `
    --allowlist-var \[IPBS\]_VOP `
    --allowlist-var cuvid.* `
    --blocklist-file .*cuda\.h `
    --blocklist-file .*std.*\.h `
    --must-use-type CUresult `
    --must-use-type cuvidDecodeStatus `
    `
    --default-enum-style=rust `
    --no-doc-comments `
    --with-derive-default `
    --with-derive-eq `
    --with-derive-hash `
    --with-derive-ord `
    --use-core `
    --merge-extern-blocks `
    --sort-semantically `
    --output cuviddec.rs ..\headers\cuviddec.h `
    -- -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include"

bindgen `
    --allowlist-type CU.* `
    --allowlist-type cudaVideo.* `
    --allowlist-type cudaAudio.* `
    --allowlist-type HEVC.* `
    --allowlist-function cuvid.* `
    --allowlist-var MAX_CLOCK_TS `
    --blocklist-file .*cuda\.h `
    --blocklist-file .*std.*\.h `
    --blocklist-file .*cuviddec\.h `
    --must-use-type CUresult `
    `
    --default-enum-style=rust `
    --no-doc-comments `
    --with-derive-default `
    --with-derive-eq `
    --with-derive-hash `
    --with-derive-ord `
    --use-core `
    --merge-extern-blocks `
    --sort-semantically `
    --output nvcuvid.rs ..\headers\nvcuvid.h `
    -- -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include"

bindgen `
    --allowlist-type NVENC.* `
    --allowlist-type NV_ENC.* `
    --allowlist-type NV_ENCODE.* `
    --allowlist-type GUID `
    --allowlist-type PENV.* `
    --allowlist-function NvEncodeAPI.* `
    --allowlist-function NvEnc.* `
    --allowlist-var NVENC.* `
    --allowlist-var NV_ENC.* `
    --allowlist-var NV_MAX.* `
    --blocklist-item NV_ENC_\w+_GUID `
    --blocklist-file .*cuda\.h `
    --blocklist-file .*std.*\.h `
    --must-use-type NVENCSTATUS `
    `
    --default-enum-style=rust `
    --no-doc-comments `
    --with-derive-default `
    --with-derive-eq `
    --with-derive-hash `
    --with-derive-ord `
    --use-core `
    --merge-extern-blocks `
    --sort-semantically `
    --output nvEncodeAPI.rs ..\headers\nvEncodeAPI.h `
    -- -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include"

# Additional preludes to make sure the bindings compile.
$(
    "use cudarc::driver::sys::*;"
    Get-Content cuviddec.rs -Raw
) | Set-Content cuviddec.rs
$(
    "use super::cuviddec::*;"
    "use cudarc::driver::sys::*;"
    Get-Content nvcuvid.rs -Raw
) | Set-Content nvcuvid.rs
$(
    "pub use super::super::version::*;"
    "pub use super::super::guid::*;"
    Get-Content nvEncodeAPI.rs -Raw
) | Set-Content nvEncodeAPI.rs
