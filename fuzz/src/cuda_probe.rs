// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::thread;

use cudarc::driver::CudaContext;
use cudarc::driver::result;
use cudarc::driver::result::DriverError;
use cudarc::driver::sys;

#[derive(Clone, Copy)]
pub enum CudaProbeMode {
    ReadOnly,
    Full,
}

impl CudaProbeMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::Full => "full",
        }
    }
}

pub fn log_cuda_driver_probe(phase: &str, mode: CudaProbeMode) {
    eprintln!(
        "===== CUDA driver probe ({phase}, mode={}) =====",
        mode.as_str()
    );
    eprintln!("probe_pid={}", std::process::id());
    eprintln!(
        "probe_thread={}",
        thread::current().name().unwrap_or("<unnamed>")
    );
    eprintln!("probe_cuda_available()={}", vortex_cuda::cuda_available());

    match result::init() {
        Ok(()) => eprintln!("cuInit=ok"),
        Err(err) => {
            log_driver_error("cuInit", err);
            return;
        }
    }

    log_current_context("before-probe");

    let device_count = match result::device::get_count() {
        Ok(count) => {
            eprintln!("cuDeviceGetCount={count}");
            count
        }
        Err(err) => {
            log_driver_error("cuDeviceGetCount", err);
            return;
        }
    };

    if device_count <= 0 {
        eprintln!("No CUDA devices reported by the driver");
        return;
    }

    for ordinal in 0..device_count {
        log_device_overview(ordinal);
    }

    if matches!(mode, CudaProbeMode::Full) {
        log_full_device_zero_probe();
    }
}

fn log_device_overview(ordinal: i32) {
    eprintln!("-- device {ordinal} --");

    let device = match result::device::get(ordinal) {
        Ok(device) => {
            eprintln!("cuDeviceGet({ordinal})=ok");
            device
        }
        Err(err) => {
            log_driver_error(&format!("cuDeviceGet({ordinal})"), err);
            return;
        }
    };

    log_value("device_name", result::device::get_name(device));
    match result::device::get_uuid(device) {
        Ok(uuid) => eprintln!("device_uuid={}", format_uuid(uuid)),
        Err(err) => log_driver_error("cuDeviceGetUuid", err),
    }
    match unsafe { result::device::total_mem(device) } {
        Ok(bytes) => eprintln!("device_total_mem_bytes={bytes}"),
        Err(err) => log_driver_error("cuDeviceTotalMem", err),
    }

    log_device_attr(
        device,
        "compute_capability_major",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    );
    log_device_attr(
        device,
        "compute_capability_minor",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    );
    match get_device_attr(
        device,
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
    ) {
        Ok(value) => eprintln!("compute_mode={} ({})", value, compute_mode_name(value)),
        Err(err) => log_driver_error("cuDeviceGetAttribute(compute_mode)", err),
    }
    log_device_attr(
        device,
        "multiprocessor_count",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    );
    log_device_attr(
        device,
        "max_threads_per_block",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    );
    log_device_attr(
        device,
        "max_threads_per_multiprocessor",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
    );
    log_device_attr(
        device,
        "unified_addressing",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
    );
    log_device_attr(
        device,
        "managed_memory",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
    );
    log_device_attr(
        device,
        "concurrent_managed_access",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
    );
    log_device_attr(
        device,
        "compute_preemption_supported",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
    );
    log_device_attr(
        device,
        "memory_pools_supported",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
    );
    log_device_attr(
        device,
        "mps_enabled",
        sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MPS_ENABLED,
    );
    log_primary_context_state(device, "before-retain");
}

fn log_full_device_zero_probe() {
    eprintln!("-- full device 0 probe --");

    let device = match result::device::get(0) {
        Ok(device) => device,
        Err(err) => {
            log_driver_error("cuDeviceGet(0)", err);
            return;
        }
    };

    match unsafe { result::primary_ctx::retain(device) } {
        Ok(primary_ctx) => {
            eprintln!(
                "cuDevicePrimaryCtxRetain=ok ctx={}",
                format_context(primary_ctx)
            );
            log_primary_context_state(device, "after-retain");
            log_current_context("after-retain-before-bind");

            match unsafe { result::ctx::set_current(primary_ctx) } {
                Ok(()) => {
                    eprintln!("cuCtxSetCurrent(primary_ctx)=ok");
                    log_current_context("after-bind-primary");
                    match result::mem_get_info() {
                        Ok((free, total)) => {
                            eprintln!(
                                "cuMemGetInfo(primary_ctx)=free_bytes={free} total_bytes={total}"
                            );
                        }
                        Err(err) => log_driver_error("cuMemGetInfo(primary_ctx)", err),
                    }
                }
                Err(err) => log_driver_error("cuCtxSetCurrent(primary_ctx)", err),
            }

            match unsafe { result::ctx::set_current(std::ptr::null_mut()) } {
                Ok(()) => eprintln!("cuCtxSetCurrent(NULL)=ok"),
                Err(err) => log_driver_error("cuCtxSetCurrent(NULL)", err),
            }

            match unsafe { result::primary_ctx::release(device) } {
                Ok(()) => eprintln!("cuDevicePrimaryCtxRelease=ok"),
                Err(err) => log_driver_error("cuDevicePrimaryCtxRelease", err),
            }
            log_primary_context_state(device, "after-release");
            log_current_context("after-release");
        }
        Err(err) => log_driver_error("cuDevicePrimaryCtxRetain", err),
    }

    match CudaContext::new(0) {
        Ok(context) => {
            eprintln!("CudaContext::new(0)=ok");
            log_context_details("primary-context-wrapper", &context);
            drop(context);
            log_current_context("after-drop-primary-context-wrapper");
        }
        Err(err) => log_driver_error("CudaContext::new(0)", err),
    }

    match CudaContext::new_non_primary(0, 0) {
        Ok(context) => {
            eprintln!("CudaContext::new_non_primary(0, 0)=ok");
            log_context_details("non-primary-context-wrapper", &context);
            drop(context);
            log_current_context("after-drop-non-primary-context-wrapper");
        }
        Err(err) => log_driver_error("CudaContext::new_non_primary(0, 0)", err),
    }
}

fn log_context_details(label: &str, context: &CudaContext) {
    eprintln!("{label}.ordinal={}", context.ordinal());
    eprintln!("{label}.is_primary={}", context.is_primary());
    eprintln!("{label}.has_async_alloc={}", context.has_async_alloc());
    log_value(&format!("{label}.name"), context.name());
    match context.uuid() {
        Ok(uuid) => eprintln!("{label}.uuid={}", format_uuid(uuid)),
        Err(err) => log_driver_error(&format!("{label}.uuid"), err),
    }
    match context.compute_capability() {
        Ok((major, minor)) => eprintln!("{label}.compute_capability={major}.{minor}"),
        Err(err) => log_driver_error(&format!("{label}.compute_capability"), err),
    }
    match context.total_mem() {
        Ok(bytes) => eprintln!("{label}.total_mem_bytes={bytes}"),
        Err(err) => log_driver_error(&format!("{label}.total_mem"), err),
    }
    match context.mem_get_info() {
        Ok((free, total)) => {
            eprintln!("{label}.mem_get_info=free_bytes={free} total_bytes={total}");
        }
        Err(err) => log_driver_error(&format!("{label}.mem_get_info"), err),
    }
    log_current_context(label);
}

fn log_primary_context_state(device: sys::CUdevice, label: &str) {
    let mut flags = 0_u32;
    let mut active = 0_i32;
    let result =
        unsafe { sys::cuDevicePrimaryCtxGetState(device, &raw mut flags, &raw mut active) };
    match result.result() {
        Ok(()) => eprintln!("primary_ctx_state[{label}]=flags=0x{flags:08x} active={active}"),
        Err(err) => log_driver_error(&format!("cuDevicePrimaryCtxGetState({label})"), err),
    }
}

fn log_current_context(label: &str) {
    match result::ctx::get_current() {
        Ok(Some(context)) => eprintln!("current_ctx[{label}]={}", format_context(context)),
        Ok(None) => eprintln!("current_ctx[{label}]=<none>"),
        Err(err) => log_driver_error(&format!("cuCtxGetCurrent({label})"), err),
    }
}

fn log_device_attr(device: sys::CUdevice, label: &str, attr: sys::CUdevice_attribute) {
    match get_device_attr(device, attr) {
        Ok(value) => eprintln!("{label}={value}"),
        Err(err) => log_driver_error(&format!("cuDeviceGetAttribute({label})"), err),
    }
}

fn get_device_attr(
    device: sys::CUdevice,
    attr: sys::CUdevice_attribute,
) -> Result<i32, DriverError> {
    unsafe { result::device::get_attribute(device, attr) }
}

fn log_value<T, E>(label: &str, result: Result<T, E>)
where
    T: std::fmt::Display,
    E: std::fmt::Display,
{
    match result {
        Ok(value) => eprintln!("{label}={value}"),
        Err(err) => eprintln!("{label}=<error: {err}>"),
    }
}

fn log_driver_error(label: &str, err: DriverError) {
    let name = err
        .error_name()
        .ok()
        .and_then(|value| value.to_str().ok())
        .unwrap_or("<unknown>");
    let description = err
        .error_string()
        .ok()
        .and_then(|value| value.to_str().ok())
        .unwrap_or("<unknown>");
    eprintln!(
        "{label}=error code={:?} name={name} description={description}",
        err.0
    );
}

fn format_context(context: sys::CUcontext) -> String {
    format!("{:p}", context.cast_const())
}

fn format_uuid(uuid: sys::CUuuid) -> String {
    uuid.bytes
        .into_iter()
        .map(|byte| format!("{:02x}", byte as u8))
        .collect::<Vec<_>>()
        .join("")
}

fn compute_mode_name(value: i32) -> &'static str {
    match value {
        x if x == sys::CUcomputemode_enum::CU_COMPUTEMODE_DEFAULT as i32 => "default",
        x if x == sys::CUcomputemode_enum::CU_COMPUTEMODE_PROHIBITED as i32 => "prohibited",
        x if x == sys::CUcomputemode_enum::CU_COMPUTEMODE_EXCLUSIVE_PROCESS as i32 => {
            "exclusive-process"
        }
        _ => "unknown",
    }
}
