// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_fuzz::cuda_probe::CudaProbeMode;
use vortex_fuzz::cuda_probe::log_cuda_driver_probe;

fn main() {
    let mut args = std::env::args().skip(1);
    let phase = args.next().unwrap_or_else(|| "manual".to_string());
    let mode = match args.next().as_deref() {
        Some("read-only" | "readonly" | "ro") => CudaProbeMode::ReadOnly,
        _ => CudaProbeMode::Full,
    };

    eprintln!("===== cuda_probe binary =====");
    log_cuda_driver_probe(&phase, mode);
}
