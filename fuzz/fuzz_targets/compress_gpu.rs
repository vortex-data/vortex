// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]
#![expect(clippy::unwrap_used)]

use std::env;
use std::fs;
use std::process::Command;
use std::sync::OnceLock;

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzCompressGpu;
use vortex_fuzz::run_compress_gpu;

static STARTUP_DIAGNOSTICS: OnceLock<()> = OnceLock::new();

fn log_command(label: &str, command: &str, args: &[&str]) {
    eprintln!("## {label}");
    match Command::new(command).args(args).output() {
        Ok(output) => {
            eprintln!("status: {}", output.status);

            let stdout = String::from_utf8_lossy(&output.stdout);
            if !stdout.trim().is_empty() {
                eprintln!("stdout:\n{stdout}");
            }

            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.trim().is_empty() {
                eprintln!("stderr:\n{stderr}");
            }
        }
        Err(err) => eprintln!("failed to run `{command}`: {err}"),
    }
}

fn log_relevant_processes(pid: u32) {
    eprintln!("## relevant processes");
    match Command::new("ps")
        .args(["-Ao", "pid,ppid,pgid,comm,args"])
        .output()
    {
        Ok(output) => {
            eprintln!("status: {}", output.status);
            let pid = pid.to_string();
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines().filter(|line| {
                line.contains(&pid)
                    || line.contains("compress_gpu")
                    || line.contains("cargo")
                    || line.contains("libFuzzer")
            }) {
                eprintln!("{line}");
            }

            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.trim().is_empty() {
                eprintln!("stderr:\n{stderr}");
            }
        }
        Err(err) => eprintln!("failed to run `ps`: {err}"),
    }
}

fn log_process_snapshot() {
    let pid = std::process::id();
    eprintln!("pid={pid}");
    eprintln!("argv={:?}", env::args().collect::<Vec<_>>());

    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        let interesting = status
            .lines()
            .filter(|line| {
                matches!(
                    line.split_once(':').map(|(key, _)| key),
                    Some("Name")
                        | Some("State")
                        | Some("Pid")
                        | Some("PPid")
                        | Some("Threads")
                        | Some("VmPeak")
                        | Some("VmSize")
                        | Some("VmRSS")
                        | Some("VmData")
                        | Some("VmSwap")
                )
            })
            .collect::<Vec<_>>();

        if !interesting.is_empty() {
            eprintln!("## /proc/self/status");
            for line in interesting {
                eprintln!("{line}");
            }
        }
    }

    let pid_arg = pid.to_string();
    log_command(
        "current process ps",
        "ps",
        &[
            "-p",
            pid_arg.as_str(),
            "-o",
            "pid,ppid,pgid,rss,vsz,etimes,comm,args",
        ],
    );
    log_relevant_processes(pid);
}

fn log_cuda_diagnostics(phase: &str) {
    eprintln!("===== compress_gpu CUDA diagnostics ({phase}) =====");
    eprintln!("cuda_available()={}", vortex_cuda::cuda_available());
    log_process_snapshot();
    eprintln!(
        "CUDA_VISIBLE_DEVICES={}",
        env::var("CUDA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".to_string())
    );
    eprintln!(
        "NVIDIA_VISIBLE_DEVICES={}",
        env::var("NVIDIA_VISIBLE_DEVICES").unwrap_or_else(|_| "<unset>".to_string())
    );
    eprintln!(
        "LD_LIBRARY_PATH={}",
        env::var("LD_LIBRARY_PATH").unwrap_or_else(|_| "<unset>".to_string())
    );

    log_command("nvcc --version", "nvcc", &["--version"]);
    log_command("nvidia-smi", "nvidia-smi", &[]);
    log_command("nvidia-smi -L", "nvidia-smi", &["-L"]);
    log_command("nvidia-smi memory", "nvidia-smi", &["-q", "-d", "Memory"]);
    log_command(
        "nvidia-smi gpu summary",
        "nvidia-smi",
        &[
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv",
        ],
    );
    log_command(
        "nvidia-smi compute processes",
        "nvidia-smi",
        &[
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
    );
}

fuzz_target!(|fuzz: FuzzCompressGpu| -> Corpus {
    STARTUP_DIAGNOSTICS.get_or_init(|| log_cuda_diagnostics("startup"));

    // Use tokio runtime to run async GPU fuzzer
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    match rt.block_on(run_compress_gpu(fuzz)) {
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
        Err(e) => {
            log_cuda_diagnostics("error");
            vortex_panic!("{e}");
        }
    }
});
