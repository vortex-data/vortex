// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Identifying the machine timings were taken on.
//!
//! Timings from the same machine identity are comparable and are aggregated together; timings from
//! different machines or architectures are kept apart.

use std::env::consts;
use std::fs;

use serde::Deserialize;
use serde::Serialize;
use vortex::utils::parallelism::get_available_parallelism;

use crate::key::stable_digest;

/// What identifies a timing machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineInfo {
    /// A short hash of `arch`, `cpu_model` and `logical_cores`.
    pub machine_id: String,
    /// The CPU architecture, e.g. `x86_64` or `aarch64`.
    pub arch: String,
    /// The CPU model name, from `/proc/cpuinfo` when available.
    pub cpu_model: String,
    /// The number of logical cores.
    pub logical_cores: usize,
    /// The operating system.
    pub os: String,
    /// The kernel release, when available.
    pub kernel: String,
}

impl MachineInfo {
    /// Describes the current machine.
    pub fn current() -> anyhow::Result<Self> {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let cpu_model = cpuinfo
            .lines()
            .find(|l| l.starts_with("model name") || l.starts_with("CPU part"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let logical_cores = get_available_parallelism().unwrap_or(1);
        let arch = consts::ARCH.to_string();
        let mut machine_id = stable_digest("machine", &(&arch, &cpu_model, logical_cores))?;
        machine_id.truncate(16);
        Ok(Self {
            machine_id,
            arch,
            cpu_model,
            logical_cores,
            os: consts::OS.to_string(),
            kernel: fs::read_to_string("/proc/sys/kernel/osrelease")
                .map(|k| k.trim().to_string())
                .unwrap_or_default(),
        })
    }
}
