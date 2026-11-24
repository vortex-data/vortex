// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Identifying the machine timings were taken on.
//!
//! Timings from the same machine identity are comparable and are aggregated together; timings from
//! different machines or architectures are kept apart. The compiled target features are part of
//! the identity, since they change which SIMD kernels run.

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
        let features = compiled_target_features();
        let mut machine_id =
            stable_digest("machine", &(&arch, &cpu_model, logical_cores, &features))?;
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

/// The conditions timings were taken under, recorded with every timing run.
///
/// Timings are only comparable within one machine id; these fields explain why two runs on the
/// same machine might still differ, and [`warnings`](Self::warnings) names conditions known to
/// add noise.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimingEnv {
    /// The CPU frequency governor of the first allowed CPU, e.g. `performance`.
    pub governor: Option<String>,
    /// Whether turbo boost is enabled, when the kernel reports it.
    pub turbo: Option<bool>,
    /// The CPUs this process may run on, from `/proc/self/status` (set with `taskset -c N`).
    pub cpus_allowed: Option<String>,
    /// Whether this is a release build.
    pub release: bool,
    /// The target features the binary was compiled with that change SIMD code paths.
    pub target_features: Vec<String>,
}

/// Target features that change which kernels run; they are part of the machine id.
fn compiled_target_features() -> Vec<String> {
    let mut features = Vec::new();
    macro_rules! check {
        ($($f:literal),*) => {
            $(if cfg!(target_feature = $f) { features.push($f.to_string()); })*
        };
    }
    check!(
        "sse4.2", "avx", "avx2", "bmi2", "fma", "avx512f", "avx512bw", "avx512vl", "neon", "sve",
        "sve2"
    );
    features
}

impl TimingEnv {
    /// Reads the current conditions.
    pub fn current() -> Self {
        let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
        let cpus_allowed = status
            .lines()
            .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
            .map(|v| v.trim().to_string());
        let first_cpu = cpus_allowed
            .as_deref()
            .and_then(|c| c.split([',', '-']).next())
            .and_then(|c| c.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let read = |path: String| fs::read_to_string(path).ok().map(|s| s.trim().to_string());
        let governor = read(format!(
            "/sys/devices/system/cpu/cpu{first_cpu}/cpufreq/scaling_governor"
        ));
        let turbo = read("/sys/devices/system/cpu/intel_pstate/no_turbo".into())
            .map(|v| v == "0")
            .or_else(|| read("/sys/devices/system/cpu/cpufreq/boost".into()).map(|v| v == "1"));
        Self {
            governor,
            turbo,
            cpus_allowed,
            release: !cfg!(debug_assertions),
            target_features: compiled_target_features(),
        }
    }

    /// Conditions known to add timing noise.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.release {
            out.push("debug build".into());
        }
        if let Some(g) = &self.governor
            && g != "performance"
        {
            out.push(format!("governor is {g}, not performance"));
        }
        if self.turbo == Some(true) {
            out.push("turbo boost is on".into());
        }
        let pinned = self
            .cpus_allowed
            .as_deref()
            .is_some_and(|c| !c.contains([',', '-']));
        if !pinned {
            out.push("not pinned to one CPU (run under `taskset -c N`)".into());
        }
        out
    }
}
