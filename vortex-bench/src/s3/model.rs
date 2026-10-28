// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::Path;

use anyhow::Context;
use serde::Deserialize;
use serde::Serialize;

const INFLIGHT: [u64; 5] = [1, 4, 16, 64, u64::MAX];

/// A frozen empirical model of response headers, first bytes and remaining transfer cost.
///
/// Samples store integer microseconds, source byte length and exact integer probability mass.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Profile {
    profile_version: u64,
    seed: u64,
    metadata_header_us: u64,
    bands: Vec<Band>,
    training: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Band {
    max_bytes: u64,
    observations: u64,
    samples: Vec<[u64; 5]>,
    concurrency: Vec<Load>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Load {
    max_inflight: u64,
    observations: u64,
    samples: Vec<[u64; 5]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Timing {
    pub header_us: u64,
    pub gap_us: u64,
    pub transfer_us: u64,
}

impl Profile {
    /// Loads and validates a learned profile with at least two recorded real measurement windows.
    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let profile: Self = serde_json::from_reader(File::open(path)?)?;
        profile.validate()?;
        anyhow::ensure!(
            profile.training["measurement_windows"]
                .as_array()
                .is_some_and(|w| w.len() >= 2),
            "A learned profile requires at least two real measurement windows"
        );
        Ok(profile)
    }

    /// Creates a no-delay endpoint for measuring the loopback/client overhead before fitting.
    pub fn zero_delay() -> Self {
        Self {
            profile_version: 1,
            seed: 20261027,
            metadata_header_us: 0,
            bands: vec![Band {
                max_bytes: u64::MAX,
                observations: 1,
                samples: vec![[0, 0, 0, 1, 1]],
                concurrency: vec![],
            }],
            training: serde_json::Value::Null,
        }
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.profile_version == 1 && !self.bands.is_empty(),
            "Unsupported or empty model profile"
        );
        anyhow::ensure!(
            self.metadata_header_us <= 3_600_000_000,
            "Metadata delay exceeds one hour"
        );
        anyhow::ensure!(
            self.bands
                .windows(2)
                .all(|w| w[0].max_bytes < w[1].max_bytes),
            "Size bands must increase"
        );
        for band in &self.bands {
            anyhow::ensure!(band.max_bytes > 0, "Size bounds must be positive");
            validate_samples(&band.samples, band.observations)?;
            anyhow::ensure!(
                band.concurrency
                    .windows(2)
                    .all(|w| w[0].max_inflight < w[1].max_inflight),
                "Concurrency bands must increase"
            );
            for load in &band.concurrency {
                anyhow::ensure!(
                    INFLIGHT.contains(&load.max_inflight),
                    "Unknown concurrency band"
                );
                validate_samples(&load.samples, load.observations)?;
            }
        }
        Ok(())
    }

    pub(super) fn metadata_header_us(&self) -> u64 {
        self.metadata_header_us
    }

    pub(super) fn timing(
        &self,
        path: &str,
        range: &str,
        bytes: u64,
        inflight: u64,
        ordinal: u64,
    ) -> anyhow::Result<Timing> {
        let band = self
            .bands
            .iter()
            .find(|b| bytes <= b.max_bytes)
            .or_else(|| self.bands.last())
            .context("No size band")?;
        let upper = INFLIGHT
            .iter()
            .find(|upper| inflight <= **upper)
            .copied()
            .context("Invalid in-flight count")?;
        let samples = band
            .concurrency
            .iter()
            .find(|load| load.max_inflight == upper)
            .map_or(&band.samples, |load| &load.samples);
        let total = samples.iter().map(|sample| sample[4]).sum::<u64>();
        let mut hash = 0xcbf29ce484222325u64;
        for byte in path
            .bytes()
            .chain([0])
            .chain(range.bytes())
            .chain(self.seed.to_le_bytes())
            .chain(ordinal.to_le_bytes())
        {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d049bb133111eb);
        hash ^= hash >> 31;
        let mut draw = hash % total;
        let sample = samples
            .iter()
            .find(|sample| {
                if draw < sample[4] {
                    true
                } else {
                    draw -= sample[4];
                    false
                }
            })
            .context("Invalid sample mass")?;
        let transfer = u128::from(sample[2]) * u128::from(bytes) / u128::from(sample[3]);
        Ok(Timing {
            header_us: sample[0],
            gap_us: sample[1],
            transfer_us: u64::try_from(transfer)
                .context("Extrapolated transfer cost exceeds model limits")?,
        })
    }
}

fn validate_samples(samples: &[[u64; 5]], observations: u64) -> anyhow::Result<()> {
    let mass = samples.iter().try_fold(0u64, |sum, sample| {
        anyhow::ensure!(
            sample[3] > 0 && sample[4] > 0,
            "Samples need positive byte size and probability mass"
        );
        anyhow::ensure!(
            sample[..3].iter().all(|value| *value <= 3_600_000_000),
            "Observed phase exceeds one hour"
        );
        sum.checked_add(sample[4])
            .context("Probability mass overflow")
    })?;
    anyhow::ensure!(
        !samples.is_empty() && observations == mass,
        "Empirical probability mass must match observations"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> anyhow::Result<Profile> {
        Ok(serde_json::from_value(serde_json::json!({
            "profile_version": 1, "seed": 42, "metadata_header_us": 100,
            "training": {}, "bands": [{"max_bytes": 1024, "observations": 3,
                "samples": [[10, 20, 30, 100, 2], [100, 200, 300, 100, 1]], "concurrency": []}]
        }))?)
    }

    #[test]
    fn integer_mass_preserves_joint_phases_and_byte_cost() -> anyhow::Result<()> {
        let profile = profile()?;
        profile.validate()?;
        let mut slow = 0;
        for ordinal in 0..1000 {
            let timing = profile.timing("file", "0..200", 200, 1, ordinal)?;
            assert_eq!(timing.gap_us, timing.header_us * 2);
            assert_eq!(timing.transfer_us, timing.header_us * 6);
            slow += usize::from(timing.header_us == 100);
            assert_eq!(timing, profile.timing("file", "0..200", 200, 1, ordinal)?);
        }
        assert!((250..420).contains(&slow));
        Ok(())
    }

    #[test]
    fn incomplete_or_overflowing_probability_mass_is_invalid() -> anyhow::Result<()> {
        let mut model = profile()?;
        model.bands[0].samples[0][4] = 0;
        assert!(model.validate().is_err());
        let mut model = profile()?;
        model.bands[0].observations = 4;
        assert!(model.validate().is_err());
        let mut model = profile()?;
        model.bands[0].samples[0][3] = 0;
        assert!(model.validate().is_err());
        Ok(())
    }
}
