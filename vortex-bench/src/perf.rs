// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optional perf-stat control for query execution, excluding setup and warmup.

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;

use anyhow::Context;

pub(crate) struct PerfControl {
    control: File,
    ack: BufReader<File>,
    warmup: usize,
    active: bool,
}

impl PerfControl {
    pub(crate) fn from_env(iterations: usize) -> anyhow::Result<Option<Self>> {
        let Some(control) = std::env::var_os("VORTEX_BENCH_PERF_CONTROL") else {
            return Ok(None);
        };
        let ack = std::env::var_os("VORTEX_BENCH_PERF_ACK")
            .context("VORTEX_BENCH_PERF_ACK is required with VORTEX_BENCH_PERF_CONTROL")?;
        let warmup = std::env::var("VORTEX_BENCH_PERF_WARMUP")
            .map_or(Ok(2), |value| value.parse::<usize>())?;
        anyhow::ensure!(
            warmup < iterations,
            "perf warmup must be smaller than iterations"
        );
        Ok(Some(Self {
            control: File::options().write(true).open(control)?,
            ack: BufReader::new(File::open(ack)?),
            warmup,
            active: false,
        }))
    }

    pub(crate) fn start(&mut self, iteration: usize) -> anyhow::Result<()> {
        if iteration >= self.warmup {
            self.command("enable")?;
            self.active = true;
        }
        Ok(())
    }

    pub(crate) fn stop(&mut self) -> anyhow::Result<()> {
        if self.active {
            self.command("disable")?;
            self.active = false;
        }
        Ok(())
    }

    fn command(&mut self, command: &str) -> anyhow::Result<()> {
        writeln!(self.control, "{command}")?;
        self.control.flush()?;
        let mut ack = String::new();
        self.ack.read_line(&mut ack)?;
        anyhow::ensure!(
            ack.trim_matches(|character: char| character.is_ascii_whitespace() || character == '\0')
                == "ack",
            "unexpected perf acknowledgement: {ack:?}"
        );
        Ok(())
    }
}

impl Drop for PerfControl {
    fn drop(&mut self) {
        if self.active {
            // A failed query must not leave counters running during error reporting.
            drop(writeln!(self.control, "disable"));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::fs::File;
    use std::io::BufReader;

    use super::PerfControl;

    #[test]
    fn warmup_is_excluded_and_query_commands_are_acknowledged() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let control = directory.path().join("control");
        let ack = directory.path().join("ack");
        // perf versions that write the terminating NUL leave it before the next line.
        fs::write(&ack, "ack\n\0ack\n\0")?;
        let mut perf = PerfControl {
            control: File::create(&control)?,
            ack: BufReader::new(File::open(&ack)?),
            warmup: 2,
            active: false,
        };
        for iteration in 0..2 {
            perf.start(iteration)?;
            perf.stop()?;
        }
        assert!(fs::read_to_string(&control)?.is_empty());
        perf.start(2)?;
        perf.stop()?;
        assert_eq!(fs::read_to_string(&control)?, "enable\ndisable\n");
        Ok(())
    }

    #[test]
    fn invalid_acknowledgement_fails_the_measurement() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let ack = directory.path().join("ack");
        fs::write(&ack, "bad\n")?;
        let mut perf = PerfControl {
            control: File::create(directory.path().join("control"))?,
            ack: BufReader::new(File::open(ack)?),
            warmup: 0,
            active: false,
        };
        assert!(perf.start(0).is_err());
        Ok(())
    }

    #[test]
    fn interrupted_query_disables_counters() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let control = directory.path().join("control");
        let ack = directory.path().join("ack");
        fs::write(&ack, "ack\n")?;
        let mut perf = PerfControl {
            control: File::create(&control)?,
            ack: BufReader::new(File::open(ack)?),
            warmup: 0,
            active: false,
        };
        perf.start(0)?;
        drop(perf);
        assert_eq!(fs::read_to_string(&control)?, "enable\ndisable\n");
        Ok(())
    }
}
