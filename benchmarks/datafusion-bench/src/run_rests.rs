// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `--opt rest-runs=<n>,rest-ms=<ms>`: the drives rest for `rest-ms` before each of a query's
//! runs 1 to `n`; its first run, 0, and the runs after `n` go straight on, back to back.
//! Instance-store SSDs read several times their steady rate for a moment after a rest, then hold
//! the rate their instance type is sold with: the rested runs measure a query's latency on rested
//! drives, the back-to-back runs after them its latency once the burst is spent.

use std::time::Duration;

use parking_lot::Mutex;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_bench::Format;
use vortex_bench::Opts;

/// When a query's runs rest first: the runs after the first, up to `rested_runs`.
pub struct RunRests {
    rested_runs: usize,
    rest: Duration,
    runs_started: Mutex<HashMap<(usize, Format), usize>>,
}

impl RunRests {
    /// The rests `rest-runs` and `rest-ms` ask for, if either is given; both or neither.
    pub fn from_opts(opts: &Opts) -> anyhow::Result<Option<Self>> {
        match (opts.get("rest-runs"), opts.get("rest-ms")) {
            (None, None) => Ok(None),
            (Some(runs), Some(ms)) => Ok(Some(Self {
                rested_runs: runs.parse()?,
                rest: Duration::from_millis(ms.parse()?),
                runs_started: Mutex::new(HashMap::new()),
            })),
            _ => anyhow::bail!("rest-runs and rest-ms go together"),
        }
    }

    /// Counts the query's run as started, and rests first if it is one of runs 1 to
    /// `rested_runs`.
    pub async fn before_run(&self, query_idx: usize, format: Format) {
        let run = {
            let mut started = self.runs_started.lock();
            let count = started.entry((query_idx, format)).or_insert(0);
            let run = *count;
            *count += 1;
            run
        };
        if (1..=self.rested_runs).contains(&run) {
            tokio::time::sleep(self.rest).await;
        }
    }
}
