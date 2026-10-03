// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! LO2 Prometheus metrics benchmark.
//!
//! Real node_exporter and cAdvisor metrics from the LO2v2 data set
//! (<https://zenodo.org/records/18937117>, CC BY 4.0), queried with real PromQL taken from
//! node-mixin, awesome-prometheus-alerts, the Node Exporter Full Grafana dashboard and prombench,
//! translated to SQL.

mod benchmark;
mod data;

pub use benchmark::*;
