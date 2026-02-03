// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use custom_labels::Labelset;
use custom_labels::asynchronous::Label;
use custom_labels::with_labels;
use datafusion::common::runtime::JoinSetTracer;
use futures::FutureExt;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use vortex_bench::Format;

static LABELS: RwLock<Vec<(&str, String)>> = RwLock::new(Vec::new());

pub fn get_static_tracer() -> &'static dyn JoinSetTracer {
    static TRACER: LabelsJoinSetTracer = LabelsJoinSetTracer;
    &TRACER
}

pub fn set_labels(name: String, query_idx: usize, format: Format) {
    let mut labelset = Labelset::clone_from_current();
    labelset.set("benchmark_name", name.as_bytes());
    labelset.set("query_idx", query_idx.to_string());
    labelset.set("format", format.to_string());

    *LABELS.write() = vec![
        ("benchmark_name", name),
        ("query_idx", query_idx.to_string()),
        ("format", format.to_string()),
    ];
}

pub struct LabelsJoinSetTracer;

impl JoinSetTracer for LabelsJoinSetTracer {
    fn trace_future(
        &self,
        fut: BoxFuture<'static, Box<dyn Any + Send>>,
    ) -> BoxFuture<'static, Box<dyn Any + Send>> {
        fut.with_labels(LABELS.read().clone()).boxed()
    }

    fn trace_block(
        &self,
        f: Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>,
    ) -> Box<dyn FnOnce() -> Box<dyn Any + Send> + Send> {
        let labels = LABELS.read().clone();
        Box::new(|| with_labels(labels, f))
    }
}
