// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use custom_labels::Labelset;
use custom_labels::asynchronous::Label;
use datafusion::common::runtime::JoinSetTracer;
use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_bench::Format;
use vortex_metrics::tracing::{get_global_labels, set_global_labels};

pub fn get_static_tracer() -> &'static dyn JoinSetTracer {
    static TRACER: LabelsJoinSetTracer = LabelsJoinSetTracer;
    &TRACER
}

pub fn get_labelset_from_global() -> Labelset {
    let mut labelset = Labelset::clone_from_current();

    let labels = get_global_labels();

    for (k, v) in labels {
        labelset.set(k, v);
    }

    labelset
}

pub fn set_labels(benchmark_name: String, query_idx: usize, format: Format) -> Labelset {
    let labels = vec![
        ("benchmark_name", benchmark_name),
        ("query_idx", query_idx.to_string()),
        ("format", format.to_string()),
    ];
    set_global_labels(labels.clone());

    let mut labelset = Labelset::clone_from_current();

    for (k, v) in labels.into_iter() {
        labelset.set(k, v);
    }

    labelset
}

pub struct LabelsJoinSetTracer;

impl JoinSetTracer for LabelsJoinSetTracer {
    fn trace_future(
        &self,
        fut: BoxFuture<'static, Box<dyn Any + Send>>,
    ) -> BoxFuture<'static, Box<dyn Any + Send>> {
        fut.with_current_labels().boxed()
    }

    fn trace_block(
        &self,
        f: Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>,
    ) -> Box<dyn FnOnce() -> Box<dyn Any + Send> + Send> {
        let mut labelset = Labelset::clone_from_current();
        Box::new(move || labelset.enter(f))
    }
}
