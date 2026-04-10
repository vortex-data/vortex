// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Integration tests in `tests/` are their own test crate; clippy's "outside test module"
// lint does not apply here. See https://github.com/rust-lang/rust-clippy/issues/11024.
#![allow(clippy::tests_outside_test_module)]
// Tests may panic or unwrap freely — this is the standard relaxation for test code.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Integration tests for `vortex-compressor`'s tracing instrumentation.
//!
//! These tests pin the names and stable fields of the events emitted by the compressor by
//! attaching an in-memory capture layer and compressing a small array. They exist to:
//!
//! 1. Catch accidental rename or deletion of observability events that downstream tooling
//!    (dashboards, alerting, perfetto recipes) depends on.
//! 2. Document, by example, what an end-to-end compression produces in the trace stream.
//!
//! The capture layer records structured fields instead of formatted strings, so assertions
//! are against typed values rather than substring matches.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::Event;
use tracing::Subscriber;
use tracing::field::Field;
use tracing::field::Visit;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::Registry;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::builtins::IntDictScheme;

/// A captured tracing event: its target, name, and structured fields.
#[derive(Debug, Clone)]
struct CapturedEvent {
    target: String,
    name: String,
    fields: BTreeMap<String, String>,
}

impl CapturedEvent {
    fn field(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }
}

/// `Visit` implementation that records field values into a `BTreeMap<String, String>`.
///
/// We normalize every value type to its Debug representation so assertions are homogeneous,
/// e.g. `event.field("accepted") == Some("true")`.
#[derive(Default)]
struct FieldVisitor {
    fields: BTreeMap<String, String>,
}

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        self.fields
            .insert(field.name().to_string(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields
            .insert(field.name().to_string(), value.to_string());
    }
}

/// A `tracing_subscriber::Layer` that pushes every event into a shared `Vec`.
///
/// Use [`install_capture_layer`] to attach it to a per-test subscriber.
#[derive(Clone, Default)]
struct CaptureLayer {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);

        let meta = event.metadata();
        // `tracing` stores the event "name" under the reserved `message` field.
        let name = visitor
            .fields
            .remove("message")
            .unwrap_or_else(|| meta.name().to_string());

        self.events.lock().push(CapturedEvent {
            target: meta.target().to_string(),
            name,
            fields: visitor.fields,
        });
    }
}

/// Installs a fresh capture layer as the thread-local default subscriber and returns the
/// shared buffer plus a dispatch guard that restores the previous default when dropped.
///
/// Cargo runs tests in parallel, so we deliberately never touch the global default.
fn install_capture_layer() -> (
    Arc<Mutex<Vec<CapturedEvent>>>,
    tracing::dispatcher::DefaultGuard,
) {
    let layer = CaptureLayer::default();
    let events = Arc::clone(&layer.events);
    let subscriber = Registry::default().with(layer);
    let guard = tracing::subscriber::set_default(subscriber);
    (events, guard)
}

/// Compressing a low-cardinality integer array must emit exactly one `scheme.winner`
/// followed by one `scheme.compress_result` for `vortex.int.dict`, with `accepted = true`.
#[test]
fn winner_and_compress_result_events_emitted() {
    let (events, _guard) = install_capture_layer();

    // 100 rows alternating between three distinct values — easily dict-encodable.
    let values: Vec<i32> = (0..100).map(|i| i % 3).collect();
    let array = PrimitiveArray::new(Buffer::from_iter(values), Validity::NonNullable).into_array();

    let compressor = CascadingCompressor::new(vec![&IntDictScheme]);
    let _compressed = compressor.compress(&array).unwrap();

    let events = events.lock();
    let winners: Vec<_> = events
        .iter()
        .filter(|e| e.name == "scheme.winner")
        .collect();
    assert_eq!(winners.len(), 1, "expected exactly one scheme.winner event");
    assert_eq!(winners[0].field("scheme"), Some("vortex.int.dict"));
    assert_eq!(winners[0].target, "vortex_compressor::select");

    let results: Vec<_> = events
        .iter()
        .filter(|e| e.name == "scheme.compress_result")
        .collect();
    assert_eq!(
        results.len(),
        1,
        "expected exactly one scheme.compress_result event"
    );
    let result = results[0];
    assert_eq!(result.field("scheme"), Some("vortex.int.dict"));
    assert_eq!(result.field("accepted"), Some("true"));
    assert_eq!(result.target, "vortex_compressor::encode");
    // Sanity-check that the numeric fields exist and parse.
    assert!(
        result
            .field("before_nbytes")
            .unwrap()
            .parse::<u64>()
            .is_ok()
    );
    assert!(result.field("after_nbytes").unwrap().parse::<u64>().is_ok());
    assert!(result.field("actual_ratio").unwrap().parse::<f64>().is_ok());
}

/// An empty array must take the `empty` short-circuit path before touching any scheme.
#[test]
fn short_circuit_empty_array_emits_reason_empty() {
    let (events, _guard) = install_capture_layer();

    let array = PrimitiveArray::new(Buffer::<i32>::empty(), Validity::NonNullable).into_array();

    let compressor = CascadingCompressor::new(vec![&IntDictScheme]);
    let _compressed = compressor.compress(&array).unwrap();

    let short_circuits: Vec<_> = events
        .lock()
        .iter()
        .filter(|e| e.name == "short_circuit")
        .cloned()
        .collect();

    assert!(
        short_circuits
            .iter()
            .any(|e| e.field("reason") == Some("empty")),
        "expected a short_circuit event with reason=empty, got: {short_circuits:#?}",
    );
}

/// A compression that uses a scheme returning `Sample` must emit `sample.collected`
/// followed by `sample.result`, both with finite positive ratios and matching scheme names.
///
/// `StringDictScheme` defers to `CompressionEstimate::Sample`, so compressing a large
/// low-cardinality string column exercises the sampling path end to end.
#[test]
fn sampling_emits_sample_collected_and_result() {
    use vortex_array::arrays::VarBinArray;
    use vortex_compressor::builtins::StringDictScheme;

    let (events, _guard) = install_capture_layer();

    // A large string column with very low cardinality — forces Sample to run on a large
    // enough input that `sample_count_approx_one_percent` produces more than one slice.
    let strs: Vec<&str> = (0..10_000)
        .map(|i| ["red", "green", "blue"][i % 3])
        .collect();
    let array = VarBinArray::from(strs).into_array();

    let compressor = CascadingCompressor::new(vec![&StringDictScheme]);
    let _compressed = compressor.compress(&array).unwrap();

    let events = events.lock();
    let collected_idx = events
        .iter()
        .position(|e| e.name == "sample.collected")
        .expect("expected a sample.collected event");
    let result_idx = events
        .iter()
        .position(|e| e.name == "sample.result")
        .expect("expected a sample.result event");

    assert!(
        collected_idx < result_idx,
        "sample.collected should precede sample.result",
    );

    let result = &events[result_idx];
    assert_eq!(result.target, "vortex_compressor::estimate");
    assert_eq!(result.field("scheme"), Some("vortex.string.dict"));
    let ratio: f64 = result
        .field("sampled_ratio")
        .expect("sampled_ratio field")
        .parse()
        .expect("sampled_ratio parses as f64");
    assert!(ratio.is_finite() && ratio > 0.0, "ratio = {ratio}");
}
