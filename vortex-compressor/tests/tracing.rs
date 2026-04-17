// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Integration tests in `tests/` are their own test crate; clippy's "outside test module"
// lint does not apply here. See https://github.com/rust-lang/rust-clippy/issues/11024.
#![allow(clippy::tests_outside_test_module)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Regression tests for the load-bearing compressor tracing events. The instrumentation is
//! otherwise considered unstable — these tests exist only for the events that downstream
//! tooling (dashboards, jq recipes) relies on.

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
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::scheme::SchemeExt;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;

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

#[derive(Clone, Default)]
struct CaptureLayer {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);

        let meta = event.metadata();
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

/// The primary event: one `scheme.compress_result` per leaf decision with `accepted = true`
/// when the scheme won and produced a smaller output.
#[test]
fn compress_result_emitted_with_accepted_true_on_win() {
    let (events, _guard) = install_capture_layer();

    let values: Vec<i32> = (0..100).map(|i| i % 3).collect();
    let array = PrimitiveArray::new(Buffer::from_iter(values), Validity::NonNullable).into_array();

    let compressor = CascadingCompressor::new(vec![&IntDictScheme]);
    let _compressed = compressor.compress(&array).unwrap();

    let events = events.lock();
    let result = events
        .iter()
        .find(|e| e.name == "scheme.compress_result")
        .expect("expected a scheme.compress_result event");
    assert_eq!(result.target, "vortex_compressor::encode");
    assert_eq!(result.field("scheme"), Some("vortex.int.dict"));
    assert_eq!(result.field("accepted"), Some("true"));
    result
        .field("before_nbytes")
        .unwrap()
        .parse::<u64>()
        .unwrap();
    result
        .field("after_nbytes")
        .unwrap()
        .parse::<u64>()
        .unwrap();
    result
        .field("actual_ratio")
        .unwrap()
        .parse::<f64>()
        .unwrap();
}

/// When a scheme reports a high estimated ratio but its compressed output is larger than the
/// canonical input, the same event fires with `accepted = false` so operators can discover
/// bad estimators (the motivating bug behind issue #7268).
#[test]
fn compress_result_emitted_with_accepted_false_on_larger_output() {
    let (events, _guard) = install_capture_layer();

    let values: Vec<i32> = (0..50).collect();
    let array = PrimitiveArray::new(Buffer::from_iter(values), Validity::NonNullable).into_array();

    let compressor = CascadingCompressor::new(vec![&OverestimatingScheme]);
    let _compressed = compressor.compress(&array).unwrap();

    let events = events.lock();
    let result = events
        .iter()
        .find(|e| e.name == "scheme.compress_result")
        .expect("expected a scheme.compress_result event");
    assert_eq!(result.field("scheme"), Some("test.overestimating"));
    assert_eq!(result.field("accepted"), Some("false"));

    let before: u64 = result.field("before_nbytes").unwrap().parse().unwrap();
    let after: u64 = result.field("after_nbytes").unwrap().parse().unwrap();
    assert!(after > before);
    assert!(
        result
            .field("estimated_ratio")
            .unwrap()
            .parse::<f64>()
            .unwrap()
            > 1.0
    );
}

/// Exceeding `MAX_CASCADE` must emit `cascade_exhausted` so the silent truncation of the
/// cascade chain is discoverable.
#[test]
fn cascade_exhausted_emitted_after_max_depth() {
    let (events, _guard) = install_capture_layer();

    let values: Vec<i32> = (0..50).collect();
    let array = PrimitiveArray::new(Buffer::from_iter(values), Validity::NonNullable).into_array();

    let compressor = CascadingCompressor::new(vec![
        &RecursiveSchemeA,
        &RecursiveSchemeB,
        &RecursiveSchemeC,
        &RecursiveSchemeD,
    ]);
    let _compressed = compressor.compress(&array).unwrap();

    let events = events.lock();
    let exhausted = events
        .iter()
        .find(|e| e.name == "cascade_exhausted")
        .expect("expected a cascade_exhausted event");
    assert_eq!(exhausted.target, "vortex_compressor::cascade");
    assert!(exhausted.field("parent").is_some());
    assert!(exhausted.field("child_index").is_some());
}

// ---- test schemes ----------------------------------------------------------

#[derive(Debug)]
struct OverestimatingScheme;

impl Scheme for OverestimatingScheme {
    fn scheme_name(&self) -> &'static str {
        "test.overestimating"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(canonical, Canonical::Primitive(p) if p.ptype().is_int())
    }

    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::Ratio(100.0))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let slice: Vec<i32> = data.array_as_primitive().as_slice::<i32>().to_vec();
        let mut doubled = Vec::with_capacity(slice.len() * 2);
        doubled.extend_from_slice(&slice);
        doubled.extend_from_slice(&slice);
        Ok(PrimitiveArray::new(Buffer::from_iter(doubled), Validity::NonNullable).into_array())
    }
}

macro_rules! declare_recursive_scheme {
    ($name:ident, $scheme_name:literal, $ratio:expr) => {
        #[derive(Debug)]
        struct $name;

        impl Scheme for $name {
            fn scheme_name(&self) -> &'static str {
                $scheme_name
            }

            fn matches(&self, canonical: &Canonical) -> bool {
                matches!(canonical, Canonical::Primitive(p) if p.ptype().is_int())
            }

            fn expected_compression_ratio(
                &self,
                _data: &mut ArrayAndStats,
                _ctx: CompressorContext,
            ) -> CompressionEstimate {
                CompressionEstimate::Verdict(EstimateVerdict::Ratio($ratio))
            }

            fn compress(
                &self,
                compressor: &CascadingCompressor,
                data: &mut ArrayAndStats,
                ctx: CompressorContext,
            ) -> VortexResult<ArrayRef> {
                compressor.compress_child(data.array(), &ctx, self.id(), 0)
            }
        }
    };
}

declare_recursive_scheme!(RecursiveSchemeA, "test.recursive.a", 10.0);
declare_recursive_scheme!(RecursiveSchemeB, "test.recursive.b", 9.0);
declare_recursive_scheme!(RecursiveSchemeC, "test.recursive.c", 8.0);
declare_recursive_scheme!(RecursiveSchemeD, "test.recursive.d", 7.0);
