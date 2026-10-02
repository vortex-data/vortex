// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Write;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use tracing::Event;
use tracing::Level;
use tracing::Metadata;
use tracing::Subscriber;
use tracing::field::Field;
use tracing::field::Visit;
use tracing::subscriber::Interest;
use tracing_subscriber::Layer;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::Context;
use tracing_subscriber::layer::Filter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::cpp;
use crate::cpp::DUCKDB_VX_LOG_LEVEL;

static MIN_LEVEL: AtomicU8 = AtomicU8::new(u8::MAX);

pub(crate) fn set_min_level(level: u8) {
    if MIN_LEVEL.swap(level, Ordering::Relaxed) != level {
        tracing::callsite::rebuild_interest_cache();
    }
}

fn duckdb_level(level: &Level) -> DUCKDB_VX_LOG_LEVEL {
    match *level {
        Level::TRACE => DUCKDB_VX_LOG_LEVEL::DUCKDB_VX_LOG_LEVEL_TRACE,
        Level::DEBUG => DUCKDB_VX_LOG_LEVEL::DUCKDB_VX_LOG_LEVEL_DEBUG,
        Level::INFO => DUCKDB_VX_LOG_LEVEL::DUCKDB_VX_LOG_LEVEL_INFO,
        Level::WARN => DUCKDB_VX_LOG_LEVEL::DUCKDB_VX_LOG_LEVEL_WARNING,
        Level::ERROR => DUCKDB_VX_LOG_LEVEL::DUCKDB_VX_LOG_LEVEL_ERROR,
    }
}

fn duckdb_enabled(metadata: &Metadata<'_>) -> bool {
    metadata.is_event() && duckdb_level(metadata.level()) as u8 >= MIN_LEVEL.load(Ordering::Relaxed)
}

struct DuckdbFilter;

impl<S> Filter<S> for DuckdbFilter {
    fn enabled(&self, metadata: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        duckdb_enabled(metadata)
    }

    fn callsite_enabled(&self, metadata: &'static Metadata<'static>) -> Interest {
        if duckdb_enabled(metadata) {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(match MIN_LEVEL.load(Ordering::Relaxed) {
            ..=10 => LevelFilter::TRACE,
            11..=20 => LevelFilter::DEBUG,
            21..=30 => LevelFilter::INFO,
            31..=40 => LevelFilter::WARN,
            41..=50 => LevelFilter::ERROR,
            _ => LevelFilter::OFF,
        })
    }
}

#[derive(Default)]
struct EventFormatter {
    message: String,
    fields: String,
}

impl Visit for EventFormatter {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            let _ = write!(self.fields, " {}={:?}", field.name(), value);
        }
    }
}

struct DuckdbLayer;

impl<S: Subscriber> Layer<S> for DuckdbLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let mut formatter = EventFormatter::default();
        event.record(&mut formatter);
        let line = format!(
            "{}: {}{}",
            metadata.target(),
            formatter.message,
            formatter.fields
        );
        unsafe {
            cpp::duckdb_vx_log(
                duckdb_level(metadata.level()),
                line.as_ptr().cast(),
                line.len(),
            )
        };
    }
}

pub fn init_tracing() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        drop(
            tracing_subscriber::registry()
                .with(DuckdbLayer.with_filter(DuckdbFilter))
                .try_init(),
        );
    });
}
