// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opt-in diagnostics for the size estimates layout writers use to coalesce data.
//!
//! Writers decide when to flush a block from [`ArrayRef::nbytes`], which sums every reachable
//! buffer and so over-counts sliced or shared buffers. Setting `VORTEX_NBYTES_AUDIT` to a file
//! path appends one JSON line per measured array to that file, recording both `nbytes` and
//! [`exact_nbytes`]. Setting `VORTEX_NBYTES_POLICY=exact` additionally makes repartitioning
//! account with [`exact_nbytes`], so the two policies can be compared on the same data.
//!
//! With neither variable set, the only cost is reading two cached flags.

use std::fmt::Write as _;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::sync::LazyLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::fns::exact_nbytes::exact_nbytes;
use vortex_error::VortexResult;

static SINK: LazyLock<Option<Mutex<File>>> = LazyLock::new(|| {
    let path = std::env::var_os("VORTEX_NBYTES_AUDIT")?;
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("cannot open VORTEX_NBYTES_AUDIT {path:?}: {e}"));
    Some(Mutex::new(file))
});

static EXACT_POLICY: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("VORTEX_NBYTES_POLICY").is_ok_and(|policy| policy.eq_ignore_ascii_case("exact"))
});

static LABEL: Mutex<String> = Mutex::new(String::new());

static NEXT_STREAM: AtomicU64 = AtomicU64::new(0);

/// Whether audit records are being written.
pub fn enabled() -> bool {
    SINK.is_some()
}

/// Whether repartitioning should account with [`exact_nbytes`] instead of `nbytes`.
pub fn use_exact_policy() -> bool {
    *EXACT_POLICY
}

/// Tag subsequent records, e.g. with the dataset and table being written.
pub fn set_label(label: impl Into<String>) {
    *LABEL.lock() = label.into();
}

/// A fresh identifier for one writer stream, so records can be grouped per column.
pub(crate) fn next_stream_id() -> u64 {
    NEXT_STREAM.fetch_add(1, Ordering::Relaxed)
}

/// Where in the writer an array was measured.
#[derive(Clone, Copy)]
pub(crate) struct Site {
    pub name: &'static str,
    pub stream: u64,
    /// The byte threshold the site compares its running size against.
    pub threshold: u64,
}

/// The size a repartition buffer should account for `array`, recording it when auditing.
pub(crate) fn measure(site: Site, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let nbytes = array.nbytes();
    if !enabled() && !use_exact_policy() {
        return Ok(nbytes);
    }
    let exact = exact_nbytes(array, ctx)?;
    record(site, array, nbytes, Some(exact));
    Ok(if use_exact_policy() { exact } else { nbytes })
}

/// Record an array a site has already sized, computing its exact size only when auditing.
pub(crate) fn observe(site: Site, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
    if enabled() {
        let exact = exact_nbytes(array, ctx)?;
        record(site, array, array.nbytes(), Some(exact));
    }
    Ok(())
}

fn record(site: Site, array: &ArrayRef, nbytes: u64, exact: Option<u64>) {
    let Some(sink) = SINK.as_ref() else {
        return;
    };

    let mut line = String::with_capacity(256);
    line.push_str("{\"label\":");
    push_json_str(&mut line, &LABEL.lock());
    line.push_str(",\"site\":");
    push_json_str(&mut line, site.name);
    let _ = write!(
        line,
        ",\"stream\":{},\"threshold\":{},\"policy\":\"{}\",\"len\":{},\"nbytes\":{}",
        site.stream,
        site.threshold,
        if use_exact_policy() {
            "exact"
        } else {
            "nbytes"
        },
        array.len(),
        nbytes,
    );
    if let Some(exact) = exact {
        let _ = write!(line, ",\"exact\":{exact}");
    }
    let _ = write!(line, ",\"held\":{}", held_bytes(array));
    line.push_str(",\"encoding\":");
    push_json_str(&mut line, &array.encoding_id().to_string());
    line.push_str(",\"dtype\":");
    let dtype = array.dtype().to_string();
    push_json_str(&mut line, &dtype[..dtype.floor_char_boundary(120)]);
    line.push_str("}\n");

    // Diagnostics must never fail a write.
    drop(sink.lock().write_all(line.as_bytes()));
}

/// Bytes of memory the array's buffers occupy, counting overlapping buffers once.
///
/// Unlike `nbytes`, buffers that alias the same allocation (e.g. the data buffers shared by
/// slices of one string array) are not counted repeatedly; unlike [`exact_nbytes`], bytes the
/// array holds but never references are still counted.
fn held_bytes(array: &ArrayRef) -> u64 {
    let mut ranges: Vec<(usize, usize)> = array
        .depth_first_traversal()
        .flat_map(|node| node.buffers())
        .filter(|buffer| !buffer.is_empty())
        .map(|buffer| {
            let start = buffer.as_ptr() as usize;
            (start, start + buffer.len())
        })
        .collect();
    ranges.sort_unstable();

    let mut held = 0usize;
    let mut current: Option<(usize, usize)> = None;
    for (start, end) in ranges {
        match current.as_mut() {
            Some(range) if start <= range.1 => range.1 = range.1.max(end),
            Some(range) => {
                held += range.1 - range.0;
                *range = (start, end);
            }
            None => current = Some((start, end)),
        }
    }
    if let Some((start, end)) = current {
        held += end - start;
    }
    held as u64
}

fn push_json_str(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
