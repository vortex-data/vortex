#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

#[cfg(target_arch = "x86_64")]
use std::arch::asm;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_MM_HINT_T0;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_cmpeq_epi8;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_load_si128;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_movemask_epi8;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_prefetch;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_set1_epi32;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_setzero_si128;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::fs;
use std::hash::BuildHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::hint::black_box;
use std::mem::MaybeUninit;
use std::time::Duration;
use std::time::Instant;

const TRACE_MAGIC: &[u8; 8] = b"OPHASH01";

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ShortKey {
    bytes: u64,
    length: u8,
}

impl PartialEq for ShortKey {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes && self.length == other.length
    }
}

impl Eq for ShortKey {}

impl Hash for ShortKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.bytes);
        state.write_u8(self.length);
    }
}

struct Trace {
    short_entries: Vec<(ShortKey, u16)>,
    long_entries: Vec<(u64, u16)>,
    short_probes: Vec<ShortKey>,
    long_probes: Vec<u64>,
    short_search_ends: Option<Vec<usize>>,
    expected_short_checksum: u64,
    expected_long_checksum: u64,
}

fn take_u64(input: &[u8], cursor: &mut usize) -> Result<u64, String> {
    let end = *cursor + 8;
    let value = u64::from_le_bytes(
        input
            .get(*cursor..end)
            .ok_or("truncated trace")?
            .try_into()
            .map_err(|_| "u64")?,
    );
    *cursor = end;
    Ok(value)
}

fn take_u16(input: &[u8], cursor: &mut usize) -> Result<u16, String> {
    let end = *cursor + 2;
    let value = u16::from_le_bytes(
        input
            .get(*cursor..end)
            .ok_or("truncated trace")?
            .try_into()
            .map_err(|_| "u16")?,
    );
    *cursor = end;
    Ok(value)
}

fn load_trace(path: &str) -> Result<Trace, String> {
    let input = fs::read(path).map_err(|error| error.to_string())?;
    if input.len() < 40 || &input[..8] != TRACE_MAGIC {
        return Err("invalid OPHASH01 trace".into());
    }
    let mut cursor = 8;
    let short_entry_count = take_u64(&input, &mut cursor)? as usize;
    let long_entry_count = take_u64(&input, &mut cursor)? as usize;
    let short_probe_count = take_u64(&input, &mut cursor)? as usize;
    let long_probe_count = take_u64(&input, &mut cursor)? as usize;

    let mut short_entries = Vec::with_capacity(short_entry_count);
    for _ in 0..short_entry_count {
        let bytes = take_u64(&input, &mut cursor)?;
        let length = *input.get(cursor).ok_or("truncated short entry")?;
        cursor += 1;
        short_entries.push((ShortKey { bytes, length }, take_u16(&input, &mut cursor)?));
    }
    let mut long_entries = Vec::with_capacity(long_entry_count);
    for _ in 0..long_entry_count {
        let key = take_u64(&input, &mut cursor)?;
        long_entries.push((key, take_u16(&input, &mut cursor)?));
    }
    let mut short_probes = Vec::with_capacity(short_probe_count);
    for _ in 0..short_probe_count {
        let bytes = take_u64(&input, &mut cursor)?;
        let length = *input.get(cursor).ok_or("truncated short probe")?;
        cursor += 1;
        short_probes.push(ShortKey { bytes, length });
    }
    let mut long_probes = Vec::with_capacity(long_probe_count);
    for _ in 0..long_probe_count {
        long_probes.push(take_u64(&input, &mut cursor)?);
    }
    if cursor != input.len() {
        return Err("trailing trace bytes".into());
    }

    let short_lookup: BTreeMap<_, _> = short_entries
        .iter()
        .map(|(key, value)| ((key.bytes, key.length), *value))
        .collect();
    let long_lookup: BTreeMap<_, _> = long_entries.iter().copied().collect();
    let expected_short_checksum = short_probes.iter().fold(0u64, |checksum, key| {
        checksum.wrapping_add(u64::from(
            short_lookup
                .get(&(key.bytes, key.length))
                .copied()
                .unwrap_or_default(),
        ))
    });
    let expected_long_checksum = long_probes.iter().fold(0u64, |checksum, key| {
        checksum.wrapping_add(u64::from(long_lookup.get(key).copied().unwrap_or_default()))
    });

    // A scalar OnPair short lookup probes descending lengths until the first
    // dictionary hit. Recover those boundaries from the exact dictionary, not
    // merely from adjacent lengths: two consecutive token searches can also
    // happen to form a descending sequence. Interleaved traces do not have
    // contiguous search groups and therefore disable grouped-prefix reports.
    let mut short_search_ends = Vec::new();
    for (index, key) in short_probes.iter().enumerate() {
        if short_lookup.contains_key(&(key.bytes, key.length)) {
            short_search_ends.push(index + 1);
        }
    }
    let mut start = 0;
    let scalar_groups = short_search_ends.iter().all(|&end| {
        let descending = short_probes[start..end]
            .windows(2)
            .all(|pair| pair[1].length.checked_add(1) == Some(pair[0].length));
        start = end;
        descending
    });
    let short_search_ends =
        (scalar_groups && start == short_probes.len()).then_some(short_search_ends);

    Ok(Trace {
        short_entries,
        long_entries,
        short_probes,
        long_probes,
        short_search_ends,
        expected_short_checksum,
        expected_long_checksum,
    })
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn measure(
    warmups: usize,
    iterations: usize,
    expected: u64,
    mut lookup: impl FnMut() -> u64,
) -> Duration {
    for _ in 0..warmups {
        assert_eq!(black_box(lookup()), expected, "lookup checksum mismatch");
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let checksum = black_box(lookup());
        samples.push(start.elapsed());
        assert_eq!(checksum, expected, "lookup checksum mismatch");
    }
    median(samples)
}

fn report<S, L>(
    name: &str,
    trace: &Trace,
    short_map: &S,
    long_map: &L,
    short_get: impl Fn(&S, &ShortKey) -> u16,
    long_get: impl Fn(&L, &u64) -> u16,
    warmups: usize,
    iterations: usize,
) {
    let short_time = measure(warmups, iterations, trace.expected_short_checksum, || {
        trace.short_probes.iter().fold(0u64, |checksum, key| {
            checksum.wrapping_add(short_get(short_map, key) as u64)
        })
    });
    let long_time = measure(warmups, iterations, trace.expected_long_checksum, || {
        trace.long_probes.iter().fold(0u64, |checksum, key| {
            checksum.wrapping_add(long_get(long_map, key) as u64)
        })
    });
    let probes = trace.short_probes.len() + trace.long_probes.len();
    let total_ns = short_time.as_nanos() + long_time.as_nanos();
    let ns_per_probe = total_ns as f64 / probes as f64;
    println!(
        "hash,name={name},warmups={warmups},iterations={iterations},short_ms={:.6},long_ms={:.6},total_ms={:.6},ns_per_probe={ns_per_probe:.6},mprobes_s={:.6}",
        short_time.as_secs_f64() * 1e3,
        long_time.as_secs_f64() * 1e3,
        total_ns as f64 / 1e6,
        1e3 / ns_per_probe,
    );
}

fn short_probe_ranges(trace: &Trace) -> impl Iterator<Item = (usize, usize)> + '_ {
    let ends = trace
        .short_search_ends
        .as_ref()
        .expect("grouped-prefix benchmark requires a scalar trace");
    let mut start = 0;
    ends.iter().copied().map(move |end| {
        let range = (start, end);
        start = end;
        range
    })
}

fn report_prefix_bounds<const FILTERS: u8, const BITS_PER_KEY: usize, S, L>(
    name: &str,
    trace: &Trace,
    prefixes: &FrozenPrefixBounds<BITS_PER_KEY>,
    short_map: &S,
    long_map: &L,
    short_get: impl Fn(&S, &ShortKey, Option<u64>) -> u16,
    long_get: impl Fn(&L, &u64) -> u16,
    warmups: usize,
    iterations: usize,
) {
    let ranges: Vec<_> = short_probe_ranges(trace).collect();
    let exact_probes: usize = ranges
        .iter()
        .copied()
        .map(|(start, end)| {
            let bound = prefixes.probe::<FILTERS>(&trace.short_probes[start]).bound;
            trace.short_probes[start..end]
                .iter()
                .filter(|key| key.length <= bound)
                .count()
        })
        .sum();
    let short_time = measure(warmups, iterations, trace.expected_short_checksum, || {
        ranges
            .iter()
            .copied()
            .fold(0u64, |mut checksum, (start, end)| {
                let prefix = prefixes.probe::<FILTERS>(&trace.short_probes[start]);
                for key in &trace.short_probes[start..end] {
                    if key.length <= prefix.bound {
                        checksum = checksum.wrapping_add(u64::from(short_get(
                            short_map,
                            key,
                            prefix.hash_for(key.length),
                        )));
                    }
                }
                checksum
            })
    });
    let long_time = measure(warmups, iterations, trace.expected_long_checksum, || {
        trace.long_probes.iter().fold(0u64, |checksum, key| {
            checksum.wrapping_add(u64::from(long_get(long_map, key)))
        })
    });
    let probes = trace.short_probes.len() + trace.long_probes.len();
    let total_ns = short_time.as_nanos() + long_time.as_nanos();
    let ns_per_probe = total_ns as f64 / probes as f64;
    println!(
        "hash,name={name},warmups={warmups},iterations={iterations},short_ms={:.6},long_ms={:.6},total_ms={:.6},ns_per_original_probe={ns_per_probe:.6},moriginal_probes_s={:.6},prefix_groups={},exact_short_probes={exact_probes},skipped_short_probes={}",
        short_time.as_secs_f64() * 1e3,
        long_time.as_secs_f64() * 1e3,
        total_ns as f64 / 1e6,
        1e3 / ns_per_probe,
        ranges.len(),
        trace.short_probes.len() - exact_probes,
    );
}

fn report_prefix_length_mask<M, S, L>(
    name: &str,
    trace: &Trace,
    prefixes: &M,
    short_map: &S,
    long_map: &L,
    prefix_get: impl Fn(&M, &ShortKey) -> PrefixLengthMaskProbe,
    short_get: impl Fn(&S, &ShortKey, Option<u64>) -> u16,
    long_get: impl Fn(&L, &u64) -> u16,
    warmups: usize,
    iterations: usize,
) {
    let ranges: Vec<_> = short_probe_ranges(trace).collect();
    let exact_probes: usize = ranges
        .iter()
        .copied()
        .map(|(start, end)| {
            let mask = prefix_get(prefixes, &trace.short_probes[start]).lengths;
            trace.short_probes[start..end]
                .iter()
                .filter(|key| key.length < 4 || mask & (1 << key.length) != 0)
                .count()
        })
        .sum();
    let short_time = measure(warmups, iterations, trace.expected_short_checksum, || {
        ranges
            .iter()
            .copied()
            .fold(0u64, |mut checksum, (start, end)| {
                let prefix = prefix_get(prefixes, &trace.short_probes[start]);
                for key in &trace.short_probes[start..end] {
                    if key.length < 4 || prefix.lengths & (1 << key.length) != 0 {
                        checksum = checksum.wrapping_add(u64::from(short_get(
                            short_map,
                            key,
                            (key.length == 4).then_some(prefix.hash),
                        )));
                    }
                }
                checksum
            })
    });
    let long_time = measure(warmups, iterations, trace.expected_long_checksum, || {
        trace.long_probes.iter().fold(0u64, |checksum, key| {
            checksum.wrapping_add(u64::from(long_get(long_map, key)))
        })
    });
    let probes = trace.short_probes.len() + trace.long_probes.len();
    let total_ns = short_time.as_nanos() + long_time.as_nanos();
    let ns_per_probe = total_ns as f64 / probes as f64;
    println!(
        "hash,name={name},warmups={warmups},iterations={iterations},short_ms={:.6},long_ms={:.6},total_ms={:.6},ns_per_original_probe={ns_per_probe:.6},moriginal_probes_s={:.6},prefix_groups={},exact_short_probes={exact_probes},skipped_short_probes={}",
        short_time.as_secs_f64() * 1e3,
        long_time.as_secs_f64() * 1e3,
        total_ns as f64 / 1e6,
        1e3 / ns_per_probe,
        ranges.len(),
        trace.short_probes.len() - exact_probes,
    );
}

fn report_lazy_prefix_length_mask<M, S, L>(
    name: &str,
    trace: &Trace,
    prefixes: &M,
    short_map: &S,
    long_map: &L,
    prefix_get: impl Fn(&M, &ShortKey) -> PrefixLengthMaskProbe,
    short_get: impl Fn(&S, &ShortKey, Option<u64>) -> u16,
    long_get: impl Fn(&L, &u64) -> u16,
    warmups: usize,
    iterations: usize,
) {
    let ranges: Vec<_> = short_probe_ranges(trace).collect();
    let exact_probes: usize = ranges
        .iter()
        .copied()
        .map(|(start, end)| {
            if start + 1 == end || trace.short_probes[start + 1].length < 4 {
                end - start
            } else {
                let mask = prefix_get(prefixes, &trace.short_probes[start]).lengths;
                1 + trace.short_probes[start + 1..end]
                    .iter()
                    .filter(|key| key.length < 4 || mask & (1 << key.length) != 0)
                    .count()
            }
        })
        .sum();
    let prefix_probes = ranges
        .iter()
        .filter(|(start, end)| start + 1 < *end && trace.short_probes[start + 1].length >= 4)
        .count();
    let short_time = measure(warmups, iterations, trace.expected_short_checksum, || {
        ranges
            .iter()
            .copied()
            .fold(0u64, |mut checksum, (start, end)| {
                let first = &trace.short_probes[start];
                checksum = checksum.wrapping_add(u64::from(short_get(
                    short_map,
                    first,
                    Some(short_hash(first)),
                )));
                if start + 1 == end {
                    return checksum;
                }
                if trace.short_probes[start + 1].length < 4 {
                    for key in &trace.short_probes[start + 1..end] {
                        checksum =
                            checksum.wrapping_add(u64::from(short_get(short_map, key, None)));
                    }
                    return checksum;
                }

                let prefix = prefix_get(prefixes, first);
                for key in &trace.short_probes[start + 1..end] {
                    if key.length < 4 || prefix.lengths & (1 << key.length) != 0 {
                        checksum = checksum.wrapping_add(u64::from(short_get(
                            short_map,
                            key,
                            (key.length == 4).then_some(prefix.hash),
                        )));
                    }
                }
                checksum
            })
    });
    let long_time = measure(warmups, iterations, trace.expected_long_checksum, || {
        trace.long_probes.iter().fold(0u64, |checksum, key| {
            checksum.wrapping_add(u64::from(long_get(long_map, key)))
        })
    });
    let probes = trace.short_probes.len() + trace.long_probes.len();
    let total_ns = short_time.as_nanos() + long_time.as_nanos();
    let ns_per_probe = total_ns as f64 / probes as f64;
    println!(
        "hash,name={name},warmups={warmups},iterations={iterations},short_ms={:.6},long_ms={:.6},total_ms={:.6},ns_per_original_probe={ns_per_probe:.6},moriginal_probes_s={:.6},prefix_groups={},prefix_probes={prefix_probes},exact_short_probes={exact_probes},skipped_short_probes={}",
        short_time.as_secs_f64() * 1e3,
        long_time.as_secs_f64() * 1e3,
        total_ns as f64 / 1e6,
        1e3 / ns_per_probe,
        ranges.len(),
        trace.short_probes.len() - exact_probes,
    );
}

fn report_gated_prefix_length_mask<const MIN_LENGTH: u8, M, S, L>(
    name: &str,
    trace: &Trace,
    prefixes: &M,
    short_map: &S,
    long_map: &L,
    prefix_get: impl Fn(&M, &ShortKey) -> PrefixLengthMaskProbe,
    short_get: impl Fn(&S, &ShortKey, Option<u64>) -> u16,
    long_get: impl Fn(&L, &u64) -> u16,
    warmups: usize,
    iterations: usize,
) {
    let ranges: Vec<_> = short_probe_ranges(trace).collect();
    let exact_probes: usize = ranges
        .iter()
        .copied()
        .map(|(start, end)| {
            if trace.short_probes[start].length < MIN_LENGTH {
                end - start
            } else {
                let mask = prefix_get(prefixes, &trace.short_probes[start]).lengths;
                trace.short_probes[start..end]
                    .iter()
                    .filter(|key| key.length < 4 || mask & (1 << key.length) != 0)
                    .count()
            }
        })
        .sum();
    let prefix_probes = ranges
        .iter()
        .filter(|(start, _)| trace.short_probes[*start].length >= MIN_LENGTH)
        .count();
    let short_time = measure(warmups, iterations, trace.expected_short_checksum, || {
        ranges
            .iter()
            .copied()
            .fold(0u64, |mut checksum, (start, end)| {
                if trace.short_probes[start].length < MIN_LENGTH {
                    for key in &trace.short_probes[start..end] {
                        checksum =
                            checksum.wrapping_add(u64::from(short_get(short_map, key, None)));
                    }
                    return checksum;
                }

                let prefix = prefix_get(prefixes, &trace.short_probes[start]);
                for key in &trace.short_probes[start..end] {
                    if key.length < 4 || prefix.lengths & (1 << key.length) != 0 {
                        checksum = checksum.wrapping_add(u64::from(short_get(
                            short_map,
                            key,
                            (key.length == 4).then_some(prefix.hash),
                        )));
                    }
                }
                checksum
            })
    });
    let long_time = measure(warmups, iterations, trace.expected_long_checksum, || {
        trace.long_probes.iter().fold(0u64, |checksum, key| {
            checksum.wrapping_add(u64::from(long_get(long_map, key)))
        })
    });
    let probes = trace.short_probes.len() + trace.long_probes.len();
    let total_ns = short_time.as_nanos() + long_time.as_nanos();
    let ns_per_probe = total_ns as f64 / probes as f64;
    println!(
        "hash,name={name},warmups={warmups},iterations={iterations},short_ms={:.6},long_ms={:.6},total_ms={:.6},ns_per_original_probe={ns_per_probe:.6},moriginal_probes_s={:.6},prefix_groups={},prefix_probes={prefix_probes},exact_short_probes={exact_probes},skipped_short_probes={}",
        short_time.as_secs_f64() * 1e3,
        long_time.as_secs_f64() * 1e3,
        total_ns as f64 / 1e6,
        1e3 / ns_per_probe,
        ranges.len(),
        trace.short_probes.len() - exact_probes,
    );
}

fn report_fused<const N: usize>(
    name: &str,
    trace: &Trace,
    tables: &RustBoostTables,
    warmups: usize,
    iterations: usize,
) {
    let short_time = measure(warmups, iterations, trace.expected_short_checksum, || {
        let mut chunks = trace.short_probes.chunks_exact(N);
        let checksum = chunks.by_ref().fold(0u64, |checksum, keys| {
            tables
                .short_get_many::<N>(keys)
                .into_iter()
                .fold(checksum, |sum, value| sum.wrapping_add(u64::from(value)))
        });
        chunks.remainder().iter().fold(checksum, |sum, key| {
            sum.wrapping_add(u64::from(tables.short_get(key)))
        })
    });
    let long_time = measure(warmups, iterations, trace.expected_long_checksum, || {
        let mut chunks = trace.long_probes.chunks_exact(N);
        let checksum = chunks.by_ref().fold(0u64, |checksum, keys| {
            tables
                .long_get_many::<N>(keys)
                .into_iter()
                .fold(checksum, |sum, value| sum.wrapping_add(u64::from(value)))
        });
        chunks.remainder().iter().fold(checksum, |sum, key| {
            sum.wrapping_add(u64::from(tables.long_get(key)))
        })
    });
    let probes = trace.short_probes.len() + trace.long_probes.len();
    let total_ns = short_time.as_nanos() + long_time.as_nanos();
    let ns_per_probe = total_ns as f64 / probes as f64;
    println!(
        "hash,name={name},warmups={warmups},iterations={iterations},short_ms={:.6},long_ms={:.6},total_ms={:.6},ns_per_probe={ns_per_probe:.6},mprobes_s={:.6}",
        short_time.as_secs_f64() * 1e3,
        long_time.as_secs_f64() * 1e3,
        total_ns as f64 / 1e6,
        1e3 / ns_per_probe,
    );
}

struct RawTables {
    short: hashbrown::HashTable<(ShortKey, u16)>,
    long: hashbrown::HashTable<(u64, u16)>,
    short_hasher: hashbrown::DefaultHashBuilder,
    long_hasher: hashbrown::DefaultHashBuilder,
}

#[inline(always)]
fn short_hash(key: &ShortKey) -> u64 {
    let value = key.bytes ^ ((key.length as u64) << 56);
    let hash = value.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash ^ (hash >> 32)
}

#[inline(always)]
fn long_hash(mut value: u64) -> u64 {
    value ^= value >> 32;
    value = value.wrapping_mul(0xd6e8_feb8_6659_fd93);
    value ^ (value >> 32)
}

#[derive(Clone, Copy)]
#[repr(align(16))]
struct BoostGroup([u8; 16]);

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn extract_overflow(metadata: std::arch::x86_64::__m128i) -> u8 {
    let value: u32;
    unsafe {
        asm!(
            "psrldq {metadata}, 15",
            "movd {value:e}, {metadata}",
            metadata = inout(xmm_reg) metadata => _,
            value = lateout(reg) value,
            options(pure, nomem, nostack),
        );
    }
    value as u8
}

impl BoostGroup {
    const MATCH_WORDS: [u32; 256] = {
        let mut words = [0; 256];
        let mut index = 0;
        while index < 256 {
            let reduced = if index < 2 { index + 8 } else { index } as u32;
            words[index] = reduced * 0x0101_0101;
            index += 1;
        }
        words
    };

    #[inline(always)]
    fn reduced_hash(hash: u64) -> u8 {
        match hash as u8 {
            0 => 8,
            1 => 9,
            value => value,
        }
    }

    #[inline(always)]
    fn match_hash(&self, hash: u64) -> (u16, u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let metadata = _mm_load_si128(self.0.as_ptr().cast());
            let word = *Self::MATCH_WORDS.get_unchecked(hash as u8 as usize);
            let wanted = _mm_set1_epi32(word as i32);
            let candidates = (_mm_movemask_epi8(_mm_cmpeq_epi8(metadata, wanted)) as u16) & 0x7fff;
            let overflow = extract_overflow(metadata);
            (candidates, overflow)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let reduced = Self::reduced_hash(hash);
            let candidates = self.0[..15]
                .iter()
                .enumerate()
                .fold(0, |mask, (index, &value)| {
                    mask | (u16::from(value == reduced) << index)
                });
            (candidates, self.0[15])
        }
    }

    #[inline(always)]
    fn match_available(&self) -> u16 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let metadata = _mm_load_si128(self.0.as_ptr().cast());
            (_mm_movemask_epi8(_mm_cmpeq_epi8(metadata, _mm_setzero_si128())) as u16) & 0x7fff
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.0[..15]
                .iter()
                .enumerate()
                .fold(0, |mask, (index, &value)| {
                    mask | (u16::from(value == 0) << index)
                })
        }
    }

    #[inline(always)]
    fn is_not_overflowed(overflow: u8, hash: u64) -> bool {
        const SHIFTS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        overflow & unsafe { *SHIFTS.get_unchecked((hash & 7) as usize) } == 0
    }

    #[inline(always)]
    fn mark_overflow(&mut self, hash: u64) {
        self.0[15] |= 1 << (hash & 7);
    }
}

#[repr(C)]
struct BoostEntry<K> {
    key: K,
    value: u16,
}

/// Rust port of Boost.Unordered's non-concurrent FOA `group15` lookup layout.
struct RustBoostTable<K> {
    groups: Vec<BoostGroup>,
    entries: Vec<MaybeUninit<BoostEntry<K>>>,
    group_mask: usize,
    shift: u32,
}

impl<K: Copy + Eq> RustBoostTable<K> {
    const ENTRY_STRIDE: usize = 16;

    fn with_capacity(capacity: usize) -> Self {
        const GROUP_WIDTH: usize = 15;
        // Boost reserve(n): rehash(ceil(n / 0.875)), then select the smallest
        // power-of-two group count that can hold that slot request plus its
        // sentinel slot.
        let requested_slots = capacity.saturating_mul(8).div_ceil(7);
        let requested_groups = requested_slots / GROUP_WIDTH + 1;
        let group_count = requested_groups.max(2).next_power_of_two();
        let mut entries = Vec::with_capacity(group_count * Self::ENTRY_STRIDE);
        entries.resize_with(group_count * Self::ENTRY_STRIDE, MaybeUninit::uninit);
        Self {
            groups: (0..group_count).map(|_| BoostGroup([0; 16])).collect(),
            entries,
            group_mask: group_count - 1,
            shift: usize::BITS - group_count.trailing_zeros(),
        }
    }

    fn insert(&mut self, hash: u64, key: K, value: u16) {
        let mut position = (hash as usize) >> self.shift;
        let mut step = 0;
        loop {
            let available = self.groups[position].match_available();
            if available != 0 {
                let slot = available.trailing_zeros() as usize;
                self.entries[position * Self::ENTRY_STRIDE + slot].write(BoostEntry { key, value });
                self.groups[position].0[slot] = BoostGroup::reduced_hash(hash);
                return;
            }
            self.groups[position].mark_overflow(hash);
            step += 1;
            position = (position + step) & self.group_mask;
        }
    }

    #[inline(always)]
    fn get(&self, hash: u64, key: K) -> u16 {
        let position = (hash as usize) >> self.shift;
        self.get_from_position(hash, key, position)
    }

    #[inline(always)]
    fn get_from_position(&self, hash: u64, key: K, position: usize) -> u16 {
        let group = unsafe { self.groups.get_unchecked(position) };
        let (mut candidates, overflow) = group.match_hash(hash);
        if candidates != 0 {
            let first = position * Self::ENTRY_STRIDE;
            #[cfg(target_arch = "x86_64")]
            unsafe {
                _mm_prefetch(self.entries.as_ptr().add(first).cast(), _MM_HINT_T0);
            }
            while candidates != 0 {
                let slot = candidates.trailing_zeros() as usize;
                let entry = unsafe { self.entries.get_unchecked(first + slot).assume_init_ref() };
                if entry.key == key {
                    return entry.value;
                }
                candidates &= candidates - 1;
            }
        }
        if BoostGroup::is_not_overflowed(overflow, hash) {
            0
        } else {
            self.get_overflow(hash, key, position)
        }
    }

    #[inline(always)]
    fn resolve_initial(
        &self,
        hash: u64,
        key: K,
        position: usize,
        mut candidates: u16,
        overflow: u8,
    ) -> u16 {
        if candidates != 0 {
            let first = position * Self::ENTRY_STRIDE;
            #[cfg(target_arch = "x86_64")]
            unsafe {
                _mm_prefetch(self.entries.as_ptr().add(first).cast(), _MM_HINT_T0);
            }
            while candidates != 0 {
                let slot = candidates.trailing_zeros() as usize;
                let entry = unsafe { self.entries.get_unchecked(first + slot).assume_init_ref() };
                if entry.key == key {
                    return entry.value;
                }
                candidates &= candidates - 1;
            }
        }
        if BoostGroup::is_not_overflowed(overflow, hash) {
            0
        } else {
            self.get_overflow(hash, key, position)
        }
    }

    #[inline(always)]
    fn get_many<const N: usize>(&self, hashes: [u64; N], keys: [K; N]) -> [u16; N] {
        let mut positions = [0usize; N];
        let mut candidates = [0u16; N];
        let mut overflows = [0u8; N];
        let mut lane = 0;
        while lane < N {
            positions[lane] = (hashes[lane] as usize) >> self.shift;
            (candidates[lane], overflows[lane]) = unsafe {
                self.groups
                    .get_unchecked(positions[lane])
                    .match_hash(hashes[lane])
            };
            lane += 1;
        }

        let mut values = [0u16; N];
        lane = 0;
        while lane < N {
            values[lane] = self.resolve_initial(
                hashes[lane],
                keys[lane],
                positions[lane],
                candidates[lane],
                overflows[lane],
            );
            lane += 1;
        }
        values
    }

    #[cold]
    #[inline(never)]
    fn get_overflow(&self, hash: u64, key: K, mut position: usize) -> u16 {
        let mut step = 0;
        loop {
            step += 1;
            position = (position + step) & self.group_mask;
            // SAFETY: the initial high-bit position and every quadratic step
            // are constrained to `group_mask`.
            let group = unsafe { self.groups.get_unchecked(position) };
            let (mut candidates, overflow) = group.match_hash(hash);
            if candidates != 0 {
                let first = position * Self::ENTRY_STRIDE;
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    _mm_prefetch(self.entries.as_ptr().add(first).cast(), _MM_HINT_T0);
                }
                while candidates != 0 {
                    let slot = candidates.trailing_zeros() as usize;
                    // SAFETY: matching nonzero metadata is written only after
                    // its corresponding entry has been initialized.
                    let entry =
                        unsafe { self.entries.get_unchecked(first + slot).assume_init_ref() };
                    if entry.key == key {
                        return entry.value;
                    }
                    candidates &= candidates - 1;
                }
            }
            if BoostGroup::is_not_overflowed(overflow, hash) {
                return 0;
            }
        }
    }

    fn probe_stats(&self, hash: u64, key: K) -> (usize, usize, bool) {
        let mut position = (hash as usize) >> self.shift;
        let mut step = 0;
        let mut groups = 0;
        let mut comparisons = 0;
        loop {
            groups += 1;
            let group = &self.groups[position];
            let (mut candidates, overflow) = group.match_hash(hash);
            let first = position * Self::ENTRY_STRIDE;
            while candidates != 0 {
                let slot = candidates.trailing_zeros() as usize;
                comparisons += 1;
                let entry = unsafe { self.entries[first + slot].assume_init_ref() };
                if entry.key == key {
                    return (groups, comparisons, true);
                }
                candidates &= candidates - 1;
            }
            if BoostGroup::is_not_overflowed(overflow, hash) {
                return (groups, comparisons, false);
            }
            step += 1;
            position = (position + step) & self.group_mask;
        }
    }
}

/// Immutable Group15 lookup table with keys and values stored separately.
///
/// Construction first uses the reference insertion algorithm, then compacts each
/// 15-entry group into its final read-only layout. `STRIDE=15` removes the unused
/// sentinel entry; `STRIDE=16` retains shift-friendly group spacing.
struct FrozenSplitTable<K, const STRIDE: usize> {
    groups: Vec<BoostGroup>,
    keys: Vec<MaybeUninit<K>>,
    values: Vec<MaybeUninit<u16>>,
    group_mask: usize,
    shift: u32,
}

impl<K: Copy + Eq, const STRIDE: usize> FrozenSplitTable<K, STRIDE> {
    fn freeze(source: &RustBoostTable<K>) -> Self {
        assert!(STRIDE >= 15);
        let slot_count = source.groups.len() * STRIDE;
        let mut keys = Vec::with_capacity(slot_count);
        keys.resize_with(slot_count, MaybeUninit::uninit);
        let mut values = Vec::with_capacity(slot_count);
        values.resize_with(slot_count, MaybeUninit::uninit);

        for position in 0..source.groups.len() {
            for slot in 0..15 {
                if source.groups[position].0[slot] == 0 {
                    continue;
                }
                let entry = unsafe {
                    source.entries[position * RustBoostTable::<K>::ENTRY_STRIDE + slot]
                        .assume_init_ref()
                };
                let frozen_slot = position * STRIDE + slot;
                keys[frozen_slot].write(entry.key);
                values[frozen_slot].write(entry.value);
            }
        }

        Self {
            groups: source.groups.clone(),
            keys,
            values,
            group_mask: source.group_mask,
            shift: source.shift,
        }
    }

    #[inline(always)]
    fn resolve_group(&self, key: K, position: usize, mut candidates: u16) -> Option<u16> {
        if candidates == 0 {
            return None;
        }
        let first = position * STRIDE;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_prefetch(self.keys.as_ptr().add(first).cast(), _MM_HINT_T0);
        }
        while candidates != 0 {
            let slot = candidates.trailing_zeros() as usize;
            let frozen_slot = first + slot;
            let candidate = unsafe { self.keys.get_unchecked(frozen_slot).assume_init_ref() };
            if *candidate == key {
                return Some(unsafe { *self.values.get_unchecked(frozen_slot).assume_init_ref() });
            }
            candidates &= candidates - 1;
        }
        None
    }

    #[inline(always)]
    fn get(&self, hash: u64, key: K) -> u16 {
        let position = (hash as usize) >> self.shift;
        let group = unsafe { self.groups.get_unchecked(position) };
        let (candidates, overflow) = group.match_hash(hash);
        if let Some(value) = self.resolve_group(key, position, candidates) {
            return value;
        }
        if BoostGroup::is_not_overflowed(overflow, hash) {
            0
        } else {
            self.get_overflow(hash, key, position)
        }
    }

    #[cold]
    #[inline(never)]
    fn get_overflow(&self, hash: u64, key: K, mut position: usize) -> u16 {
        let mut step = 0;
        loop {
            step += 1;
            position = (position + step) & self.group_mask;
            let group = unsafe { self.groups.get_unchecked(position) };
            let (candidates, overflow) = group.match_hash(hash);
            if let Some(value) = self.resolve_group(key, position, candidates) {
                return value;
            }
            if BoostGroup::is_not_overflowed(overflow, hash) {
                return 0;
            }
        }
    }

    fn storage_bytes(&self) -> usize {
        self.groups.len() * std::mem::size_of::<BoostGroup>()
            + self.keys.len() * std::mem::size_of::<K>()
            + self.values.len() * std::mem::size_of::<u16>()
    }
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct PackedLongEntry {
    key: u64,
    value: u16,
}

/// Frozen long-key table with exactly 15 unaligned 10-byte entries per group.
struct FrozenPackedLongTable {
    groups: Vec<BoostGroup>,
    entries: Vec<MaybeUninit<PackedLongEntry>>,
    group_mask: usize,
    shift: u32,
}

impl FrozenPackedLongTable {
    const STRIDE: usize = 15;

    fn freeze(source: &RustBoostTable<u64>) -> Self {
        let slot_count = source.groups.len() * Self::STRIDE;
        let mut entries = Vec::with_capacity(slot_count);
        entries.resize_with(slot_count, MaybeUninit::uninit);
        for position in 0..source.groups.len() {
            for slot in 0..15 {
                if source.groups[position].0[slot] == 0 {
                    continue;
                }
                let entry = unsafe {
                    source.entries[position * RustBoostTable::<u64>::ENTRY_STRIDE + slot]
                        .assume_init_ref()
                };
                entries[position * Self::STRIDE + slot].write(PackedLongEntry {
                    key: entry.key,
                    value: entry.value,
                });
            }
        }
        Self {
            groups: source.groups.clone(),
            entries,
            group_mask: source.group_mask,
            shift: source.shift,
        }
    }

    #[inline(always)]
    fn resolve_group(&self, key: u64, position: usize, mut candidates: u16) -> Option<u16> {
        if candidates == 0 {
            return None;
        }
        let first = position * Self::STRIDE;
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_prefetch(self.entries.as_ptr().add(first).cast(), _MM_HINT_T0);
        }
        while candidates != 0 {
            let slot = candidates.trailing_zeros() as usize;
            let entry = unsafe {
                self.entries
                    .as_ptr()
                    .add(first + slot)
                    .cast::<PackedLongEntry>()
            };
            let candidate = unsafe { std::ptr::addr_of!((*entry).key).read_unaligned() };
            if candidate == key {
                return Some(unsafe { std::ptr::addr_of!((*entry).value).read_unaligned() });
            }
            candidates &= candidates - 1;
        }
        None
    }

    #[inline(always)]
    fn get(&self, hash: u64, key: u64) -> u16 {
        let position = (hash as usize) >> self.shift;
        let group = unsafe { self.groups.get_unchecked(position) };
        let (candidates, overflow) = group.match_hash(hash);
        if let Some(value) = self.resolve_group(key, position, candidates) {
            return value;
        }
        if BoostGroup::is_not_overflowed(overflow, hash) {
            0
        } else {
            self.get_overflow(hash, key, position)
        }
    }

    #[cold]
    #[inline(never)]
    fn get_overflow(&self, hash: u64, key: u64, mut position: usize) -> u16 {
        let mut step = 0;
        loop {
            step += 1;
            position = (position + step) & self.group_mask;
            let group = unsafe { self.groups.get_unchecked(position) };
            let (candidates, overflow) = group.match_hash(hash);
            if let Some(value) = self.resolve_group(key, position, candidates) {
                return value;
            }
            if BoostGroup::is_not_overflowed(overflow, hash) {
                return 0;
            }
        }
    }

    fn storage_bytes(&self) -> usize {
        self.groups.len() * std::mem::size_of::<BoostGroup>()
            + self.entries.len() * std::mem::size_of::<PackedLongEntry>()
    }
}

struct FrozenSplitTables<const STRIDE: usize> {
    short: FrozenSplitTable<ShortKey, STRIDE>,
    long: FrozenSplitTable<u64, STRIDE>,
}

impl<const STRIDE: usize> FrozenSplitTables<STRIDE> {
    fn freeze(source: &RustBoostTables) -> Self {
        Self {
            short: FrozenSplitTable::freeze(&source.short),
            long: FrozenSplitTable::freeze(&source.long),
        }
    }

    #[inline(always)]
    fn short_get(&self, key: &ShortKey) -> u16 {
        self.short.get(short_hash(key), *key)
    }

    #[inline(always)]
    fn long_get(&self, key: &u64) -> u16 {
        self.long.get(long_hash(*key), *key)
    }

    fn storage_bytes(&self) -> usize {
        self.short.storage_bytes() + self.long.storage_bytes()
    }
}

/// One-hash negative filter for a read-only table.
///
/// The Group15 hash is already avalanching, so one middle-hash bit index is
/// enough. False positives fall through to the exact table; false negatives are
/// impossible. A power-of-two bit count keeps lookup to a shift, mask, load, and
/// bit test.
struct FrozenBloom {
    words: Box<[u64]>,
    bit_mask: usize,
}

impl FrozenBloom {
    fn from_hashes(
        hashes: impl IntoIterator<Item = u64>,
        key_count: usize,
        bits_per_key: usize,
    ) -> Self {
        let bit_count = key_count
            .saturating_mul(bits_per_key)
            .max(64)
            .next_power_of_two();
        let mut words = vec![0u64; bit_count / 64].into_boxed_slice();
        let bit_mask = bit_count - 1;
        for hash in hashes {
            let bit = ((hash >> 8) as usize) & bit_mask;
            words[bit >> 6] |= 1u64 << (bit & 63);
        }
        Self { words, bit_mask }
    }

    #[inline(always)]
    fn may_contain(&self, hash: u64) -> bool {
        let bit = ((hash >> 8) as usize) & self.bit_mask;
        (unsafe { *self.words.get_unchecked(bit >> 6) } & (1u64 << (bit & 63))) != 0
    }

    fn storage_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }
}

/// Blocked Bloom filter: all probes share one 64-bit word load.
struct FrozenBlockedBloom<const PROBES: usize, const THIRD_SHIFT: u32 = 48> {
    words: Box<[u64]>,
    word_mask: usize,
}

impl<const PROBES: usize, const THIRD_SHIFT: u32> FrozenBlockedBloom<PROBES, THIRD_SHIFT> {
    fn from_hashes(
        hashes: impl IntoIterator<Item = u64>,
        key_count: usize,
        bits_per_key: usize,
    ) -> Self {
        let word_count = key_count
            .saturating_mul(bits_per_key)
            .div_ceil(64)
            .max(1)
            .next_power_of_two();
        let mut words = vec![0u64; word_count].into_boxed_slice();
        let word_mask = word_count - 1;
        for hash in hashes {
            let word = ((hash >> 8) as usize) & word_mask;
            words[word] |= Self::bit_mask(hash);
        }
        Self { words, word_mask }
    }

    #[inline(always)]
    fn bit_mask(hash: u64) -> u64 {
        let mut mask = 1u64 << (hash & 63);
        if PROBES >= 2 {
            mask |= 1u64 << ((hash >> 32) & 63);
        }
        if PROBES >= 3 {
            mask |= 1u64 << ((hash >> THIRD_SHIFT) & 63);
        }
        mask
    }

    #[inline(always)]
    fn may_contain(&self, hash: u64) -> bool {
        let word = ((hash >> 8) as usize) & self.word_mask;
        let mask = Self::bit_mask(hash);
        (unsafe { *self.words.get_unchecked(word) } & mask) == mask
    }

    fn storage_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }
}

/// Three independent prefix filters used to bound OnPair's descending
/// short-token search before any exact hash-table lookup.
struct FrozenPrefixBounds<const BITS_PER_KEY: usize> {
    prefix8: FrozenBlockedBloom<3>,
    prefix6: FrozenBlockedBloom<3>,
    prefix4: FrozenBlockedBloom<3>,
}

#[derive(Clone, Copy)]
struct PrefixProbe {
    bound: u8,
    filters: u8,
    hash8: u64,
    hash6: u64,
    hash4: u64,
}

impl PrefixProbe {
    #[inline(always)]
    fn hash_for(&self, length: u8) -> Option<u64> {
        match length {
            8 if self.filters & 0b100 != 0 => Some(self.hash8),
            6 if self.filters & 0b010 != 0 => Some(self.hash6),
            4 if self.filters & 0b001 != 0 => Some(self.hash4),
            _ => None,
        }
    }
}

impl<const BITS_PER_KEY: usize> FrozenPrefixBounds<BITS_PER_KEY> {
    fn build(entries: &[(ShortKey, u16)]) -> Self {
        fn hashes(entries: &[(ShortKey, u16)], length: u8) -> Vec<u64> {
            let mask = if length == 8 {
                u64::MAX
            } else {
                (1u64 << (u32::from(length) * 8)) - 1
            };
            let mut hashes: Vec<_> = entries
                .iter()
                .filter(|(key, _)| key.length >= length)
                .map(|(key, _)| {
                    short_hash(&ShortKey {
                        bytes: key.bytes & mask,
                        length,
                    })
                })
                .collect();
            hashes.sort_unstable();
            hashes.dedup();
            hashes
        }

        let hashes8 = hashes(entries, 8);
        let hashes6 = hashes(entries, 6);
        let hashes4 = hashes(entries, 4);
        Self {
            prefix8: FrozenBlockedBloom::from_hashes(
                hashes8.iter().copied(),
                hashes8.len(),
                BITS_PER_KEY,
            ),
            prefix6: FrozenBlockedBloom::from_hashes(
                hashes6.iter().copied(),
                hashes6.len(),
                BITS_PER_KEY,
            ),
            prefix4: FrozenBlockedBloom::from_hashes(
                hashes4.iter().copied(),
                hashes4.len(),
                BITS_PER_KEY,
            ),
        }
    }

    #[inline(always)]
    fn probe<const FILTERS: u8>(&self, input: &ShortKey) -> PrefixProbe {
        let bytes = input.bytes;
        let hash8 = if FILTERS & 0b100 != 0 {
            short_hash(&ShortKey { bytes, length: 8 })
        } else {
            0
        };
        let hash6 = if FILTERS & 0b010 != 0 {
            short_hash(&ShortKey {
                bytes: bytes & 0x0000_ffff_ffff_ffff,
                length: 6,
            })
        } else {
            0
        };
        let hash4 = if FILTERS & 0b001 != 0 {
            short_hash(&ShortKey {
                bytes: bytes & 0x0000_0000_ffff_ffff,
                length: 4,
            })
        } else {
            0
        };

        // Materialize all three addresses and words before testing any result;
        // these loads are independent and can overlap on the miss-heavy path.
        let may8 = FILTERS & 0b100 == 0 || self.prefix8.may_contain(hash8);
        let may6 = FILTERS & 0b010 == 0 || self.prefix6.may_contain(hash6);
        let may4 = FILTERS & 0b001 == 0 || self.prefix4.may_contain(hash4);

        let mut bound = input.length;
        if FILTERS & 0b100 != 0 && bound >= 8 && !may8 {
            bound = 7;
        }
        if FILTERS & 0b010 != 0 && bound >= 6 && !may6 {
            bound = 5;
        }
        if FILTERS & 0b001 != 0 && bound >= 4 && !may4 {
            bound = 3;
        }
        PrefixProbe {
            bound,
            filters: FILTERS,
            hash8,
            hash6,
            hash4,
        }
    }

    fn storage_bytes(&self) -> usize {
        self.prefix8.storage_bytes() + self.prefix6.storage_bytes() + self.prefix4.storage_bytes()
    }
}

/// Immutable directory from a four-byte prefix to the exact set of short-token
/// lengths that can follow it. One lookup replaces multiple approximate prefix
/// probes and can eliminate holes inside the maximum-length bound.
struct FrozenPrefixLengthMask {
    table: FrozenSplitTable<u32, 15>,
}

#[derive(Clone, Copy)]
struct PrefixLengthMaskProbe {
    lengths: u16,
    hash: u64,
}

impl FrozenPrefixLengthMask {
    fn build(entries: &[(ShortKey, u16)]) -> Self {
        let mut masks = BTreeMap::<u32, u16>::new();
        for (key, _) in entries.iter().filter(|(key, _)| key.length >= 4) {
            let prefix = key.bytes as u32;
            *masks.entry(prefix).or_default() |= 1 << key.length;
        }

        let mut table = RustBoostTable::with_capacity(masks.len());
        for (prefix, lengths) in masks {
            let hash = short_hash(&ShortKey {
                bytes: u64::from(prefix),
                length: 4,
            });
            table.insert(hash, prefix, lengths);
        }
        Self {
            table: FrozenSplitTable::freeze(&table),
        }
    }

    #[inline(always)]
    fn get(&self, input: &ShortKey) -> PrefixLengthMaskProbe {
        if input.length < 4 {
            return PrefixLengthMaskProbe {
                lengths: 0,
                hash: 0,
            };
        }
        let prefix = input.bytes as u32;
        let hash = short_hash(&ShortKey {
            bytes: u64::from(prefix),
            length: 4,
        });
        PrefixLengthMaskProbe {
            lengths: self.table.get(hash, prefix),
            hash,
        }
    }

    fn storage_bytes(&self) -> usize {
        self.table.storage_bytes()
    }
}

/// A lossy length-mask directory: colliding four-byte prefixes OR their masks
/// into the same byte. It has Bloom-filter semantics (false positives only),
/// but one metadata load returns all viable short-token lengths at once.
struct FrozenApproxPrefixLengthMask<const SLOTS_PER_KEY: usize> {
    masks: Box<[u8]>,
    slot_mask: usize,
}

impl<const SLOTS_PER_KEY: usize> FrozenApproxPrefixLengthMask<SLOTS_PER_KEY> {
    fn build(entries: &[(ShortKey, u16)]) -> Self {
        let mut prefix_masks = BTreeMap::<u32, u8>::new();
        for (key, _) in entries.iter().filter(|(key, _)| key.length >= 4) {
            let prefix = key.bytes as u32;
            *prefix_masks.entry(prefix).or_default() |= 1 << (key.length - 4);
        }
        let slot_count = prefix_masks
            .len()
            .saturating_mul(SLOTS_PER_KEY)
            .max(1)
            .next_power_of_two();
        let slot_mask = slot_count - 1;
        let mut masks = vec![0u8; slot_count].into_boxed_slice();
        for (prefix, lengths) in prefix_masks {
            let hash = short_hash(&ShortKey {
                bytes: u64::from(prefix),
                length: 4,
            });
            masks[((hash >> 8) as usize) & slot_mask] |= lengths;
        }
        Self { masks, slot_mask }
    }

    #[inline(always)]
    fn get(&self, input: &ShortKey) -> PrefixLengthMaskProbe {
        if input.length < 4 {
            return PrefixLengthMaskProbe {
                lengths: 0,
                hash: 0,
            };
        }
        let prefix = input.bytes as u32;
        let hash = short_hash(&ShortKey {
            bytes: u64::from(prefix),
            length: 4,
        });
        let packed = unsafe {
            *self
                .masks
                .get_unchecked(((hash >> 8) as usize) & self.slot_mask)
        };
        PrefixLengthMaskProbe {
            lengths: u16::from(packed) << 4,
            hash,
        }
    }

    fn storage_bytes(&self) -> usize {
        self.masks.len()
    }
}

struct FrozenBlockedBloomTables<
    'a,
    const BITS_PER_KEY: usize,
    const PROBES: usize,
    const THIRD_SHIFT: u32 = 48,
> {
    short_table: &'a RustBoostTable<ShortKey>,
    long_table: &'a RustBoostTable<u64>,
    short_filter: FrozenBlockedBloom<PROBES, THIRD_SHIFT>,
    long_filter: FrozenBlockedBloom<PROBES, THIRD_SHIFT>,
}

/// One blocked Bloom word per immutable Group15 home group.
///
/// Matching the filter and table indexes lets lookup reuse the computed home
/// position. The filter is populated by home position rather than final entry
/// position so it remains valid when insertion overflowed into another group.
struct FrozenHomeBloom<const WORDS_PER_GROUP: usize, const PROBES: usize> {
    words: Box<[u64]>,
}

impl<const WORDS_PER_GROUP: usize, const PROBES: usize> FrozenHomeBloom<WORDS_PER_GROUP, PROBES> {
    fn from_hashes(hashes: impl IntoIterator<Item = u64>, shift: u32, group_count: usize) -> Self {
        assert!(WORDS_PER_GROUP.is_power_of_two());
        let mut words = vec![0u64; group_count * WORDS_PER_GROUP].into_boxed_slice();
        for hash in hashes {
            let position = (hash as usize) >> shift;
            let selector = ((hash >> 16) as usize) & (WORDS_PER_GROUP - 1);
            words[position * WORDS_PER_GROUP + selector] |=
                FrozenBlockedBloom::<PROBES>::bit_mask(hash);
        }
        Self { words }
    }

    #[inline(always)]
    fn may_contain(&self, position: usize, hash: u64) -> bool {
        let selector = ((hash >> 16) as usize) & (WORDS_PER_GROUP - 1);
        let mask = FrozenBlockedBloom::<PROBES>::bit_mask(hash);
        (unsafe {
            *self
                .words
                .get_unchecked(position * WORDS_PER_GROUP + selector)
        } & mask)
            == mask
    }

    fn storage_bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u64>()
    }
}

struct FrozenHomeBloomTables<'a, const WORDS_PER_GROUP: usize, const PROBES: usize> {
    short_table: &'a RustBoostTable<ShortKey>,
    long_table: &'a RustBoostTable<u64>,
    short_filter: FrozenHomeBloom<WORDS_PER_GROUP, PROBES>,
    long_filter: FrozenHomeBloom<WORDS_PER_GROUP, PROBES>,
}

impl<'a, const WORDS_PER_GROUP: usize, const PROBES: usize>
    FrozenHomeBloomTables<'a, WORDS_PER_GROUP, PROBES>
{
    fn build(source: &'a RustBoostTables, trace: &Trace) -> Self {
        Self {
            short_table: &source.short,
            long_table: &source.long,
            short_filter: FrozenHomeBloom::from_hashes(
                trace.short_entries.iter().map(|(key, _)| short_hash(key)),
                source.short.shift,
                source.short.groups.len(),
            ),
            long_filter: FrozenHomeBloom::from_hashes(
                trace.long_entries.iter().map(|(key, _)| long_hash(*key)),
                source.long.shift,
                source.long.groups.len(),
            ),
        }
    }

    #[inline(always)]
    fn short_get(&self, key: &ShortKey) -> u16 {
        let hash = short_hash(key);
        let position = (hash as usize) >> self.short_table.shift;
        if self.short_filter.may_contain(position, hash) {
            self.short_table.get_from_position(hash, *key, position)
        } else {
            0
        }
    }

    #[inline(always)]
    fn long_get(&self, key: &u64) -> u16 {
        let hash = long_hash(*key);
        let position = (hash as usize) >> self.long_table.shift;
        if self.long_filter.may_contain(position, hash) {
            self.long_table.get_from_position(hash, *key, position)
        } else {
            0
        }
    }

    fn storage_bytes(&self) -> usize {
        self.short_filter.storage_bytes() + self.long_filter.storage_bytes()
    }
}

impl<'a, const BITS_PER_KEY: usize, const PROBES: usize, const THIRD_SHIFT: u32>
    FrozenBlockedBloomTables<'a, BITS_PER_KEY, PROBES, THIRD_SHIFT>
{
    fn build(source: &'a RustBoostTables, trace: &Trace) -> Self {
        Self {
            short_table: &source.short,
            long_table: &source.long,
            short_filter: FrozenBlockedBloom::from_hashes(
                trace.short_entries.iter().map(|(key, _)| short_hash(key)),
                trace.short_entries.len(),
                BITS_PER_KEY,
            ),
            long_filter: FrozenBlockedBloom::from_hashes(
                trace.long_entries.iter().map(|(key, _)| long_hash(*key)),
                trace.long_entries.len(),
                BITS_PER_KEY,
            ),
        }
    }

    #[inline(always)]
    fn short_get(&self, key: &ShortKey) -> u16 {
        let hash = short_hash(key);
        if self.short_filter.may_contain(hash) {
            self.short_table.get(hash, *key)
        } else {
            0
        }
    }

    #[inline(always)]
    fn long_get(&self, key: &u64) -> u16 {
        let hash = long_hash(*key);
        if self.long_filter.may_contain(hash) {
            self.long_table.get(hash, *key)
        } else {
            0
        }
    }

    fn storage_bytes(&self) -> usize {
        self.short_filter.storage_bytes() + self.long_filter.storage_bytes()
    }
}

/// Frozen filter policy specialized for OnPair's two lookup streams.
///
/// Short probes are miss-heavy enough to benefit from three Bloom bits. Long
/// probes have a substantially higher hit rate, making two bits cheaper there.
struct FrozenOnPairTables<'a> {
    short_table: &'a RustBoostTable<ShortKey>,
    long_table: &'a RustBoostTable<u64>,
    short_filter: FrozenBlockedBloom<3>,
    long_filter: FrozenBlockedBloom<2>,
}

impl<'a> FrozenOnPairTables<'a> {
    fn build(source: &'a RustBoostTables, trace: &Trace) -> Self {
        Self {
            short_table: &source.short,
            long_table: &source.long,
            short_filter: FrozenBlockedBloom::from_hashes(
                trace.short_entries.iter().map(|(key, _)| short_hash(key)),
                trace.short_entries.len(),
                16,
            ),
            long_filter: FrozenBlockedBloom::from_hashes(
                trace.long_entries.iter().map(|(key, _)| long_hash(*key)),
                trace.long_entries.len(),
                16,
            ),
        }
    }

    #[inline(always)]
    fn short_get(&self, key: &ShortKey) -> u16 {
        let hash = short_hash(key);
        if self.short_filter.may_contain(hash) {
            self.short_table.get(hash, *key)
        } else {
            0
        }
    }

    #[inline(always)]
    fn long_get(&self, key: &u64) -> u16 {
        let hash = long_hash(*key);
        if self.long_filter.may_contain(hash) {
            self.long_table.get(hash, *key)
        } else {
            0
        }
    }

    fn storage_bytes(&self) -> usize {
        self.short_filter.storage_bytes() + self.long_filter.storage_bytes()
    }
}

struct FrozenBloomTables<'a, const BITS_PER_KEY: usize> {
    short_table: &'a RustBoostTable<ShortKey>,
    long_table: &'a RustBoostTable<u64>,
    short_filter: FrozenBloom,
    long_filter: FrozenBloom,
}

impl<'a, const BITS_PER_KEY: usize> FrozenBloomTables<'a, BITS_PER_KEY> {
    fn build(source: &'a RustBoostTables, trace: &Trace) -> Self {
        Self {
            short_table: &source.short,
            long_table: &source.long,
            short_filter: FrozenBloom::from_hashes(
                trace.short_entries.iter().map(|(key, _)| short_hash(key)),
                trace.short_entries.len(),
                BITS_PER_KEY,
            ),
            long_filter: FrozenBloom::from_hashes(
                trace.long_entries.iter().map(|(key, _)| long_hash(*key)),
                trace.long_entries.len(),
                BITS_PER_KEY,
            ),
        }
    }

    #[inline(always)]
    fn short_get(&self, key: &ShortKey) -> u16 {
        let hash = short_hash(key);
        if self.short_filter.may_contain(hash) {
            self.short_table.get(hash, *key)
        } else {
            0
        }
    }

    #[inline(always)]
    fn long_get(&self, key: &u64) -> u16 {
        let hash = long_hash(*key);
        if self.long_filter.may_contain(hash) {
            self.long_table.get(hash, *key)
        } else {
            0
        }
    }

    fn storage_bytes(&self) -> usize {
        self.short_filter.storage_bytes() + self.long_filter.storage_bytes()
    }
}

/// Static three-way XOR filter. Construction peels a random hypergraph and may
/// try several cheap hash rotations; lookup has a 1/256 false-positive rate.
struct FrozenXor8 {
    fingerprints: Box<[u8]>,
    block_length: usize,
    rotation: u32,
}

impl FrozenXor8 {
    const ROTATIONS: [u32; 8] = [7, 13, 17, 23, 29, 31, 37, 43];

    fn build(hashes: &[u64]) -> Option<Self> {
        let block_length = hashes
            .len()
            .saturating_mul(13)
            .div_ceil(30)
            .max(2)
            .next_power_of_two();
        let size = block_length * 3;

        for rotation in Self::ROTATIONS {
            let mut degrees = vec![0u32; size];
            let mut xors = vec![0u64; size];
            for &base_hash in hashes {
                let hash = Self::derive(base_hash, rotation);
                for position in Self::positions(hash, block_length) {
                    degrees[position] += 1;
                    xors[position] ^= hash;
                }
            }

            let mut queue = VecDeque::new();
            for (position, &degree) in degrees.iter().enumerate() {
                if degree == 1 {
                    queue.push_back(position);
                }
            }
            let mut peeled = Vec::with_capacity(hashes.len());
            while let Some(selected) = queue.pop_front() {
                if degrees[selected] != 1 {
                    continue;
                }
                let hash = xors[selected];
                peeled.push((hash, selected));
                degrees[selected] = 0;
                for position in Self::positions(hash, block_length) {
                    if position == selected || degrees[position] == 0 {
                        continue;
                    }
                    degrees[position] -= 1;
                    xors[position] ^= hash;
                    if degrees[position] == 1 {
                        queue.push_back(position);
                    }
                }
            }
            if peeled.len() != hashes.len() {
                continue;
            }

            let mut fingerprints = vec![0u8; size].into_boxed_slice();
            for &(hash, selected) in peeled.iter().rev() {
                let mut fingerprint = Self::fingerprint(hash);
                for position in Self::positions(hash, block_length) {
                    if position != selected {
                        fingerprint ^= fingerprints[position];
                    }
                }
                fingerprints[selected] = fingerprint;
            }
            return Some(Self {
                fingerprints,
                block_length,
                rotation,
            });
        }
        None
    }

    #[inline(always)]
    fn derive(hash: u64, rotation: u32) -> u64 {
        hash ^ hash.rotate_left(rotation)
    }

    #[inline(always)]
    fn positions(hash: u64, block_length: usize) -> [usize; 3] {
        let mask = block_length - 1;
        [
            (hash as usize) & mask,
            block_length + ((hash >> 21) as usize & mask),
            block_length * 2 + ((hash >> 42) as usize & mask),
        ]
    }

    #[inline(always)]
    fn fingerprint(hash: u64) -> u8 {
        (hash >> 56) as u8
    }

    #[inline(always)]
    fn may_contain(&self, base_hash: u64) -> bool {
        let hash = Self::derive(base_hash, self.rotation);
        let [first, second, third] = Self::positions(hash, self.block_length);
        let fingerprint = unsafe {
            *self.fingerprints.get_unchecked(first)
                ^ *self.fingerprints.get_unchecked(second)
                ^ *self.fingerprints.get_unchecked(third)
        };
        fingerprint == Self::fingerprint(hash)
    }

    fn storage_bytes(&self) -> usize {
        self.fingerprints.len()
    }
}

struct FrozenXor8Tables<'a> {
    short_table: &'a RustBoostTable<ShortKey>,
    long_table: &'a RustBoostTable<u64>,
    short_filter: FrozenXor8,
    long_filter: FrozenXor8,
}

impl<'a> FrozenXor8Tables<'a> {
    fn build(source: &'a RustBoostTables, trace: &Trace) -> Option<Self> {
        let short_hashes: Vec<_> = trace
            .short_entries
            .iter()
            .map(|(key, _)| short_hash(key))
            .collect();
        let long_hashes: Vec<_> = trace
            .long_entries
            .iter()
            .map(|(key, _)| long_hash(*key))
            .collect();
        Some(Self {
            short_table: &source.short,
            long_table: &source.long,
            short_filter: FrozenXor8::build(&short_hashes)?,
            long_filter: FrozenXor8::build(&long_hashes)?,
        })
    }

    #[inline(always)]
    fn short_get(&self, key: &ShortKey) -> u16 {
        let hash = short_hash(key);
        if self.short_filter.may_contain(hash) {
            self.short_table.get(hash, *key)
        } else {
            0
        }
    }

    #[inline(always)]
    fn long_get(&self, key: &u64) -> u16 {
        let hash = long_hash(*key);
        if self.long_filter.may_contain(hash) {
            self.long_table.get(hash, *key)
        } else {
            0
        }
    }

    fn storage_bytes(&self) -> usize {
        self.short_filter.storage_bytes() + self.long_filter.storage_bytes()
    }
}

struct RustBoostTables {
    short: RustBoostTable<ShortKey>,
    long: RustBoostTable<u64>,
}

impl RustBoostTables {
    fn new(trace: &Trace) -> Self {
        let mut short = RustBoostTable::with_capacity(trace.short_entries.len());
        for &(key, value) in &trace.short_entries {
            short.insert(short_hash(&key), key, value);
        }
        let mut long = RustBoostTable::with_capacity(trace.long_entries.len());
        for &(key, value) in &trace.long_entries {
            long.insert(long_hash(key), key, value);
        }
        Self { short, long }
    }

    #[inline(always)]
    fn short_get(&self, key: &ShortKey) -> u16 {
        self.short.get(short_hash(key), *key)
    }

    #[inline(always)]
    fn long_get(&self, key: &u64) -> u16 {
        self.long.get(long_hash(*key), *key)
    }

    #[inline(always)]
    fn short_get_many<const N: usize>(&self, keys: &[ShortKey]) -> [u16; N] {
        let keys = std::array::from_fn(|lane| keys[lane]);
        let hashes = std::array::from_fn(|lane| short_hash(&keys[lane]));
        self.short.get_many(hashes, keys)
    }

    #[inline(always)]
    fn long_get_many<const N: usize>(&self, keys: &[u64]) -> [u16; N] {
        let keys = std::array::from_fn(|lane| keys[lane]);
        let hashes = std::array::from_fn(|lane| long_hash(keys[lane]));
        self.long.get_many(hashes, keys)
    }
}

impl RawTables {
    fn new(trace: &Trace) -> Self {
        let mut result = Self {
            short: hashbrown::HashTable::with_capacity(trace.short_entries.len()),
            long: hashbrown::HashTable::with_capacity(trace.long_entries.len()),
            short_hasher: hashbrown::DefaultHashBuilder::default(),
            long_hasher: hashbrown::DefaultHashBuilder::default(),
        };
        for &(key, value) in &trace.short_entries {
            let hash = result.short_hasher.hash_one(key);
            let hasher = &result.short_hasher;
            result
                .short
                .insert_unique(hash, (key, value), |(key, _)| hasher.hash_one(key));
        }
        for &(key, value) in &trace.long_entries {
            let hash = result.long_hasher.hash_one(key);
            let hasher = &result.long_hasher;
            result
                .long
                .insert_unique(hash, (key, value), |(key, _)| hasher.hash_one(key));
        }
        result
    }

    fn short_get(&self, key: &ShortKey) -> u16 {
        self.short
            .find(self.short_hasher.hash_one(key), |(candidate, _)| {
                candidate == key
            })
            .map_or(0, |(_, value)| *value)
    }

    fn long_get(&self, key: &u64) -> u16 {
        self.long
            .find(self.long_hasher.hash_one(key), |(candidate, _)| {
                candidate == key
            })
            .map_or(0, |(_, value)| *value)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn asm_hashbrown_long(
    tables: *const (),
    keys: *const u64,
    len: usize,
) -> u64 {
    let tables = unsafe { &*tables.cast::<RawTables>() };
    let keys = unsafe { std::slice::from_raw_parts(keys, len) };
    keys.iter().fold(0u64, |checksum, key| {
        checksum.wrapping_add(tables.long_get(key) as u64)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn asm_rust_group15_long(
    tables: *const (),
    keys: *const u64,
    len: usize,
) -> u64 {
    let tables = unsafe { &*tables.cast::<RustBoostTables>() };
    let keys = unsafe { std::slice::from_raw_parts(keys, len) };
    keys.iter().fold(0u64, |checksum, key| {
        checksum.wrapping_add(tables.long_get(key) as u64)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn asm_frozen_split15_long(
    tables: *const (),
    keys: *const u64,
    len: usize,
) -> u64 {
    let tables = unsafe { &*tables.cast::<FrozenSplitTables<15>>() };
    let keys = unsafe { std::slice::from_raw_parts(keys, len) };
    keys.iter().fold(0u64, |checksum, key| {
        checksum.wrapping_add(tables.long_get(key) as u64)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn asm_frozen_split16_long(
    tables: *const (),
    keys: *const u64,
    len: usize,
) -> u64 {
    let tables = unsafe { &*tables.cast::<FrozenSplitTables<16>>() };
    let keys = unsafe { std::slice::from_raw_parts(keys, len) };
    keys.iter().fold(0u64, |checksum, key| {
        checksum.wrapping_add(tables.long_get(key) as u64)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn asm_rust_group15_get4_long(
    tables: *const (),
    keys: *const u64,
    len: usize,
) -> u64 {
    let tables = unsafe { &*tables.cast::<RustBoostTables>() };
    let keys = unsafe { std::slice::from_raw_parts(keys, len) };
    let mut chunks = keys.chunks_exact(4);
    let mut checksum = chunks.by_ref().fold(0u64, |checksum, keys| {
        tables
            .long_get_many::<4>(keys)
            .into_iter()
            .fold(checksum, |sum, value| sum.wrapping_add(u64::from(value)))
    });
    for key in chunks.remainder() {
        checksum = checksum.wrapping_add(u64::from(tables.long_get(key)));
    }
    checksum
}

fn main() -> Result<(), String> {
    black_box(asm_hashbrown_long as unsafe extern "C" fn(*const (), *const u64, usize) -> u64);
    black_box(asm_rust_group15_long as unsafe extern "C" fn(*const (), *const u64, usize) -> u64);
    black_box(asm_frozen_split15_long as unsafe extern "C" fn(*const (), *const u64, usize) -> u64);
    black_box(asm_frozen_split16_long as unsafe extern "C" fn(*const (), *const u64, usize) -> u64);
    black_box(
        asm_rust_group15_get4_long as unsafe extern "C" fn(*const (), *const u64, usize) -> u64,
    );

    let path = std::env::args().nth(1).ok_or("usage: bench TRACE")?;
    let warmups: usize = std::env::var("HASH_WARMUPS")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .map_err(|_| "invalid HASH_WARMUPS")?;
    let iterations: usize = std::env::var("HASH_ITERATIONS")
        .unwrap_or_else(|_| "15".into())
        .parse()
        .map_err(|_| "invalid HASH_ITERATIONS")?;
    if iterations == 0 {
        return Err("HASH_ITERATIONS must be positive".into());
    }

    let trace = load_trace(&path)?;
    println!(
        "trace,short_entries={},long_entries={},short_probes={},long_probes={},total_probes={}",
        trace.short_entries.len(),
        trace.long_entries.len(),
        trace.short_probes.len(),
        trace.long_probes.len(),
        trace.short_probes.len() + trace.long_probes.len(),
    );

    let hashbrown = RawTables::new(&trace);
    let build_start = Instant::now();
    let group15 = RustBoostTables::new(&trace);
    let build_time = build_start.elapsed();
    let freeze15_start = Instant::now();
    let frozen15 = FrozenSplitTables::<15>::freeze(&group15);
    let freeze15_time = freeze15_start.elapsed();
    let freeze16_start = Instant::now();
    let frozen16 = FrozenSplitTables::<16>::freeze(&group15);
    let freeze16_time = freeze16_start.elapsed();
    let packed_long_start = Instant::now();
    let packed_long = FrozenPackedLongTable::freeze(&group15.long);
    let packed_long_time = packed_long_start.elapsed();
    let bloom4_start = Instant::now();
    let bloom4 = FrozenBloomTables::<4>::build(&group15, &trace);
    let bloom4_time = bloom4_start.elapsed();
    let bloom8_start = Instant::now();
    let bloom8 = FrozenBloomTables::<8>::build(&group15, &trace);
    let bloom8_time = bloom8_start.elapsed();
    let bloom16_start = Instant::now();
    let bloom16 = FrozenBloomTables::<16>::build(&group15, &trace);
    let bloom16_time = bloom16_start.elapsed();
    let bloom32_start = Instant::now();
    let bloom32 = FrozenBloomTables::<32>::build(&group15, &trace);
    let bloom32_time = bloom32_start.elapsed();
    let blocked8_start = Instant::now();
    let blocked8 = FrozenBlockedBloomTables::<8, 2>::build(&group15, &trace);
    let blocked8_time = blocked8_start.elapsed();
    let blocked16_start = Instant::now();
    let blocked16 = FrozenBlockedBloomTables::<16, 2>::build(&group15, &trace);
    let blocked16_time = blocked16_start.elapsed();
    let blocked3_start = Instant::now();
    let blocked3 = FrozenBlockedBloomTables::<16, 3>::build(&group15, &trace);
    let blocked3_time = blocked3_start.elapsed();
    let blocked32_start = Instant::now();
    let blocked32 = FrozenBlockedBloomTables::<32, 3>::build(&group15, &trace);
    let blocked32_time = blocked32_start.elapsed();
    let prefix_start = Instant::now();
    let prefix_bounds = FrozenPrefixBounds::<16>::build(&trace.short_entries);
    let prefix_time = prefix_start.elapsed();
    let prefix8_start = Instant::now();
    let prefix8_bounds = FrozenPrefixBounds::<8>::build(&trace.short_entries);
    let prefix8_time = prefix8_start.elapsed();
    let prefix_length_mask_start = Instant::now();
    let prefix_length_mask = FrozenPrefixLengthMask::build(&trace.short_entries);
    let prefix_length_mask_time = prefix_length_mask_start.elapsed();
    let approx_prefix_mask1 = FrozenApproxPrefixLengthMask::<1>::build(&trace.short_entries);
    let approx_prefix_mask2 = FrozenApproxPrefixLengthMask::<2>::build(&trace.short_entries);
    let approx_prefix_mask4 = FrozenApproxPrefixLengthMask::<4>::build(&trace.short_entries);
    let approx_prefix_mask8 = FrozenApproxPrefixLengthMask::<8>::build(&trace.short_entries);
    let approx_prefix_mask16 = FrozenApproxPrefixLengthMask::<16>::build(&trace.short_entries);
    let onpair_start = Instant::now();
    let onpair = FrozenOnPairTables::build(&group15, &trace);
    let onpair_time = onpair_start.elapsed();
    let home2_start = Instant::now();
    let home2 = FrozenHomeBloomTables::<1, 2>::build(&group15, &trace);
    let home2_time = home2_start.elapsed();
    let home3_start = Instant::now();
    let home3 = FrozenHomeBloomTables::<1, 3>::build(&group15, &trace);
    let home3_time = home3_start.elapsed();
    let home2x3_start = Instant::now();
    let home2x3 = FrozenHomeBloomTables::<2, 3>::build(&group15, &trace);
    let home2x3_time = home2x3_start.elapsed();
    let home4x3_start = Instant::now();
    let home4x3 = FrozenHomeBloomTables::<4, 3>::build(&group15, &trace);
    let home4x3_time = home4x3_start.elapsed();
    let xor8_start = Instant::now();
    let xor8 = match FrozenXor8Tables::build(&group15, &trace) {
        Some(table) => table,
        None => return Err("failed to construct frozen Xor8 filter".into()),
    };
    let xor8_time = xor8_start.elapsed();

    if trace.short_search_ends.is_some() {
        for (start, end) in short_probe_ranges(&trace) {
            let bound = prefix_bounds
                .probe::<0b111>(&trace.short_probes[start])
                .bound;
            if trace.short_probes[end - 1].length > bound {
                return Err("prefix filter produced a false-negative length bound".into());
            }
        }
    }

    for key in &trace.short_probes {
        let expected = hashbrown.short_get(key);
        if group15.short_get(key) != expected
            || frozen15.short_get(key) != expected
            || frozen16.short_get(key) != expected
            || bloom4.short_get(key) != expected
            || bloom8.short_get(key) != expected
            || bloom16.short_get(key) != expected
            || bloom32.short_get(key) != expected
            || blocked8.short_get(key) != expected
            || blocked16.short_get(key) != expected
            || blocked3.short_get(key) != expected
            || blocked32.short_get(key) != expected
            || onpair.short_get(key) != expected
            || home2.short_get(key) != expected
            || home3.short_get(key) != expected
            || home2x3.short_get(key) != expected
            || home4x3.short_get(key) != expected
            || xor8.short_get(key) != expected
        {
            return Err("short lookup mismatch in Group15 layout variants".into());
        }
    }
    for key in &trace.long_probes {
        let expected = hashbrown.long_get(key);
        if group15.long_get(key) != expected
            || frozen15.long_get(key) != expected
            || frozen16.long_get(key) != expected
            || packed_long.get(long_hash(*key), *key) != expected
            || bloom4.long_get(key) != expected
            || bloom8.long_get(key) != expected
            || bloom16.long_get(key) != expected
            || bloom32.long_get(key) != expected
            || blocked8.long_get(key) != expected
            || blocked16.long_get(key) != expected
            || blocked3.long_get(key) != expected
            || blocked32.long_get(key) != expected
            || onpair.long_get(key) != expected
            || home2.long_get(key) != expected
            || home3.long_get(key) != expected
            || home2x3.long_get(key) != expected
            || home4x3.long_get(key) != expected
            || xor8.long_get(key) != expected
        {
            return Err("long lookup mismatch in Group15 layout variants".into());
        }
    }

    if std::env::var_os("HASH_PROBE_STATS").is_some() {
        let mut short_groups = 0;
        let mut short_comparisons = 0;
        let mut short_hits = 0;
        let mut short_bloom16_positives = 0;
        let mut short_xor8_positives = 0;
        let mut short_blocked16_positives = 0;
        let mut short_blocked3_positives = 0;
        for &key in &trace.short_probes {
            let hash = short_hash(&key);
            let (group_count, comparison_count, hit) = group15.short.probe_stats(hash, key);
            short_groups += group_count;
            short_comparisons += comparison_count;
            short_hits += usize::from(hit);
            short_bloom16_positives += usize::from(bloom16.short_filter.may_contain(hash));
            short_xor8_positives += usize::from(xor8.short_filter.may_contain(hash));
            short_blocked16_positives += usize::from(blocked16.short_filter.may_contain(hash));
            short_blocked3_positives += usize::from(blocked3.short_filter.may_contain(hash));
        }
        let mut groups = 0;
        let mut comparisons = 0;
        let mut hits = 0;
        let mut long_bloom16_positives = 0;
        let mut long_xor8_positives = 0;
        let mut long_blocked16_positives = 0;
        let mut long_blocked3_positives = 0;
        for &key in &trace.long_probes {
            let hash = long_hash(key);
            let (group_count, comparison_count, hit) = group15.long.probe_stats(hash, key);
            groups += group_count;
            comparisons += comparison_count;
            hits += usize::from(hit);
            long_bloom16_positives += usize::from(bloom16.long_filter.may_contain(hash));
            long_xor8_positives += usize::from(xor8.long_filter.may_contain(hash));
            long_blocked16_positives += usize::from(blocked16.long_filter.may_contain(hash));
            long_blocked3_positives += usize::from(blocked3.long_filter.may_contain(hash));
        }
        println!(
            "short_probe_stats,probes={},groups={short_groups},comparisons={short_comparisons},hits={short_hits},bloom16_positives={short_bloom16_positives},blocked16_positives={short_blocked16_positives},blocked3_positives={short_blocked3_positives},xor8_positives={short_xor8_positives}",
            trace.short_probes.len()
        );
        println!(
            "long_probe_stats,probes={},groups={groups},comparisons={comparisons},hits={hits},bloom16_positives={long_bloom16_positives},blocked16_positives={long_blocked16_positives},blocked3_positives={long_blocked3_positives},xor8_positives={long_xor8_positives}",
            trace.long_probes.len()
        );
    }

    println!(
        "rust_group15_layout,short_groups={},long_groups={},short_slots={},long_slots={},correct=true",
        group15.short.groups.len(),
        group15.long.groups.len(),
        group15.short.entries.len(),
        group15.long.entries.len(),
    );
    let mutable_bytes = group15.short.groups.len() * std::mem::size_of::<BoostGroup>()
        + group15.long.groups.len() * std::mem::size_of::<BoostGroup>()
        + group15.short.entries.len() * std::mem::size_of::<BoostEntry<ShortKey>>()
        + group15.long.entries.len() * std::mem::size_of::<BoostEntry<u64>>();
    println!(
        "construction,mutable_ms={:.6},freeze15_ms={:.6},freeze16_ms={:.6},mutable_bytes={mutable_bytes},frozen15_bytes={},frozen16_bytes={}",
        build_time.as_secs_f64() * 1e3,
        freeze15_time.as_secs_f64() * 1e3,
        freeze16_time.as_secs_f64() * 1e3,
        frozen15.storage_bytes(),
        frozen16.storage_bytes(),
    );
    println!(
        "packed_long_construction,freeze_ms={:.6},bytes={}",
        packed_long_time.as_secs_f64() * 1e3,
        packed_long.storage_bytes(),
    );
    println!(
        "bloom_construction,bloom4_ms={:.6},bloom8_ms={:.6},bloom16_ms={:.6},bloom32_ms={:.6},bloom4_bytes={},bloom8_bytes={},bloom16_bytes={},bloom32_bytes={}",
        bloom4_time.as_secs_f64() * 1e3,
        bloom8_time.as_secs_f64() * 1e3,
        bloom16_time.as_secs_f64() * 1e3,
        bloom32_time.as_secs_f64() * 1e3,
        bloom4.storage_bytes(),
        bloom8.storage_bytes(),
        bloom16.storage_bytes(),
        bloom32.storage_bytes(),
    );
    println!(
        "xor8_construction,ms={:.6},bytes={}",
        xor8_time.as_secs_f64() * 1e3,
        xor8.storage_bytes(),
    );
    println!(
        "blocked_bloom_construction,blocked8_ms={:.6},blocked16_ms={:.6},blocked3_ms={:.6},blocked32_ms={:.6},prefix_ms={:.6},prefix8_ms={:.6},onpair_ms={:.6},home2_ms={:.6},home3_ms={:.6},home2x3_ms={:.6},home4x3_ms={:.6},blocked8_bytes={},blocked16_bytes={},blocked3_bytes={},blocked32_bytes={},prefix_bytes={},prefix8_bytes={},onpair_bytes={},home2_bytes={},home3_bytes={},home2x3_bytes={},home4x3_bytes={}",
        blocked8_time.as_secs_f64() * 1e3,
        blocked16_time.as_secs_f64() * 1e3,
        blocked3_time.as_secs_f64() * 1e3,
        blocked32_time.as_secs_f64() * 1e3,
        prefix_time.as_secs_f64() * 1e3,
        prefix8_time.as_secs_f64() * 1e3,
        onpair_time.as_secs_f64() * 1e3,
        home2_time.as_secs_f64() * 1e3,
        home3_time.as_secs_f64() * 1e3,
        home2x3_time.as_secs_f64() * 1e3,
        home4x3_time.as_secs_f64() * 1e3,
        blocked8.storage_bytes(),
        blocked16.storage_bytes(),
        blocked3.storage_bytes(),
        blocked32.storage_bytes(),
        prefix_bounds.storage_bytes(),
        prefix8_bounds.storage_bytes(),
        onpair.storage_bytes(),
        home2.storage_bytes(),
        home3.storage_bytes(),
        home2x3.storage_bytes(),
        home4x3.storage_bytes(),
    );
    println!(
        "prefix_length_mask_construction,ms={:.6},exact_bytes={},approx1_bytes={},approx2_bytes={},approx4_bytes={},approx8_bytes={},approx16_bytes={}",
        prefix_length_mask_time.as_secs_f64() * 1e3,
        prefix_length_mask.storage_bytes(),
        approx_prefix_mask1.storage_bytes(),
        approx_prefix_mask2.storage_bytes(),
        approx_prefix_mask4.storage_bytes(),
        approx_prefix_mask8.storage_bytes(),
        approx_prefix_mask16.storage_bytes(),
    );

    let final_only = std::env::var_os("HASH_FINAL_ONLY").is_some();
    report(
        "hashbrown-0.16-HashTable-foldhash",
        &trace,
        &hashbrown,
        &hashbrown,
        |tables, key| tables.short_get(key),
        |tables, key| tables.long_get(key),
        warmups,
        iterations,
    );
    report(
        "rust-group15-scalar",
        &trace,
        &group15,
        &group15,
        |tables, key| tables.short_get(key),
        |tables, key| tables.long_get(key),
        warmups,
        iterations,
    );
    if !final_only {
        report(
            "rust-frozen-split15",
            &trace,
            &frozen15,
            &frozen15,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-split16",
            &trace,
            &frozen16,
            &frozen16,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-packed-long",
            &trace,
            &group15.short,
            &packed_long,
            |table, key| table.get(short_hash(key), *key),
            |table, key| table.get(long_hash(*key), *key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-bloom4",
            &trace,
            &bloom4,
            &bloom4,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-bloom8",
            &trace,
            &bloom8,
            &bloom8,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-bloom16",
            &trace,
            &bloom16,
            &bloom16,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-blocked2-bloom8",
            &trace,
            &blocked8,
            &blocked8,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
    }
    report(
        "rust-frozen-blocked2-bloom16",
        &trace,
        &blocked16,
        &blocked16,
        |tables, key| tables.short_get(key),
        |tables, key| tables.long_get(key),
        warmups,
        iterations,
    );
    report(
        "rust-frozen-blocked3-bloom16",
        &trace,
        &blocked3,
        &blocked3,
        |tables, key| tables.short_get(key),
        |tables, key| tables.long_get(key),
        warmups,
        iterations,
    );
    report(
        "rust-frozen-blocked3-bloom32",
        &trace,
        &blocked32,
        &blocked32,
        |tables, key| tables.short_get(key),
        |tables, key| tables.long_get(key),
        warmups,
        iterations,
    );
    report(
        "rust-frozen-onpair-k3-short-k2-long",
        &trace,
        &onpair,
        &onpair,
        |tables, key| tables.short_get(key),
        |tables, key| tables.long_get(key),
        warmups,
        iterations,
    );
    if !final_only && trace.short_search_ends.is_some() {
        report_prefix_bounds::<0b000, 16, _, _>(
            "rust-grouped-control-group15",
            &trace,
            &prefix_bounds,
            &group15,
            &group15,
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_bounds::<0b110, 16, _, _>(
            "rust-frozen-prefix-86-group15",
            &trace,
            &prefix_bounds,
            &group15,
            &group15,
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_bounds::<0b101, 16, _, _>(
            "rust-frozen-prefix-84-group15",
            &trace,
            &prefix_bounds,
            &group15,
            &group15,
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_bounds::<0b011, 16, _, _>(
            "rust-frozen-prefix-64-group15",
            &trace,
            &prefix_bounds,
            &group15,
            &group15,
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_length_mask(
            "rust-frozen-prefix4-length-mask-group15",
            &trace,
            &prefix_length_mask,
            &group15,
            &group15,
            |prefixes, key| prefixes.get(key),
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_length_mask(
            "rust-frozen-approx-prefix4-mask1-group15",
            &trace,
            &approx_prefix_mask1,
            &group15,
            &group15,
            |prefixes, key| prefixes.get(key),
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_length_mask(
            "rust-frozen-approx-prefix4-mask2-group15",
            &trace,
            &approx_prefix_mask2,
            &group15,
            &group15,
            |prefixes, key| prefixes.get(key),
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_length_mask(
            "rust-frozen-approx-prefix4-mask4-group15",
            &trace,
            &approx_prefix_mask4,
            &group15,
            &group15,
            |prefixes, key| prefixes.get(key),
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_lazy_prefix_length_mask(
            "rust-frozen-lazy-prefix4-mask4-group15",
            &trace,
            &approx_prefix_mask4,
            &group15,
            &group15,
            |prefixes, key| prefixes.get(key),
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_length_mask(
            "rust-frozen-approx-prefix4-mask8-group15",
            &trace,
            &approx_prefix_mask8,
            &group15,
            &group15,
            |prefixes, key| prefixes.get(key),
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_length_mask(
            "rust-frozen-approx-prefix4-mask16-group15",
            &trace,
            &approx_prefix_mask16,
            &group15,
            &group15,
            |prefixes, key| prefixes.get(key),
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_gated_prefix_length_mask::<6, _, _, _>(
            "rust-frozen-prefix4-mask16-min6-group15",
            &trace,
            &approx_prefix_mask16,
            &group15,
            &group15,
            |prefixes, key| prefixes.get(key),
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_gated_prefix_length_mask::<8, _, _, _>(
            "rust-frozen-prefix4-mask16-min8-group15",
            &trace,
            &approx_prefix_mask16,
            &group15,
            &group15,
            |prefixes, key| prefixes.get(key),
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_bounds::<0b011, 8, _, _>(
            "rust-frozen-prefix-64-bpk8-group15",
            &trace,
            &prefix8_bounds,
            &group15,
            &group15,
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_bounds::<0b111, 16, _, _>(
            "rust-frozen-prefix-864-group15",
            &trace,
            &prefix_bounds,
            &group15,
            &group15,
            |tables, key, hash| {
                tables
                    .short
                    .get(hash.unwrap_or_else(|| short_hash(key)), *key)
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report_prefix_bounds::<0b111, 16, _, _>(
            "rust-frozen-prefix-864-blocked3",
            &trace,
            &prefix_bounds,
            &blocked3,
            &blocked3,
            |tables, key, hash| {
                let hash = hash.unwrap_or_else(|| short_hash(key));
                if tables.short_filter.may_contain(hash) {
                    tables.short_table.get(hash, *key)
                } else {
                    0
                }
            },
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
    } else if !final_only {
        println!("grouped_prefix_reports,skipped=true,reason=interleaved_trace");
    }
    if !final_only {
        report(
            "rust-frozen-home2-bloom",
            &trace,
            &home2,
            &home2,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-home3-bloom",
            &trace,
            &home3,
            &home3,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-home2x3-bloom",
            &trace,
            &home2x3,
            &home2x3,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-home4x3-bloom",
            &trace,
            &home4x3,
            &home4x3,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
    }
    if !final_only {
        report(
            "rust-frozen-xor8",
            &trace,
            &xor8,
            &xor8,
            |tables, key| tables.short_get(key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-long-bloom8",
            &trace,
            &group15.short,
            &bloom8,
            |table, key| table.get(short_hash(key), *key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-long-bloom16",
            &trace,
            &group15.short,
            &bloom16,
            |table, key| table.get(short_hash(key), *key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-long-bloom32",
            &trace,
            &group15.short,
            &bloom32,
            |table, key| table.get(short_hash(key), *key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        report(
            "rust-frozen-long-xor8",
            &trace,
            &group15.short,
            &xor8,
            |table, key| table.get(short_hash(key), *key),
            |tables, key| tables.long_get(key),
            warmups,
            iterations,
        );
        let packed_long_bloom16 = (&packed_long, &bloom16.long_filter);
        report(
            "rust-frozen-packed-long-bloom16",
            &trace,
            &group15.short,
            &packed_long_bloom16,
            |table, key| table.get(short_hash(key), *key),
            |(table, filter), key| {
                let hash = long_hash(*key);
                if filter.may_contain(hash) {
                    table.get(hash, *key)
                } else {
                    0
                }
            },
            warmups,
            iterations,
        );
        report_fused::<4>("rust-group15-get4", &trace, &group15, warmups, iterations);
    }
    Ok(())
}
