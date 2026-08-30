use std::fs;
use std::hash::{BuildHasher, Hash, Hasher};
use std::hint::black_box;
use std::mem::MaybeUninit;
use std::time::{Duration, Instant};

#[cfg(target_arch = "x86_64")]
use std::arch::asm;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _MM_HINT_T0, _mm_cmpeq_epi8, _mm_load_si128, _mm_movemask_epi8, _mm_prefetch, _mm_set1_epi32,
    _mm_setzero_si128,
};

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
    Ok(Trace {
        short_entries,
        long_entries,
        short_probes,
        long_probes,
    })
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn measure(warmups: usize, iterations: usize, mut lookup: impl FnMut() -> u64) -> Duration {
    for _ in 0..warmups {
        black_box(lookup());
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(lookup());
        samples.push(start.elapsed());
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
    let short_time = measure(warmups, iterations, || {
        trace.short_probes.iter().fold(0u64, |checksum, key| {
            checksum.wrapping_add(short_get(short_map, key) as u64)
        })
    });
    let long_time = measure(warmups, iterations, || {
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

fn report_fused<const N: usize>(
    name: &str,
    trace: &Trace,
    tables: &RustBoostTables,
    warmups: usize,
    iterations: usize,
) {
    let short_time = measure(warmups, iterations, || {
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
    let long_time = measure(warmups, iterations, || {
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
    let group15 = RustBoostTables::new(&trace);

    for key in &trace.short_probes {
        if group15.short_get(key) != hashbrown.short_get(key) {
            return Err("short lookup mismatch between hashbrown and Rust Group15".into());
        }
    }
    for key in &trace.long_probes {
        if group15.long_get(key) != hashbrown.long_get(key) {
            return Err("long lookup mismatch between hashbrown and Rust Group15".into());
        }
    }

    if std::env::var_os("HASH_PROBE_STATS").is_some() {
        let mut groups = 0;
        let mut comparisons = 0;
        let mut hits = 0;
        for &key in &trace.long_probes {
            let (group_count, comparison_count, hit) =
                group15.long.probe_stats(long_hash(key), key);
            groups += group_count;
            comparisons += comparison_count;
            hits += usize::from(hit);
        }
        println!(
            "long_probe_stats,probes={},groups={groups},comparisons={comparisons},hits={hits}",
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
    report_fused::<4>("rust-group15-get4", &trace, &group15, warmups, iterations);
    Ok(())
}
