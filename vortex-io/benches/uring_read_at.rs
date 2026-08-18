// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-workload positional-read comparison. Run with `--help` for usage.

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("this benchmark is Linux-only");
}

#[cfg(target_os = "linux")]
mod bench {
    use std::env;
    use std::fs::File;
    use std::hint::black_box;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::FileExt;
    use std::path::Path;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::Receiver;
    use std::sync::mpsc::SyncSender;
    use std::sync::mpsc::TryRecvError;
    use std::sync::mpsc::sync_channel;
    use std::thread;
    use std::time::Duration;
    use std::time::Instant;

    use io_uring::IoUring;
    use io_uring::opcode;
    use io_uring::types;
    use parking_lot::Mutex;
    use rustix::fs::Advice;
    use rustix::fs::fadvise;
    use vortex_utils::aliases::hash_map::HashMap;

    pub fn main() -> io::Result<()> {
        let config = Config::parse()?;
        if config.help {
            help();
            return Ok(());
        }
        let file = Arc::new(File::open(&config.path)?);
        let file_len = file.metadata()?.len();
        let max_len = config.sizes.iter().copied().max().unwrap_or(0);
        if max_len == 0 || file_len < max_len as u64 {
            return Err(invalid(
                "input file is smaller than the largest non-zero read",
            ));
        }
        fadvise(&*file, 0, None, Advice::Random)?;
        prepare_cache(&file, file_len, config.cache)?;

        let device_before = config.device.as_deref().map(device_stats).transpose()?;
        let started = Instant::now();
        let mut result = run(Arc::clone(&file), file_len, &config)?;
        let elapsed = started.elapsed();
        let device_after = config.device.as_deref().map(device_stats).transpose()?;
        result.latencies.sort_unstable();
        let seconds = elapsed.as_secs_f64();

        println!(
            "mode={} engine_threads={} clients={} requests={} sizes={} cpu_ns={} cache={}",
            config.mode,
            config.engine_threads,
            config.clients,
            config.requests,
            config
                .sizes
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            config.cpu_ns,
            config.cache,
        );
        println!(
            "elapsed_s={seconds:.6} logical_reads={} kernel_read_ops={} submission_calls={} ops_per_submit={:.2} bytes={} throughput_mib_s={:.2} reads_s={:.0}",
            result.latencies.len(),
            result.kernel_ops,
            result.submissions,
            result.kernel_ops as f64 / result.submissions.max(1) as f64,
            result.bytes,
            result.bytes as f64 / 1_048_576.0 / seconds,
            result.latencies.len() as f64 / seconds,
        );
        println!(
            "latency_us_p50={:.1} latency_us_p95={:.1} latency_us_p99={:.1} latency_us_max={:.1} checksum={}",
            percentile(&result.latencies, 50).as_secs_f64() * 1e6,
            percentile(&result.latencies, 95).as_secs_f64() * 1e6,
            percentile(&result.latencies, 99).as_secs_f64() * 1e6,
            result
                .latencies
                .last()
                .copied()
                .unwrap_or_default()
                .as_secs_f64()
                * 1e6,
            result.checksum,
        );
        if let Some((before, after)) = device_before.zip(device_after) {
            println!(
                "device_read_ios={} device_read_mib={:.2} device_read_ms={} device_inflight_end={}",
                after.read_ios.saturating_sub(before.read_ios),
                after.sectors.saturating_sub(before.sectors) as f64 * 512.0 / 1_048_576.0,
                after.read_ms.saturating_sub(before.read_ms),
                after.inflight,
            );
        }
        Ok(())
    }

    fn run(file: Arc<File>, file_len: u64, config: &Config) -> io::Result<ResultRow> {
        let engine = match config.mode {
            Mode::Inline => None,
            Mode::Pool => Some(Arc::new(Engine::new(
                Arc::clone(&file),
                EngineKind::Pread,
                config.engine_threads,
                config.queue_depth,
            )?)),
            Mode::Uring => Some(Arc::new(Engine::new(
                Arc::clone(&file),
                EngineKind::Uring,
                config.engine_threads,
                config.queue_depth,
            )?)),
        };
        let next = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(config.clients + 1));
        let mut joins = Vec::with_capacity(config.clients);
        for client_id in 0..config.clients {
            let file = Arc::clone(&file);
            let engine = engine.as_ref().map(Arc::clone);
            let next = Arc::clone(&next);
            let barrier = Arc::clone(&barrier);
            let sizes = Arc::clone(&config.sizes);
            let request_count = config.requests;
            let cpu_ns = config.cpu_ns;
            joins.push(thread::spawn(move || -> io::Result<ClientRow> {
                let mut row = ClientRow::default();
                barrier.wait();
                loop {
                    let request_id = next.fetch_add(1, Ordering::Relaxed);
                    if request_id >= request_count {
                        break;
                    }
                    let len = sizes[request_id % sizes.len()];
                    let offset = (random_at(request_id as u64)
                        % ((file_len - len as u64) / 4096 + 1))
                        * 4096;
                    let started = Instant::now();
                    let buffer = match &engine {
                        Some(engine) => engine.read(offset, len, client_id)?,
                        None => {
                            let mut buffer = vec![0; len];
                            file.read_exact_at(&mut buffer, offset)?;
                            buffer
                        }
                    };
                    row.latencies.push(started.elapsed());
                    row.bytes += len as u64;
                    row.checksum = row.checksum.wrapping_add(sample(&buffer));
                    busy_cpu(cpu_ns, row.checksum);
                }
                Ok(row)
            }));
        }
        barrier.wait();
        let mut result = ResultRow::default();
        for join in joins {
            let row = join
                .join()
                .map_err(|_| io::Error::other("client panicked"))??;
            result.latencies.extend(row.latencies);
            result.bytes += row.bytes;
            result.checksum = result.checksum.wrapping_add(row.checksum);
        }
        result.kernel_ops = match engine {
            Some(engine) => {
                let stats = engine.shutdown()?;
                result.submissions = stats.submissions;
                stats.operations
            }
            None => {
                result.submissions = config.requests as u64;
                config.requests as u64
            }
        };
        Ok(result)
    }

    struct Engine {
        senders: Vec<SyncSender<Message>>,
        joins: Mutex<Vec<thread::JoinHandle<io::Result<WorkerStats>>>>,
    }

    #[derive(Clone, Copy)]
    enum EngineKind {
        Pread,
        Uring,
    }

    impl Engine {
        fn new(file: Arc<File>, kind: EngineKind, n: usize, depth: usize) -> io::Result<Self> {
            if n == 0 {
                return Err(invalid("engine-threads must be non-zero"));
            }
            let mut senders = Vec::with_capacity(n);
            let mut joins = Vec::with_capacity(n);
            for id in 0..n {
                let (tx, rx) = sync_channel(depth);
                let file = Arc::clone(&file);
                joins.push(
                    thread::Builder::new()
                        .name(format!("read-engine-{id}"))
                        .spawn(move || match kind {
                            EngineKind::Pread => pread_worker(file, rx),
                            EngineKind::Uring => uring_worker(file, rx, depth),
                        })?,
                );
                senders.push(tx);
            }
            Ok(Self {
                senders,
                joins: Mutex::new(joins),
            })
        }

        fn read(&self, offset: u64, len: usize, shard: usize) -> io::Result<Vec<u8>> {
            let (complete, receive) = sync_channel(1);
            self.senders[shard % self.senders.len()]
                .send(Message::Read(Request {
                    offset,
                    buffer: vec![0; len],
                    filled: 0,
                    complete,
                }))
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "engine stopped"))?;
            receive
                .recv()
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "completion dropped"))?
        }

        fn shutdown(&self) -> io::Result<WorkerStats> {
            for sender in &self.senders {
                sender
                    .send(Message::Stop)
                    .map_err(|_| io::Error::other("engine stopped"))?;
            }
            let mut stats = WorkerStats::default();
            for join in self.joins.lock().drain(..) {
                let worker = join
                    .join()
                    .map_err(|_| io::Error::other("engine panicked"))??;
                stats.operations += worker.operations;
                stats.submissions += worker.submissions;
            }
            Ok(stats)
        }
    }

    enum Message {
        Read(Request),
        Stop,
    }

    struct Request {
        offset: u64,
        buffer: Vec<u8>,
        filled: usize,
        complete: SyncSender<io::Result<Vec<u8>>>,
    }

    fn pread_worker(file: Arc<File>, rx: Receiver<Message>) -> io::Result<WorkerStats> {
        let mut operations = 0;
        while let Ok(message) = rx.recv() {
            match message {
                Message::Read(mut request) => {
                    let result = file
                        .read_exact_at(&mut request.buffer, request.offset)
                        .map(|()| request.buffer);
                    operations += 1;
                    drop(request.complete.send(result));
                }
                Message::Stop => break,
            }
        }
        Ok(WorkerStats {
            operations,
            submissions: operations,
        })
    }

    fn uring_worker(
        file: Arc<File>,
        rx: Receiver<Message>,
        depth: usize,
    ) -> io::Result<WorkerStats> {
        let entries = u32::try_from(depth.next_power_of_two()).map_err(io::Error::other)?;
        let mut ring: IoUring = IoUring::builder()
            .setup_single_issuer()
            .setup_defer_taskrun()
            .build(entries)?;
        let mut pending: HashMap<u64, Request> = HashMap::with_capacity(depth);
        let mut next_id = 1_u64;
        let mut operations = 0;
        let mut submissions = 0;
        let mut stopping = false;
        loop {
            let completions = ring
                .completion()
                .map(|cqe| (cqe.user_data(), cqe.result()))
                .collect::<Vec<_>>();
            for (user_data, completion_result) in completions {
                let Some(mut request) = pending.remove(&user_data) else {
                    return Err(io::Error::other("unknown completion"));
                };
                operations += 1;
                match completion_result {
                    result if result < 0 => drop(
                        request
                            .complete
                            .send(Err(io::Error::from_raw_os_error(-result))),
                    ),
                    0 => drop(request.complete.send(Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "io_uring read reached EOF",
                    )))),
                    result => {
                        request.filled += result as usize;
                        if request.filled == request.buffer.len() {
                            drop(request.complete.send(Ok(request.buffer)));
                        } else {
                            push(&mut ring, &file, request, &mut pending, &mut next_id)?;
                        }
                    }
                }
            }
            if stopping && pending.is_empty() {
                break;
            }
            let mut accepted = 0;
            while pending.len() < depth {
                let message = if pending.is_empty() && accepted == 0 {
                    rx.recv()
                        .map_err(|_| io::Error::other("request queue disconnected"))?
                } else {
                    match rx.try_recv() {
                        Ok(message) => message,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            stopping = true;
                            break;
                        }
                    }
                };
                match message {
                    Message::Read(request) => {
                        push(&mut ring, &file, request, &mut pending, &mut next_id)?;
                        accepted += 1;
                    }
                    Message::Stop => {
                        stopping = true;
                        break;
                    }
                }
            }
            if !pending.is_empty() {
                ring.submit_and_wait(1)?;
                submissions += 1;
            }
        }
        Ok(WorkerStats {
            operations,
            submissions,
        })
    }

    fn push(
        ring: &mut IoUring,
        file: &File,
        request: Request,
        pending: &mut HashMap<u64, Request>,
        next_id: &mut u64,
    ) -> io::Result<()> {
        let id = *next_id;
        *next_id = next_id.wrapping_add(1);
        let remaining = request.buffer.len() - request.filled;
        let len = u32::try_from(remaining.min(u32::MAX as usize)).map_err(io::Error::other)?;
        let pointer = unsafe { request.buffer.as_ptr().add(request.filled).cast_mut() };
        let entry = opcode::Read::new(types::Fd(file.as_raw_fd()), pointer, len)
            .offset(request.offset + request.filled as u64)
            .build()
            .user_data(id);
        // SAFETY: `pending` owns the stable allocation until this operation's CQE is reaped.
        unsafe {
            ring.submission()
                .push(&entry)
                .map_err(|_| io::Error::new(io::ErrorKind::WouldBlock, "SQ full"))?;
        }
        pending.insert(id, request);
        Ok(())
    }

    fn prepare_cache(file: &File, file_len: u64, mode: Cache) -> io::Result<()> {
        match mode {
            Cache::Keep => Ok(()),
            Cache::Cold => {
                file.sync_all()?;
                fadvise(file, 0, None, Advice::DontNeed)?;
                Ok(())
            }
            Cache::Warm => {
                let mut buffer = vec![0; 1024 * 1024];
                let mut offset = 0;
                while offset < file_len {
                    let len = usize::try_from((file_len - offset).min(buffer.len() as u64))
                        .map_err(io::Error::other)?;
                    file.read_exact_at(&mut buffer[..len], offset)?;
                    offset += len as u64;
                }
                black_box(sample(&buffer));
                Ok(())
            }
        }
    }

    fn busy_cpu(ns: u64, seed: u64) {
        if ns == 0 {
            return;
        }
        let start = Instant::now();
        let duration = Duration::from_nanos(ns);
        let mut value = seed;
        while start.elapsed() < duration {
            for _ in 0..64 {
                value = value.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17)
                    ^ 0xe703_7ed1_a0b4_28db;
            }
        }
        black_box(value);
    }

    fn sample(buffer: &[u8]) -> u64 {
        u64::from(buffer[0])
            ^ (u64::from(buffer[buffer.len() / 2]) << 8)
            ^ (u64::from(buffer[buffer.len() - 1]) << 16)
    }

    fn percentile(values: &[Duration], p: usize) -> Duration {
        values
            .get((values.len().saturating_sub(1)) * p / 100)
            .copied()
            .unwrap_or_default()
    }

    #[derive(Default)]
    struct ResultRow {
        latencies: Vec<Duration>,
        bytes: u64,
        checksum: u64,
        kernel_ops: u64,
        submissions: u64,
    }

    #[derive(Default)]
    struct WorkerStats {
        operations: u64,
        submissions: u64,
    }

    #[derive(Default)]
    struct ClientRow {
        latencies: Vec<Duration>,
        bytes: u64,
        checksum: u64,
    }

    fn random_at(index: u64) -> u64 {
        let mut z = index.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ z >> 30).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ z >> 27).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ z >> 31
    }

    #[derive(Clone, Copy)]
    enum Mode {
        Inline,
        Pool,
        Uring,
    }
    impl std::fmt::Display for Mode {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{}",
                match self {
                    Self::Inline => "pread-inline",
                    Self::Pool => "pread-pool",
                    Self::Uring => "uring",
                }
            )
        }
    }
    #[derive(Clone, Copy)]
    enum Cache {
        Keep,
        Cold,
        Warm,
    }
    impl std::fmt::Display for Cache {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{}",
                match self {
                    Self::Keep => "keep",
                    Self::Cold => "cold",
                    Self::Warm => "warm",
                }
            )
        }
    }

    struct Config {
        path: PathBuf,
        mode: Mode,
        clients: usize,
        engine_threads: usize,
        queue_depth: usize,
        requests: usize,
        sizes: Arc<[usize]>,
        cpu_ns: u64,
        cache: Cache,
        device: Option<String>,
        help: bool,
    }

    impl Config {
        fn parse() -> io::Result<Self> {
            let mut c = Self {
                path: PathBuf::new(),
                mode: Mode::Uring,
                clients: 32,
                engine_threads: 1,
                queue_depth: 256,
                requests: 10_000,
                sizes: Arc::from([64 * 1024]),
                cpu_ns: 0,
                cache: Cache::Keep,
                device: None,
                help: false,
            };
            let mut args = env::args().skip(1);
            while let Some(arg) = args.next() {
                let value = args.next();
                match arg.as_str() {
                    "--help" | "-h" => c.help = true,
                    "--path" => c.path = value.ok_or_else(|| invalid("missing path"))?.into(),
                    "--mode" => {
                        c.mode = match value.as_deref() {
                            Some("pread-inline") => Mode::Inline,
                            Some("pread-pool") => Mode::Pool,
                            Some("uring") => Mode::Uring,
                            _ => return Err(invalid("bad mode")),
                        }
                    }
                    "--clients" => c.clients = number(value, &arg)?,
                    "--engine-threads" => c.engine_threads = number(value, &arg)?,
                    "--queue-depth" => c.queue_depth = number(value, &arg)?,
                    "--requests" => c.requests = number(value, &arg)?,
                    "--cpu-ns" => c.cpu_ns = number(value, &arg)?,
                    "--sizes" => {
                        c.sizes = value
                            .ok_or_else(|| invalid("missing sizes"))?
                            .split(',')
                            .map(size)
                            .collect::<io::Result<Vec<_>>>()?
                            .into()
                    }
                    "--cache" => {
                        c.cache = match value.as_deref() {
                            Some("keep") => Cache::Keep,
                            Some("cold") => Cache::Cold,
                            Some("warm") => Cache::Warm,
                            _ => return Err(invalid("bad cache")),
                        }
                    }
                    "--device" => c.device = value,
                    _ => return Err(invalid(format!("unknown argument {arg}"))),
                }
            }
            if !c.help && c.path.as_os_str().is_empty() {
                return Err(invalid("--path is required"));
            }
            if c.clients == 0
                || c.engine_threads == 0
                || c.queue_depth == 0
                || c.requests == 0
                || c.sizes.is_empty()
            {
                return Err(invalid("counts and sizes must be non-zero"));
            }
            Ok(c)
        }
    }

    fn number<T: std::str::FromStr>(value: Option<String>, name: &str) -> io::Result<T> {
        value
            .ok_or_else(|| invalid(format!("missing {name}")))?
            .parse()
            .map_err(|_| invalid(format!("bad {name}")))
    }
    fn size(value: &str) -> io::Result<usize> {
        let (n, multiplier) = match value.as_bytes().last() {
            Some(b'K' | b'k') => (&value[..value.len() - 1], 1024),
            Some(b'M' | b'm') => (&value[..value.len() - 1], 1024 * 1024),
            _ => (value, 1),
        };
        n.parse::<usize>()
            .ok()
            .and_then(|n| n.checked_mul(multiplier))
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid(format!("bad size {value}")))
    }
    fn invalid(message: impl Into<String>) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput, message.into())
    }
    fn help() {
        println!(
            "usage: uring_read_at --path FILE [--mode pread-inline|pread-pool|uring] [--clients N] [--engine-threads N] [--queue-depth N] [--requests N] [--sizes 4K,64K,1M] [--cpu-ns N] [--cache keep|cold|warm] [--device nvme0n1]"
        );
    }

    #[derive(Default)]
    struct DeviceStats {
        read_ios: u64,
        sectors: u64,
        read_ms: u64,
        inflight: u64,
    }
    fn device_stats(device: &str) -> io::Result<DeviceStats> {
        let values =
            std::fs::read_to_string(Path::new("/sys/class/block").join(device).join("stat"))?
                .split_whitespace()
                .map(|v| v.parse::<u64>().map_err(io::Error::other))
                .collect::<io::Result<Vec<_>>>()?;
        if values.len() < 9 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "short block stat",
            ));
        }
        Ok(DeviceStats {
            read_ios: values[0],
            sectors: values[2],
            read_ms: values[3],
            inflight: values[8],
        })
    }
}

#[cfg(target_os = "linux")]
fn main() -> std::io::Result<()> {
    bench::main()
}
