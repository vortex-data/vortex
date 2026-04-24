# I/O Subsystem

The `vortex-io` crate provides the I/O abstractions that underpin file reading, object store
access, and segment fetching. The system adapts its strategy -- coalescing, concurrency, and
prefetching -- to the characteristics of the underlying storage backend.

## VortexReadAt

The `VortexReadAt` trait is the unified interface for positional reads. It defines a single
async `read_at(offset, length, alignment)` method that returns an aligned buffer, along with
metadata methods that inform the I/O scheduler:

- **`concurrency()`** -- the maximum number of concurrent reads the backend can efficiently
  sustain. Local files default to 32; object stores default to 192.
- **`coalesce_config()`** -- optional parameters that control read merging (see below).
- **`size()`** -- the total size of the source, used for footer reads and bounds checking.

Implementations exist for local files (`FileReadAdapter`), object stores
(`ObjectStoreSource` via the `object_store` crate), and in-memory buffers.

## Read Coalescing

When many small reads target nearby offsets -- as is common when reading columnar segments --
the I/O system merges them into fewer, larger reads. The `CoalesceConfig` controls this
behaviour with two parameters:

- **`distance`** -- the maximum gap between two reads that will be merged. Reads separated by
  more than this are issued independently.
- **`max_size`** -- the maximum span of a single coalesced read.

Default configurations are tuned per backend. Local files use 8 KB for both distance and max
size, reflecting the low cost of small NVMe reads. Object stores use 1 MB distance and 16 MB
max size, reflecting the high per-request overhead of HTTP round-trips.

The coalescing algorithm runs inside an `IoRequestStream` that maintains a spatial index of
pending requests. When a request is polled, the stream scans for nearby requests within the
coalescing window and merges them into a single read. The merged buffer is then sliced to
fulfil each individual request at its original offset and alignment.

## Prefetching

Segment reads are issued through a `FileSegmentSource` that spawns a background task to drive
the I/O stream. Reads pass through four states:

1. **Registered** -- the read has been requested but its future has not yet been polled. It is
   eligible for coalescing with other nearby requests.
2. **Polled** -- the future has been awaited, signalling that the caller needs the data soon.
3. **In-flight** -- the coalesced read has been dispatched to the storage backend.
4. **Resolved** -- the data has arrived and the caller's future is completed.

Because the background task continuously drives the stream, reads that are registered but not
yet polled can be coalesced with reads that are already in flight. This provides implicit
prefetching: creating a segment read future is enough to start moving data toward the caller,
even before the caller awaits it.

Dropped futures notify the I/O stream so that cancelled requests are excluded from future
coalescing decisions.

## Memory Backpressure

A `SizeLimitedStream` wraps the I/O pipeline to prevent unbounded memory accumulation during
bulk reads. It uses a semaphore to track the total bytes of in-flight reads and blocks new
reads from being dispatched until completed reads free capacity. Permits are returned
automatically when a buffer is dropped, so cancellation and errors are handled correctly.

## Segment Cache

The `SegmentCache` trait provides a key-value interface for caching fetched segments by their
`SegmentId`. Three implementations are provided:

- **`NoOpSegmentCache`** -- no caching; used when segments are already in memory.
- **`MokaSegmentCache`** -- an in-memory LFU cache backed by the Moka library, sized by total
  byte capacity.
- **`InitialReadSegmentCache`** -- a two-level cache that captures segments read during the
  initial file footer parse and delegates misses to a fallback cache.

A `SharedSegmentSource` deduplicates concurrent requests for the same segment using weak
shared futures, ensuring that only one underlying I/O request is issued regardless of how many
callers request the same segment simultaneously.

## Backend Adaptation

Each `VortexReadAt` implementation provides its own concurrency and coalescing parameters,
allowing the I/O scheduler to adapt automatically:

| Backend       | Concurrency | Coalesce Distance | Coalesce Max Size |
|---------------|-------------|--------------------|--------------------|
| Local file    | 32          | 8 KB               | 8 KB               |
| Object store  | 192         | 1 MB               | 16 MB              |

Local file reads are dispatched via `spawn_blocking` to avoid blocking the async executor.
Object store reads are natively async and wrapped with `async_compat` for runtime
compatibility.
