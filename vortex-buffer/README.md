# Vortex Buffer

`Buffer<T>` and `BufferMut<T>` are zero-copy, aligned buffers of `T`. A `Buffer` is an immutable,
reference-counted view that is cheap to clone and slice. A `BufferMut` is uniquely owned and
growable, and `freeze` turns it into a `Buffer` without copying.

Buffers allocate through a pluggable `BufferAllocator`, or adopt memory they did not allocate:

- `Buffer::from(Vec<T>)` and `BufferMut::from(Vec<T>)` take over a `Vec`'s allocation, and
  `try_into_vec` hands it back without copying when nothing else shares it and it has not been
  sliced.
- `Buffer::from_owner` reads the values any owner exposes through `AsRef<[T]>`, and
  `BufferMut::from_owner` reads and writes those exposed through `AsMut<[T]>`, such as a writable
  memory map.
- `bytes::Bytes` and Arrow buffers convert in and out without copying.

A buffer declares the alignment of its start. Slicing lowers the declaration to whatever the new
start satisfies, and `aligned` raises it, copying only when the address falls short.
