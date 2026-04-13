// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_buffer::ALIGNMENT_TO_HOST_COPY;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_mask::MaskIter;
use vortex_utils::dyn_traits::DynEq;
use vortex_utils::dyn_traits::DynHash;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::Precision;

/// A handle to a buffer allocation.
///
/// There are two kinds of buffer allocations supported:
///
///   * **host** allocations, which were allocated by the global allocator and reside in main memory
///   * **device** allocations, which are remote to the CPU and live on a GPU or other external
///     device.
///
/// A device allocation can be copied to the host, yielding a new [`ByteBuffer`] containing the
/// copied data. Copying can fail at runtime, error recovery is system-dependent.
#[derive(Debug, Clone)]
pub struct BufferHandle(Inner);

#[derive(Debug, Clone)]
enum Inner {
    /// On the host/cpu.
    Host(ByteBuffer),
    /// On the device/gpu.
    Device(Arc<dyn DeviceBuffer>),
}

/// A buffer that is stored on the GPU.
pub trait DeviceBuffer: 'static + Send + Sync + Debug + DynEq + DynHash {
    /// Returns a reference as `Any` to enable downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the length of the buffer in bytes.
    fn len(&self) -> usize;

    /// Returns the alignment of the buffer.
    fn alignment(&self) -> Alignment;

    /// Returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Attempts to copy the device buffer to a host ByteBuffer.
    ///
    /// # Errors
    ///
    /// This operation may fail, depending on the device implementation and the underlying hardware.
    fn copy_to_host_sync(&self, alignment: Alignment) -> VortexResult<ByteBuffer>;

    /// Copies the device buffer to a host buffer asynchronously.
    ///
    /// Schedules an async copy and returns a future that completes when the copy is finished.
    ///
    /// # Arguments
    ///
    /// * `alignment` - The memory alignment to use for the host buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the async copy operation fails.
    fn copy_to_host(
        &self,
        alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>>;

    /// Create a new buffer that references a subrange of this buffer at the given
    /// slice indices.
    ///
    /// Note that slice indices are in byte units.
    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer>;

    /// Select and concatenate multiple byte ranges from this buffer into a new buffer.
    ///
    /// Unlike [`slice`](DeviceBuffer::slice), this method allocates new memory and copies the
    /// selected ranges into a contiguous buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the device cannot allocate memory or copy the data.
    fn copy_ranges(&self, ranges: &[Range<usize>]) -> VortexResult<Arc<dyn DeviceBuffer>>;

    /// Filter this buffer using a mask, where each element is `byte_width` bytes wide.
    ///
    /// Implementations can inspect the mask to decide whether to extract sparse ranges or
    /// read the entire buffer. The default implementation extracts contiguous slices from
    /// the mask and delegates to [`copy_ranges`](DeviceBuffer::copy_ranges).
    ///
    /// # Errors
    ///
    /// Returns an error if the device cannot allocate memory or copy the data.
    fn filter(&self, mask: &Mask, byte_width: usize) -> VortexResult<Arc<dyn DeviceBuffer>> {
        let slices = match mask.slices() {
            AllOr::Some(slices) => slices,
            AllOr::All => return Ok(self.slice(0..self.len())),
            AllOr::None => return self.copy_ranges(&[]),
        };
        let byte_ranges: Vec<Range<usize>> = slices
            .iter()
            .map(|&(s, e)| (s * byte_width)..(e * byte_width))
            .collect();
        self.copy_ranges(&byte_ranges)
    }

    /// Return a buffer with the given alignment. Where possible, this will be zero-copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer cannot be aligned (e.g., allocation or copy failure).
    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>>;
}

pub trait DeviceBufferExt: DeviceBuffer {
    /// Slice a range of elements `T` out of the device buffer.
    fn slice_typed<T: Sized>(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer>;
}

impl<B: DeviceBuffer> DeviceBufferExt for B {
    fn slice_typed<T: Sized>(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        let start_bytes = range.start * size_of::<T>();
        let end_bytes = range.end * size_of::<T>();
        self.slice(start_bytes..end_bytes)
    }
}

impl Hash for dyn DeviceBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

impl PartialEq for dyn DeviceBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}
impl Eq for dyn DeviceBuffer {}

impl BufferHandle {
    /// Create a new handle to a host [`ByteBuffer`].
    pub fn new_host(byte_buffer: ByteBuffer) -> Self {
        BufferHandle(Inner::Host(byte_buffer))
    }

    /// Create a new handle to a memory allocation that exists on an external device.
    ///
    /// Allocations on external devices are not cheaply accessible from the CPU and most be copied
    /// into new memory when we read them.
    pub fn new_device(device: Arc<dyn DeviceBuffer>) -> Self {
        BufferHandle(Inner::Device(device))
    }
}

impl BufferHandle {
    /// Returns `true` if this buffer resides on the device (GPU).
    pub fn is_on_device(&self) -> bool {
        matches!(&self.0, Inner::Device(_))
    }

    /// Returns `true` if this buffer resides on the host (CPU).
    pub fn is_on_host(&self) -> bool {
        matches!(&self.0, Inner::Host(_))
    }

    /// Gets the size of the buffer, in bytes.
    pub fn len(&self) -> usize {
        match &self.0 {
            Inner::Host(bytes) => bytes.len(),
            Inner::Device(device) => device.len(),
        }
    }

    /// Returns the alignment of the buffer.
    pub fn alignment(&self) -> Alignment {
        match &self.0 {
            Inner::Host(bytes) => bytes.alignment(),
            Inner::Device(device) => device.alignment(),
        }
    }

    /// Returns true if the buffer is aligned to the given alignment.
    pub fn is_aligned_to(&self, alignment: Alignment) -> bool {
        self.alignment().is_aligned_to(alignment)
    }

    /// Ensure the buffer satisfies the requested alignment.
    ///
    /// Both host and device buffers will be copied if necessary to satisfy the alignment.
    pub fn ensure_aligned(self, alignment: Alignment) -> VortexResult<Self> {
        match self.0 {
            Inner::Host(buffer) => Ok(BufferHandle::new_host(buffer.aligned(alignment))),
            Inner::Device(device) => Ok(BufferHandle::new_device(device.aligned(alignment)?)),
        }
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a new handle to a subrange of memory at the given bind range.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// # use vortex_array::buffer::BufferHandle;
    /// # use vortex_buffer::buffer;
    /// let handle1 = BufferHandle::new_host(buffer![1u8,2,3,4]);
    /// let handle2 = handle1.slice(1..4);
    /// assert_eq!(handle2.unwrap_host(), buffer![2u8,3,4]);
    /// ```
    pub fn slice(&self, range: Range<usize>) -> Self {
        match &self.0 {
            Inner::Host(host) => BufferHandle::new_host(host.slice(range)),
            Inner::Device(device) => BufferHandle::new_device(device.slice(range)),
        }
    }

    /// Select and concatenate multiple byte ranges from this buffer into a new buffer.
    ///
    /// Unlike [`slice`](BufferHandle::slice), this method allocates a new buffer and copies
    /// the selected ranges.
    ///
    /// # Example
    ///
    /// ```
    /// # use vortex_array::buffer::BufferHandle;
    /// # use vortex_buffer::buffer;
    /// let handle = BufferHandle::new_host(buffer![1u8, 2, 3, 4, 5, 6]);
    /// let filtered = handle.copy_ranges(&[0..2, 4..6]).unwrap();
    /// assert_eq!(filtered.unwrap_host(), buffer![1u8, 2, 5, 6]);
    /// ```
    pub fn copy_ranges(&self, ranges: &[Range<usize>]) -> VortexResult<Self> {
        match &self.0 {
            Inner::Host(host) => {
                let total_len: usize = ranges.iter().map(|r| r.len()).sum();
                let mut result = ByteBufferMut::with_capacity_aligned(total_len, host.alignment());
                for range in ranges {
                    result.extend_from_slice(&host.as_slice()[range.start..range.end]);
                }
                Ok(BufferHandle::new_host(result.freeze()))
            }
            Inner::Device(device) => Ok(BufferHandle::new_device(device.copy_ranges(ranges)?)),
        }
    }

    /// Filter this buffer using a mask, where each element is `byte_width` bytes wide.
    ///
    /// For device buffers, the mask is passed through to the device so it can decide
    /// whether to extract sparse ranges or read the entire buffer.
    ///
    /// # Example
    ///
    /// ```
    /// # use vortex_array::buffer::BufferHandle;
    /// # use vortex_buffer::{buffer, Buffer};
    /// # use vortex_mask::Mask;
    /// let values = buffer![1u32, 2u32, 3u32, 4u32, 5u32, 6u32];
    /// let handle = BufferHandle::new_host(values.into_byte_buffer());
    /// let mask = Mask::from_slices(6, vec![(0, 2), (4, 6)]);
    /// let filtered = handle.filter(&mask, size_of::<u32>()).unwrap();
    /// let result = Buffer::<u32>::from_byte_buffer(filtered.to_host_sync());
    /// assert_eq!(result, buffer![1, 2, 5, 6]);
    /// ```
    pub fn filter(&self, mask: &Mask, byte_width: usize) -> VortexResult<Self> {
        match &self.0 {
            Inner::Host(host) => {
                match mask.threshold_iter(FILTER_SELECTIVITY_THRESHOLD) {
                    AllOr::All => Ok(self.clone()),
                    AllOr::None => self.copy_ranges(&[]),
                    AllOr::Some(MaskIter::Slices(slices)) => {
                        Ok(BufferHandle::new_host(filter_bytes_by_slices(
                            host.as_slice(),
                            slices,
                            byte_width,
                            host.alignment(),
                        )))
                    }
                    AllOr::Some(MaskIter::Indices(indices)) => {
                        Ok(BufferHandle::new_host(filter_bytes_by_indices(
                            host.as_slice(),
                            indices,
                            byte_width,
                            host.alignment(),
                        )))
                    }
                }
            }
            Inner::Device(device) => Ok(BufferHandle::new_device(device.filter(mask, byte_width)?)),
        }
    }

    /// Reinterpret the pointee as a buffer of `T` and slice the provided element range.
    ///
    /// # Example
    ///
    /// ```
    /// # use vortex_array::buffer::BufferHandle;
    /// # use vortex_buffer::{buffer, Buffer};
    /// let values = buffer![1u32, 2u32, 3u32, 4u32];
    /// let handle = BufferHandle::new_host(values.into_byte_buffer());
    /// let sliced = handle.slice_typed::<u32>(1..4);
    /// let result = Buffer::<u32>::from_byte_buffer(sliced.to_host_sync());
    /// assert_eq!(result, buffer![2, 3, 4]);
    /// ```
    pub fn slice_typed<T: Sized>(&self, range: Range<usize>) -> Self {
        let start = range.start * size_of::<T>();
        let end = range.end * size_of::<T>();

        self.slice(start..end)
    }

    #[allow(clippy::panic)]
    /// Unwraps the handle as host memory.
    ///
    /// # Panics
    ///
    /// This will panic if the handle points to device memory.
    pub fn unwrap_host(self) -> ByteBuffer {
        match self.0 {
            Inner::Host(b) => b,
            Inner::Device(_) => panic!("unwrap_host called for Device allocation"),
        }
    }

    #[allow(clippy::panic)]
    /// Unwraps the handle as device memory.
    ///
    /// # Panics
    ///
    /// This will panic if the handle points to host memory.
    pub fn unwrap_device(self) -> Arc<dyn DeviceBuffer> {
        match self.0 {
            Inner::Device(b) => b,
            Inner::Host(_) => panic!("unwrap_device called for Host allocation"),
        }
    }

    /// Downcast this handle as a handle to a host-resident buffer, or `None`.
    pub fn as_host_opt(&self) -> Option<&ByteBuffer> {
        match &self.0 {
            Inner::Host(buffer) => Some(buffer),
            Inner::Device(_) => None,
        }
    }

    /// Downcast this handle as a handle to a device buffer, or `None`.
    pub fn as_device_opt(&self) -> Option<&Arc<dyn DeviceBuffer>> {
        match &self.0 {
            Inner::Host(_) => None,
            Inner::Device(device) => Some(device),
        }
    }

    /// A version of [`as_host_opt`][Self::as_host_opt] that panics if the allocation is
    /// not a host allocation.
    pub fn as_host(&self) -> &ByteBuffer {
        self.as_host_opt().vortex_expect("expected host buffer")
    }

    /// A version of [`as_device_opt`][Self::as_device_opt] that panics if the allocation is
    /// not a device allocation.
    pub fn as_device(&self) -> &Arc<dyn DeviceBuffer> {
        self.as_device_opt().vortex_expect("expected device buffer")
    }

    /// Returns a host-resident copy of the data in the buffer.
    ///
    /// If the data was already host-resident, this is trivial.
    ///
    /// If the data was device-resident, data will be copied from the device to a new allocation
    /// on the host.
    ///
    /// # Panics
    ///
    /// This function will never panic if the data is already host-resident.
    ///
    /// For a device-resident handle, any errors triggered by the copying from device to host will
    /// result in a panic.
    ///
    /// See also: [`try_to_host`][Self::try_to_host].
    pub fn to_host_sync(&self) -> ByteBuffer {
        self.try_to_host_sync()
            .vortex_expect("to_host: copy from device to host failed")
    }

    /// Returns a host-resident copy of the data behind the handle, consuming the handle.
    ///
    /// If the data was already host-resident, this completes trivially.
    ///
    /// See also [`to_host`][Self::to_host].
    ///
    /// # Panics
    ///
    /// See the panic documentation on [`to_host`][Self::to_host].
    pub fn into_host_sync(self) -> ByteBuffer {
        self.try_into_host_sync()
            .vortex_expect("into_host: copy from device to host failed")
    }

    /// Attempts to load this buffer into a host-resident allocation.
    ///
    /// If the allocation is already host-resident, this trivially completes with success.
    ///
    /// If it is a device allocation, then this issues an operation that attempts to copy the data
    /// from the device into a host-resident buffer, and returns a handle to that buffer.
    pub fn try_to_host_sync(&self) -> VortexResult<ByteBuffer> {
        match &self.0 {
            Inner::Host(b) => Ok(b.clone()),
            Inner::Device(device) => device.copy_to_host_sync(ALIGNMENT_TO_HOST_COPY),
        }
    }

    /// Attempts to load this buffer into a host-resident allocation, consuming the handle.
    ///
    /// See also [`try_to_host`][Self::try_to_host].
    pub fn try_into_host_sync(self) -> VortexResult<ByteBuffer> {
        match self.0 {
            Inner::Host(b) => Ok(b),
            Inner::Device(device) => device.copy_to_host_sync(ALIGNMENT_TO_HOST_COPY),
        }
    }

    /// Asynchronously copies the buffer to the host.
    ///
    /// This is a no-op if the buffer is already on the host.
    ///
    /// # Returns
    ///
    /// A future that resolves to the host buffer when the copy completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the async copy operation fails.
    pub fn try_to_host(&self) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        match &self.0 {
            Inner::Host(b) => {
                let buffer = b.clone();
                Ok(Box::pin(async move { Ok(buffer) }))
            }
            Inner::Device(device) => device.copy_to_host(ALIGNMENT_TO_HOST_COPY),
        }
    }

    /// Asynchronously copies the buffer to the host, consuming the handle.
    ///
    /// This is a no-op if the buffer is already on the host.
    ///
    /// # Returns
    ///
    /// A future that resolves to the host buffer when the copy completes.
    ///
    /// # Errors
    ///
    /// Returns an error if the async copy operation fails.
    pub fn try_into_host(self) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        match self.0 {
            Inner::Host(b) => Ok(Box::pin(async move { Ok(b) })),
            Inner::Device(device) => device.copy_to_host(ALIGNMENT_TO_HOST_COPY),
        }
    }

    /// Asynchronously copies the buffer to the host.
    ///
    /// # Panics
    ///
    /// Any errors triggered by the copying from device to host will result in a panic.
    pub fn to_host(&self) -> BoxFuture<'static, ByteBuffer> {
        let future = self
            .try_to_host()
            .vortex_expect("to_host: failed to initiate copy from device to host");
        Box::pin(async move {
            future
                .await
                .vortex_expect("to_host: copy from device to host failed")
        })
    }

    /// Asynchronously copies the buffer to the host, consuming the handle.
    ///
    /// # Panics
    ///
    /// Any errors triggered by the copying from device to host will result in a panic.
    pub fn into_host(self) -> BoxFuture<'static, ByteBuffer> {
        let future = self
            .try_into_host()
            .vortex_expect("into_host: failed to initiate copy from device to host");
        Box::pin(async move {
            future
                .await
                .vortex_expect("into_host: copy from device to host failed")
        })
    }
}

/// Selectivity threshold for dispatching between the indices and slices filter paths.
///
/// Mirrors the constant used in `filter::execute::slice`.  When mask density is above this
/// threshold we copy contiguous runs (slices); below it we copy individual elements (indices).
const FILTER_SELECTIVITY_THRESHOLD: f64 = 0.8;

/// Filter a byte buffer using a set of element-level `(start, end)` ranges.
fn filter_bytes_by_slices(
    src: &[u8],
    slices: &[(usize, usize)],
    byte_width: usize,
    alignment: Alignment,
) -> ByteBuffer {
    let total: usize = slices.iter().map(|(s, e)| (e - s) * byte_width).sum();
    let mut out = ByteBufferMut::with_capacity_aligned(total, alignment);
    for &(start, end) in slices {
        out.extend_from_slice(&src[start * byte_width..end * byte_width]);
    }
    out.freeze()
}

/// Filter a byte buffer by copying `byte_width` bytes for each selected index.
fn filter_bytes_by_indices(
    src: &[u8],
    indices: &[usize],
    byte_width: usize,
    alignment: Alignment,
) -> ByteBuffer {
    let total = indices.len() * byte_width;
    let mut out = ByteBufferMut::with_capacity_aligned(total, alignment);
    for &idx in indices {
        out.extend_from_slice(&src[idx * byte_width..(idx + 1) * byte_width]);
    }
    out.freeze()
}

impl ArrayHash for BufferHandle {
    // TODO(aduffy): implement for array hash
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        match &self.0 {
            Inner::Host(host) => host.array_hash(state, precision),
            Inner::Device(dev) => match precision {
                Precision::Ptr => {
                    Arc::as_ptr(dev).hash(state);
                }
                Precision::Value => {
                    dev.hash(state);
                }
            },
        }
    }
}

impl ArrayEq for BufferHandle {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        match (&self.0, &other.0) {
            (Inner::Host(b), Inner::Host(b2)) => b.array_eq(b2, precision),
            (Inner::Device(b), Inner::Device(b2)) => match precision {
                Precision::Ptr => Arc::ptr_eq(b, b2),
                Precision::Value => b.eq(b2),
            },
            _ => false,
        }
    }
}
