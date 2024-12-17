use std::any::Any;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter};
use std::ops;
use std::ops::Deref;
use std::slice::SliceIndex;
use std::sync::Arc;

use crate::buffer_mut::BufferMut;

/// A [`Buffer`] holds immutable bytes with zero-copy slicing and cloning bytes.
///
/// The implementation is loosely based on the `anybytes` crate and Arrow's buffer.
#[derive(Clone)]
pub struct Buffer {
    /// A ref-counted owner of the underlying bytes.
    pub(crate) owner: Arc<dyn ByteOwner>,

    /// A valid pointer into `owner`.
    ///
    /// We store a pointer instead of an offset to avoid pointer arithmetic
    /// which causes LLVM to fail to vectorise code correctly
    pub(crate) ptr: *const u8,

    /// Byte length of the buffer.
    ///
    /// Must be less than or equal to `data.len()`
    pub(crate) length: usize,
}

// ByteOwner is Send + Sync and Buffer is immutable.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Debug for Buffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("length", &self.length)
            .finish()
    }
}

/// A trait for types that can provide a byte buffer interface.
pub unsafe trait ByteSource {
    /// The type of the owner of the bytes.
    type Owner: ByteOwner;

    /// Returns the bytes as a slice.
    fn as_bytes(&self) -> &[u8];

    /// Returns the raw owner of the bytes.
    fn into_owner(self) -> Self::Owner;

    /// Attempts to convert this buffer into a mutable buffer.
    fn into_mut(self: Arc<Self>) -> Result<BufferMut, Arc<dyn ByteOwner>>;
}

/// A trait for types that own the bytes.
pub trait ByteOwner: Sync + Send + 'static {
    /// Downcasts the owner to `Any`.
    fn as_any(self: Arc<Self>) -> Arc<dyn Any + Sync + Send>;

    /// Attempts to convert this buffer into a mutable buffer.
    fn into_mut(self: Arc<Self>) -> Result<BufferMut, Arc<dyn ByteOwner>>;
}

impl<T: ByteSource + Sync + Send + 'static> ByteOwner for T {
    #[inline(always)]
    fn as_any(self: Arc<Self>) -> Arc<dyn Any + Sync + Send> {
        self
    }

    #[inline(always)]
    fn into_mut(self: Arc<Self>) -> Result<BufferMut, Arc<dyn ByteOwner>>
    where
        Self: Sized,
    {
        <T as ByteSource>::into_mut(self)
    }
}

impl Buffer {
    /// Creates a new empty `Buffer`.
    pub fn empty() -> Self {
        Self::from_owner(&[0u8; 0][..])
    }

    /// Creates `Buffer` from a [`ByteOwner`] + [`ByteSource`].
    pub fn from_owner(owner: impl ByteSource + ByteOwner) -> Self {
        let owner = Arc::new(owner);
        let slice = owner.as_bytes();
        let ptr = slice.as_ptr();
        let length = slice.len();
        Self { owner, ptr, length }
    }

    /// Creates `Buffer` from an `Arc<ByteSource + ByteOwner>`.
    pub fn from_owner_arc(arc: Arc<impl ByteSource + ByteOwner>) -> Self {
        let slice = arc.as_bytes();
        let ptr = slice.as_ptr();
        let length = slice.len();
        Self {
            owner: arc,
            ptr,
            length,
        }
    }

    /// Extract and downcast the owner of the bytes.
    /// TODO(ngates): does this API make it too easy to forget about sliced data?
    pub fn into_owner<T>(self) -> Option<Arc<T>>
    where
        T: Send + Sync + 'static,
    {
        self.owner.as_any().downcast::<T>().ok()
    }

    /// Attempts to convert this buffer into a mutable buffer if there is only a single
    /// reference to the buffer. Otherwise, returns the original buffer.
    pub fn try_into_mut(self) -> Result<BufferMut, Self> {
        self.owner
            .into_mut()
            .map(|owner_mut| BufferMut {
                owner: owner_mut.owner,
                ptr: self.ptr as *mut u8,
                length: self.length,
            })
            .map_err(|owner| Self {
                owner,
                ptr: self.ptr,
                length: self.length,
            })
    }

    /// Returns a slice of self for the provided range.
    /// This operation is `O(1)`.
    pub fn slice(&self, offset: usize) -> Self {
        let mut sliced = self.clone();

        assert!(
            offset <= sliced.length,
            "the offset of the new Buffer cannot exceed the existing length: offset={} length={}",
            offset,
            sliced.length
        );
        sliced.length -= offset;
        // SAFETY: `offset <= self.length`
        sliced.ptr = unsafe { sliced.ptr.add(offset) };
        sliced
    }

    /// Returns a new [Buffer] that is a slice of this buffer starting at `offset`, with
    /// `length` bytes.
    ///
    /// # Panics
    ///
    /// Panics iff `(offset + length)` is larger than the existing length.
    pub fn slice_with_length(&self, offset: usize, length: usize) -> Self {
        assert!(
            offset.saturating_add(length) <= self.length,
            "the offset of the new Buffer cannot exceed the existing length: slice offset={offset} length={length} selflen={}",
            self.length
        );
        // SAFETY: offset + length <= self.length
        let ptr = unsafe { self.ptr.add(offset) };
        Self {
            owner: self.owner.clone(),
            ptr,
            length,
        }
    }

    /// Returns the underlying slice of bytes.
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` is a valid pointer to `self.length` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.length) }
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref().eq(other.as_ref())
    }
}

impl Eq for Buffer {}

impl PartialOrd for Buffer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_ref().partial_cmp(other.as_ref())
    }
}

impl Deref for Buffer {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[u8]> for Buffer {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<I: SliceIndex<[u8]>> ops::Index<I> for Buffer {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        ops::Index::index(&**self, index)
    }
}
