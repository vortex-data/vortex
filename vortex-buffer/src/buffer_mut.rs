use std::any::Any;
use std::ops::Deref;

/// A [`BufferMut`] holds mutable bytes.
pub struct BufferMut {
    /// A ref-counted owner of the underlying bytes.
    pub(crate) owner: Box<dyn ByteOwnerMut>,

    /// A valid pointer into `owner`.
    ///
    /// We store a pointer instead of an offset to avoid pointer arithmetic
    /// which causes LLVM to fail to vectorise code correctly
    pub(crate) ptr: *mut u8,

    /// Byte length of the buffer.
    ///
    /// Must be less than or equal to `data.len()`
    pub(crate) length: usize,
}

/// A trait for types that can provide a mutable byte buffer interface.
pub unsafe trait ByteSourceMut {
    /// The type of the owner of the bytes.
    type OwnerMut: ByteOwnerMut;

    /// Returns the bytes as a mutable slice.
    fn as_mut_bytes(&mut self) -> &mut [u8];

    /// Returns the raw owner of the bytes.
    fn into_owner(self) -> Self::OwnerMut;
}

/// A trait for types that own mutable bytes.
pub trait ByteOwnerMut: Send + 'static {
    /// Downcasts the owner to `Any`.
    fn as_any(self: Box<Self>) -> Box<dyn Any + Send>;
}

impl<T: ByteSourceMut + Send + 'static> ByteOwnerMut for T {
    fn as_any(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }
}

impl BufferMut {
    /// Creates `BufferMut` from a [`ByteOwnerMut`] + [`ByteSourceMut`].
    pub fn from_owner(owner: impl ByteSourceMut + ByteOwnerMut) -> Self {
        let mut owner = Box::new(owner);
        let slice = owner.as_mut_bytes();
        let ptr = slice.as_mut_ptr();
        let length = slice.len();
        Self { owner, ptr, length }
    }

    /// Creates `BufferMut` from a `Box<ByteSourceMut + ByteOwnerMut>`.
    pub fn from_owner_box(mut owner: Box<impl ByteSourceMut + ByteOwnerMut>) -> Self {
        let slice = owner.as_mut_bytes();
        let ptr = slice.as_mut_ptr();
        let length = slice.len();
        Self { owner, ptr, length }
    }

    /// Returns the underlying slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.length) }
    }

    /// Returns the underlying mut slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.length) }
    }
}

impl Deref for BufferMut {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[u8]> for BufferMut {
    #[inline(always)]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for BufferMut {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}
