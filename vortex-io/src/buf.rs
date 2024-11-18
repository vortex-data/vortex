use std::io;

use bytes::Bytes;

use crate::VortexReadAt;

/// A stateful asynchronous reader that wraps an internal [stateless reader][VortexReadAt].
///
/// Read operations will advance the cursor.
#[derive(Clone)]
pub struct VortexBufReader<R> {
    inner: R,
    pos: u64,
}

impl<R> VortexBufReader<R> {
    /// Create a new buffered reader wrapping a stateless reader, with reads
    /// beginning at offset 0.
    pub fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    /// Set the position of the next `read_bytes` call directly.
    ///
    /// Note: this method will not fail if the position is past the end of the valid range,
    /// the failure will occur at read time and result in an [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof] error.
    pub fn set_position(&mut self, pos: u64) {
        self.pos = pos;
    }
}

impl<R: VortexReadAt> VortexBufReader<R> {
    /// Perform an exactly-sized read at the current cursor position, advancing
    /// the cursor and returning the bytes.
    ///
    /// If there are not enough bytes available to fulfill the request, an
    /// [`UnexpectedEof`][std::io::ErrorKind::UnexpectedEof] error is returned.
    ///
    /// See also [`VortexReadAt::read_byte_range`].
    pub async fn read_bytes(&mut self, len: u64) -> io::Result<Bytes> {
        let result = self.inner.read_byte_range(self.pos, len).await?;
        self.pos += len;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use bytes::Bytes;

    use crate::VortexBufReader;

    #[tokio::test]
    async fn test_buf_reader() {
        let reader = Bytes::from("0123456789".as_bytes());
        let mut buf_reader = VortexBufReader::new(reader);

        let first2 = buf_reader.read_bytes(2).await.unwrap();
        assert_eq!(first2.as_ref(), "01".as_bytes());

        buf_reader.set_position(8);
        let last2 = buf_reader.read_bytes(2).await.unwrap();
        assert_eq!(last2.as_ref(), "89".as_bytes());
    }

    #[tokio::test]
    async fn test_eof() {
        let reader = Bytes::from("0123456789".as_bytes());
        let mut buf_reader = VortexBufReader::new(reader);

        // Read past end of internal reader
        buf_reader.set_position(10);

        assert_eq!(
            buf_reader.read_bytes(1).await.unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof,
        );
    }
}
