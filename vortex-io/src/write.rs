// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::{Future, ready};
use std::io::{self, Cursor, Write};

use vortex_buffer::ByteBufferMut;

use crate::IoBuf;

pub trait VortexWrite {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>>;
    fn flush(&mut self) -> impl Future<Output = io::Result<()>>;
    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>>;
}

impl VortexWrite for Vec<u8> {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        self.extend_from_slice(buffer.as_slice());
        ready(Ok(buffer))
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Ok(()))
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Ok(()))
    }
}

impl VortexWrite for ByteBufferMut {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        self.extend_from_slice(buffer.as_slice());
        ready(Ok(buffer))
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Ok(()))
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Ok(()))
    }
}

impl<T> VortexWrite for Cursor<T>
where
    Cursor<T>: Write,
{
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        ready(Write::write_all(self, buffer.as_slice()).map(|_| buffer))
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Write::flush(self))
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Ok(()))
    }
}

impl<W: VortexWrite> VortexWrite for futures::io::Cursor<W> {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        self.set_position(self.position() + buffer.as_slice().len() as u64);
        VortexWrite::write_all(self.get_mut(), buffer)
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        VortexWrite::flush(self.get_mut())
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        VortexWrite::shutdown(self.get_mut())
    }
}

impl<W: VortexWrite> VortexWrite for &mut W {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        (*self).write_all(buffer)
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        (*self).flush()
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        (*self).shutdown()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;

    use super::*;

    #[rstest]
    #[case::single_write(vec![vec![1, 2, 3]], vec![1, 2, 3])]
    #[case::two_writes(vec![vec![1, 2], vec![3, 4]], vec![1, 2, 3, 4])]
    #[case::three_writes(vec![vec![1], vec![2], vec![3]], vec![1, 2, 3])]
    #[case::with_empty(vec![vec![1, 2], vec![], vec![3, 4]], vec![1, 2, 3, 4])]
    #[tokio::test]
    async fn test_vec_multiple_writes(#[case] writes: Vec<Vec<u8>>, #[case] expected: Vec<u8>) {
        let mut writer = Vec::new();

        for data in writes {
            VortexWrite::write_all(&mut writer, data).await.unwrap();
        }

        VortexWrite::flush(&mut writer).await.unwrap();
        VortexWrite::shutdown(&mut writer).await.unwrap();
        assert_eq!(writer, expected);
    }

    #[rstest]
    #[case::single_write(vec![vec![5, 6, 7]], vec![5, 6, 7])]
    #[case::two_writes(vec![vec![10], vec![20]], vec![10, 20])]
    #[case::multiple_small(vec![vec![1], vec![2], vec![3], vec![4]], vec![1, 2, 3, 4])]
    #[tokio::test]
    async fn test_byte_buffer_mut_operations(
        #[case] writes: Vec<Vec<u8>>,
        #[case] expected: Vec<u8>,
    ) {
        let mut buffer = ByteBufferMut::with_capacity(0);

        for data in writes {
            VortexWrite::write_all(&mut buffer, data).await.unwrap();
        }

        VortexWrite::flush(&mut buffer).await.unwrap();
        VortexWrite::shutdown(&mut buffer).await.unwrap();
        assert_eq!(buffer.as_ref(), &expected[..]);
    }

    #[rstest]
    #[case::empty(vec![], 0)]
    #[case::single_byte(vec![42], 1)]
    #[case::multiple_bytes(vec![1, 2, 3, 4, 5], 5)]
    #[case::large(vec![0; 1024], 1024)]
    #[tokio::test]
    async fn test_various_write_sizes(#[case] data: Vec<u8>, #[case] expected_len: usize) {
        let mut writer = Vec::new();
        VortexWrite::write_all(&mut writer, data.clone())
            .await
            .unwrap();
        assert_eq!(writer.len(), expected_len);
        assert_eq!(writer, data);
    }

    #[tokio::test]
    async fn test_cursor_operations() {
        let mut data = [0u8; 20];
        {
            let mut cursor = Cursor::new(&mut data[..]);

            // Write to cursor
            VortexWrite::write_all(&mut cursor, vec![1, 2, 3, 4, 5])
                .await
                .unwrap();
            assert_eq!(cursor.position(), 5);

            // Write more data
            VortexWrite::write_all(&mut cursor, vec![6, 7, 8, 9, 10])
                .await
                .unwrap();
            assert_eq!(cursor.position(), 10);

            // Test flush
            VortexWrite::flush(&mut cursor).await.unwrap();
        }

        // Check data after cursor is dropped
        assert_eq!(&data[..10], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    #[tokio::test]
    async fn test_futures_cursor() {
        let mut vec = Vec::new();
        {
            let mut cursor = futures::io::Cursor::new(&mut vec);

            // Test write operations
            VortexWrite::write_all(&mut cursor, vec![10, 20, 30])
                .await
                .unwrap();
            assert_eq!(cursor.position(), 3);

            VortexWrite::write_all(&mut cursor, vec![40, 50])
                .await
                .unwrap();
            assert_eq!(cursor.position(), 5);

            // Test flush and shutdown
            VortexWrite::flush(&mut cursor).await.unwrap();
            VortexWrite::shutdown(&mut cursor).await.unwrap();
        }

        assert_eq!(vec, vec![10, 20, 30, 40, 50]);
    }

    #[tokio::test]
    async fn test_mutable_reference() {
        let mut vec = Vec::new();
        {
            let mut writer_ref = &mut vec;

            // Test operations through mutable reference
            VortexWrite::write_all(&mut writer_ref, vec![100, 101, 102])
                .await
                .unwrap();

            VortexWrite::flush(&mut writer_ref).await.unwrap();
            VortexWrite::shutdown(&mut writer_ref).await.unwrap();
        }

        assert_eq!(vec, vec![100, 101, 102]);
    }

    #[tokio::test]
    async fn test_large_writes() {
        let mut writer = Vec::new();
        let large_data = vec![42u8; 100_000];

        VortexWrite::write_all(&mut writer, large_data.clone())
            .await
            .unwrap();
        assert_eq!(writer.len(), 100_000);
        assert!(writer.iter().all(|&b| b == 42));
    }

    #[tokio::test]
    async fn test_empty_writes() {
        let mut writer = Vec::new();
        let empty = vec![];

        VortexWrite::write_all(&mut writer, empty.clone())
            .await
            .unwrap();
        assert!(writer.is_empty());

        VortexWrite::write_all(&mut writer, vec![1, 2, 3])
            .await
            .unwrap();
        VortexWrite::write_all(&mut writer, empty).await.unwrap();
        assert_eq!(writer, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_sequential_accumulation() {
        let mut buffer = ByteBufferMut::with_capacity(0);

        for i in 0u8..5 {
            VortexWrite::write_all(&mut buffer, vec![i]).await.unwrap();
        }

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_ref(), &[0, 1, 2, 3, 4]);
    }

    #[rstest]
    #[case::vec_writer(0)]
    #[case::byte_buffer(1)]
    #[tokio::test]
    async fn test_writer_types(#[case] writer_type: usize) {
        let data = vec![10, 20, 30];

        match writer_type {
            0 => {
                let mut writer = Vec::new();
                VortexWrite::write_all(&mut writer, data.clone())
                    .await
                    .unwrap();
                assert_eq!(writer, data);
            }
            1 => {
                let mut writer = ByteBufferMut::with_capacity(0);
                VortexWrite::write_all(&mut writer, data.clone())
                    .await
                    .unwrap();
                assert_eq!(writer.as_ref(), &data[..]);
            }
            _ => unreachable!(),
        }
    }
}
