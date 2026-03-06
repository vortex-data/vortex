// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for Vortex.

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::ByteBufferMut;
    use vortex::error::VortexResult;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::session::VortexSession;

    /// Test that compression produces deterministic results.
    #[tokio::test]
    async fn test_compression_determinism() -> VortexResult<()> {
        // Create a deterministic array with linear-with-noise pattern
        let values: Vec<i64> = (0i64..100_000).map(|i| i + (i * 7 + 3) % 11).collect();

        let array =
            PrimitiveArray::new(Buffer::from_iter(values), Validity::NonNullable).into_array();

        // Write concurrently and verify all sizes match expected
        let futures: Vec<_> = (0..5)
            .map(|_| {
                let array = array.clone();
                async move {
                    let session = VortexSession::default();
                    let mut buf = ByteBufferMut::empty();
                    session
                        .write_options()
                        .write(&mut buf, array.to_array_stream())
                        .await?;
                    VortexResult::Ok(buf.len())
                }
            })
            .collect();

        #[cfg(feature = "unstable_encodings")]
        const EXPECTED_SIZE: usize = 216004;
        #[cfg(not(feature = "unstable_encodings"))]
        const EXPECTED_SIZE: usize = 215972;

        let sizes = futures::future::try_join_all(futures).await?;
        for (i, size) in sizes.iter().enumerate() {
            assert_eq!(
                *size, EXPECTED_SIZE,
                "Run {i} compressed the array to {size} bytes instead of the expected {EXPECTED_SIZE}."
            );
        }

        Ok(())
    }
}
