// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests of the file size estimate that [`Backpressure`](vortex_io::Backpressure) gives while a
//! file is written.

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;

    use crate::WriteOptionsSessionExt;
    use crate::WriteStrategyBuilder;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();

        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);

        session
    });

    const BATCH_ROWS: usize = 8192;
    const BATCHES: usize = 32;
    const CATEGORIES: [&str; 8] = [
        "electronics",
        "books",
        "garden",
        "toys",
        "groceries",
        "clothing",
        "sports",
        "automotive",
    ];

    /// A batch of a compressible table with integer, float and string columns.
    fn batch(index: usize) -> VortexResult<ArrayRef> {
        let mut state = (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let start = (index * BATCH_ROWS) as i64;
        let ids = PrimitiveArray::from_iter(start..start + BATCH_ROWS as i64);
        let quantities = PrimitiveArray::from_iter((0..BATCH_ROWS).map(|_| (next() % 100) as i32));
        let prices =
            PrimitiveArray::from_iter((0..BATCH_ROWS).map(|_| (next() % 10_000) as f64 / 100.0));
        let categories = VarBinViewArray::from_iter_str(
            (0..BATCH_ROWS).map(|_| CATEGORIES[(next() % CATEGORIES.len() as u64) as usize]),
        );

        Ok(StructArray::from_fields(&[
            ("id", ids.into_array()),
            ("quantity", quantities.into_array()),
            ("price", prices.into_array()),
            ("category", categories.into_array()),
        ])?
        .into_array())
    }

    fn dtype() -> VortexResult<DType> {
        Ok(batch(0)?.dtype().clone())
    }

    /// Write options with the default strategy, but with small data blocks, so that a small test
    /// table fills many blocks.
    fn write_options() -> crate::VortexWriteOptions {
        let strategy = WriteStrategyBuilder::from_session(&SESSION)
            .with_data_block_target_bytes(Some(64 << 10))
            .build();
        SESSION.write_options().with_strategy(strategy)
    }

    #[tokio::test]
    async fn estimate_is_writer_scoped() -> VortexResult<()> {
        let mut first_buf = ByteBufferMut::empty();
        let mut first = write_options().writer(&mut first_buf, dtype()?);
        let mut second_buf = ByteBufferMut::empty();
        let mut second = write_options().writer(&mut second_buf, dtype()?);

        for index in 0..4 {
            first.push(batch(index)?).await?;
        }

        assert!(first.estimated_size() > 0);
        assert_eq!(second.estimated_size(), 0);

        second.push(batch(0)?).await?;
        assert!(second.estimated_size() > 0);

        first.finish().await?;
        second.finish().await?;
        Ok(())
    }

    #[rstest]
    #[case::quarter(BATCHES / 4)]
    #[case::half(BATCHES / 2)]
    #[case::three_quarters(3 * BATCHES / 4)]
    #[tokio::test]
    async fn estimate_is_close_to_the_size_partway(#[case] batches: usize) -> VortexResult<()> {
        let mut buf = ByteBufferMut::empty();
        let mut writer = write_options().writer(&mut buf, dtype()?);
        for index in 0..batches {
            writer.push(batch(index)?).await?;
        }

        let estimate = writer.estimated_size() as f64;
        let size = writer.finish().await?.size() as f64;

        let error = (estimate - size).abs() / size;
        assert!(
            error <= 0.2,
            "estimate {estimate} is not within 20% of the size {size}"
        );
        Ok(())
    }

    /// A batch of a table with a sequence column and two dictionary columns, in the default
    /// batch size of the Iceberg writer.
    fn dictionary_batch(index: usize) -> VortexResult<ArrayRef> {
        const ROWS: usize = 2048;

        let start = index * ROWS;
        let rows = start..start + ROWS;
        let ids = PrimitiveArray::from_iter(rows.clone().map(|row| row as i64));
        let categories = PrimitiveArray::from_iter(rows.clone().map(|row| (row % 16) as i32));
        let data = VarBinArray::from_iter(
            rows.map(|row| Some(format!("value-{}", row % 1000))),
            DType::Utf8(Nullability::Nullable),
        );

        Ok(StructArray::from_fields(&[
            ("id", ids.into_array()),
            ("category", categories.into_array()),
            ("data", data.into_array()),
        ])?
        .into_array())
    }

    /// The dictionary columns hold codes before compression. The codes compress at a different
    /// ratio than the values and the other columns, so a single ratio for the write is far off.
    #[rstest]
    #[case::one_block(64)]
    #[case::two_blocks(128)]
    #[tokio::test]
    async fn estimate_is_close_for_dictionary_columns(#[case] batches: usize) -> VortexResult<()> {
        let mut buf = ByteBufferMut::empty();
        let mut writer = SESSION
            .write_options()
            .writer(&mut buf, dictionary_batch(0)?.dtype().clone());
        for index in 0..batches {
            writer.push(dictionary_batch(index)?).await?;
        }

        let estimate = writer.estimated_size() as f64;
        let size = writer.finish().await?.size() as f64;

        let error = (estimate - size).abs() / size;
        assert!(
            error <= 0.2,
            "estimate {estimate} is not within 20% of the size {size}"
        );
        Ok(())
    }

    /// Random floats and a constant string column.
    fn constant_column_batch(index: usize) -> VortexResult<ArrayRef> {
        let mut state = (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let padding = "x".repeat(100);
        let values = PrimitiveArray::from_iter(
            (0..BATCH_ROWS).map(|_| (next() >> 11) as f64 / (1u64 << 53) as f64),
        );
        let paddings = VarBinViewArray::from_iter_str((0..BATCH_ROWS).map(|_| padding.as_str()));

        Ok(StructArray::from_fields(&[
            ("value", values.into_array()),
            ("padding", paddings.into_array()),
        ])?
        .into_array())
    }

    #[rstest]
    #[case::before_the_first_block(8)]
    #[case::after_the_first_block(24)]
    #[tokio::test]
    async fn estimate_is_close_for_constant_columns(#[case] batches: usize) -> VortexResult<()> {
        let mut buf = ByteBufferMut::empty();
        let mut writer = SESSION
            .write_options()
            .writer(&mut buf, constant_column_batch(0)?.dtype().clone());
        for index in 0..batches {
            writer.push(constant_column_batch(index)?).await?;
        }

        let estimate = writer.estimated_size() as f64;
        let size = writer.finish().await?.size() as f64;

        let error = (estimate - size).abs() / size;
        assert!(
            error <= 0.2,
            "estimate {estimate} is not within 20% of the size {size}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn estimate_does_not_drop_sharply_and_ends_at_the_size() -> VortexResult<()> {
        let options = write_options();
        let estimator = options.backpressure();

        let mut buf = ByteBufferMut::empty();
        let mut writer = options.writer(&mut buf, dtype()?);
        let mut largest = 0u64;
        for index in 0..BATCHES {
            writer.push(batch(index)?).await?;

            // Before the write measures a ratio, it counts the uncompressed bytes in full. The
            // first measurement then lowers the estimate. Only check the samples after it.
            let measured = estimator.compression_ratio() < 1.0;
            let estimate = writer.estimated_size();
            if !measured {
                continue;
            }

            assert!(
                estimate as f64 >= 0.9 * largest as f64,
                "estimate {estimate} after batch {index} dropped from {largest}"
            );
            largest = largest.max(estimate);
        }

        let summary = writer.finish().await?;
        assert_eq!(estimator.buffered_bytes(), 0);
        assert_eq!(estimator.pending_bytes(), 0);
        assert_eq!(estimator.estimated_size(), summary.size());
        assert_eq!(summary.size(), buf.len() as u64);
        Ok(())
    }
}
