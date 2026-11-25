// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod byte_length;
mod cast;
mod compare;
mod filter;
mod like;
mod slice;
mod take;
pub(crate) mod uncompressed_size_in_bytes;

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DEFAULT_CONFIG;
    use crate::OnPair;
    use crate::compress::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::utf8(VarBinArray::from_iter(
        [
            "https://www.example.com/page",
            "https://www.example.com/data",
            "https://www.test.org/page",
            "ftp://files.example.com/x",
            "https://www.example.com/page",
        ]
        .map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case::utf8_nullable(VarBinArray::from_iter(
        [Some("hello"), None, Some("world"), Some("test"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    #[case::binary(VarBinArray::from_iter(
        [b"alpha".as_slice(), b"beta", b"gamma", b"delta", b"epsilon"].map(Some),
        DType::Binary(Nullability::NonNullable),
    ))]
    #[case::binary_nullable(VarBinArray::from_iter(
        [Some(b"foo".as_slice()), None, Some(b"bar"), None, Some(b"baz")],
        DType::Binary(Nullability::Nullable),
    ))]
    #[case::single(VarBinArray::from_iter(
        ["single element"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case::empty_strings(VarBinArray::from_iter(
        ["", "test", "", "hello", ""].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case::large_nullable(VarBinArray::from_iter(
        (0..1500).map(|i| (i % 11 != 0).then(|| format!("https://www.example.com/items/{i:06}"))),
        DType::Utf8(Nullability::Nullable),
    ))]
    fn test_onpair_consistency_and_filter_conformance(
        #[case] varbin: VarBinArray,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = onpair_compress(&varbin.into_array(), DEFAULT_CONFIG, &mut ctx)?;
        assert!(encoded.is::<OnPair>());
        test_array_consistency(&encoded, &mut ctx);
        test_filter_conformance(&encoded, &mut ctx);
        if encoded.len() > 2 {
            let sliced = encoded
                .slice(1..encoded.len() - 1)?
                .optimize_ctx(&SESSION)?;
            assert!(sliced.is::<OnPair>());
            test_array_consistency(&sliced, &mut ctx);
            test_filter_conformance(&sliced, &mut ctx);
        }
        Ok(())
    }
}
