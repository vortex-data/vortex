// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use super::*;

#[test]
fn test_empty_input() {
    let compressor = Compressor12::train(&[]);
    let compressed = compressor.compress(b"");
    assert!(compressed.is_empty());
}

#[test]
fn test_single_byte() {
    let samples: Vec<&[u8]> = vec![b"aaaa", b"aaaa", b"aaaa", b"aaaa", b"aaaa"];
    let compressor = Compressor12::train(&samples);
    let decompressor = compressor.decompressor();

    let input = b"aaaa";
    let compressed = compressor.compress(input);
    let decompressed = decompressor.decompress(&compressed);
    assert_eq!(&decompressed, input);
}

#[test]
fn test_roundtrip_simple() {
    let corpus: Vec<&[u8]> = vec![
        b"hello world",
        b"hello there",
        b"hello world again",
        b"world hello",
        b"hello world!",
    ];
    let compressor = Compressor12::train(&corpus);
    let decompressor = compressor.decompressor();

    for input in &corpus {
        let compressed = compressor.compress(input);
        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(&decompressed, *input, "roundtrip failed for {:?}", input);
    }
}

#[test]
fn test_roundtrip_unseen_data() {
    let corpus: Vec<&[u8]> = vec![
        b"hello world",
        b"hello world",
        b"hello world",
        b"hello world",
        b"hello world",
    ];
    let compressor = Compressor12::train(&corpus);
    let decompressor = compressor.decompressor();

    let input = b"xyz123!@#";
    let compressed = compressor.compress(input);
    let decompressed = decompressor.decompress(&compressed);
    assert_eq!(&decompressed, input);
}

#[test]
fn test_roundtrip_all_byte_values() {
    let compressor = Compressor12::train(&[b"test"]);
    let decompressor = compressor.decompressor();

    let input: Vec<u8> = (0..=255).collect();
    let compressed = compressor.compress(&input);
    let decompressed = decompressor.decompress(&compressed);
    assert_eq!(decompressed, input);
}

#[test]
fn test_roundtrip_urls() {
    let corpus: Vec<&[u8]> = vec![
        b"http://example.com/page?id=123",
        b"http://example.com/page?id=456",
        b"http://example.com/search?q=test",
        b"https://other.org/api/v1/data",
        b"http://example.com/page?id=789",
    ];
    let compressor = Compressor12::train(&corpus);
    let decompressor = compressor.decompressor();

    for input in &corpus {
        let compressed = compressor.compress(input);
        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(&decompressed, *input);
    }
}

#[test]
fn test_roundtrip_json() {
    let corpus: Vec<&[u8]> = vec![
        br#"{"name":"Alice","age":30,"city":"NYC"}"#,
        br#"{"name":"Bob","age":25,"city":"LA"}"#,
        br#"{"name":"Charlie","age":35,"city":"NYC"}"#,
        br#"{"name":"Diana","age":28,"city":"Chicago"}"#,
        br#"{"name":"Eve","age":32,"city":"NYC"}"#,
    ];
    let compressor = Compressor12::train(&corpus);
    let decompressor = compressor.decompressor();

    for input in &corpus {
        let compressed = compressor.compress(input);
        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(&decompressed, *input);
    }
}

#[test]
fn test_decompress_into() {
    let corpus: Vec<&[u8]> = vec![b"hello", b"hello", b"hello", b"hello", b"hello"];
    let compressor = Compressor12::train(&corpus);
    let decompressor = compressor.decompressor();

    let input = b"hello";
    let compressed = compressor.compress(input);

    let mut output = vec![0u8; 256];
    let len = decompressor.decompress_into(&compressed, &mut output);
    assert_eq!(&output[..len], input);
}

#[test]
fn test_symbol_concat() {
    let a = Symbol12::from_bytes(b"hel");
    let b = Symbol12::from_bytes(b"lo");
    let merged = a.concat(b).unwrap();
    assert_eq!(merged.len(), 5);

    let bytes = merged.value.to_le_bytes();
    assert_eq!(&bytes[..5], b"hello");
}

#[test]
fn test_symbol_concat_too_long() {
    let a = Symbol12::from_bytes(b"hello");
    let b = Symbol12::from_bytes(b"world");
    assert!(a.concat(b).is_none());
}

#[test]
fn test_rebuild_compressor() {
    let symbols = vec![Symbol12::from_bytes(b"he"), Symbol12::from_bytes(b"ll")];
    let compressor = Compressor12::rebuild(&symbols);
    let decompressor = compressor.decompressor();

    let input = b"hello";
    let compressed = compressor.compress(input);
    let decompressed = decompressor.decompress(&compressed);
    assert_eq!(&decompressed, input);
}

#[test]
fn test_no_escapes_needed() {
    let compressor = Compressor12::train(&[b"abc"]);
    let decompressor = compressor.decompressor();

    let input: Vec<u8> = (32..127).collect();
    let compressed = compressor.compress(&input);
    let decompressed = decompressor.decompress(&compressed);
    assert_eq!(decompressed, input);
}

#[test]
fn test_multi_byte_symbols_reduce_size() {
    // Repetitive data should benefit from multi-byte symbols with 12-bit packing
    let line = b"the quick brown fox jumps over the lazy dog ";
    let corpus: Vec<&[u8]> = (0..100).map(|_| line.as_ref()).collect();
    let compressor = Compressor12::train(&corpus);

    let compressed = compressor.compress(line);
    // With 12-bit packing, each code is 1.5 bytes.
    // Raw bytes would be 1.5 bytes each (worse), but multi-byte symbols
    // cover multiple bytes per 1.5-byte code, so total should be smaller.
    assert!(
        compressed.len() < line.len(),
        "compressed size ({}) should be less than input size ({})",
        compressed.len(),
        line.len(),
    );
}

#[test]
fn test_large_diverse_corpus() {
    let mut corpus: Vec<Vec<u8>> = Vec::new();
    for i in 0..200 {
        let s = format!("prefix_{i}_middle_{}_suffix_{}", i * 7 % 50, i * 13 % 30);
        corpus.push(s.into_bytes());
    }
    let refs: Vec<&[u8]> = corpus.iter().map(|v| v.as_slice()).collect();

    let compressor = Compressor12::train(&refs);

    assert!(
        compressor.symbols().len() > 5,
        "should learn multi-byte symbols, got {}",
        compressor.symbols().len()
    );

    let decompressor = compressor.decompressor();
    for input in &refs {
        let compressed = compressor.compress(input);
        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(&decompressed, *input);
    }
}

#[test]
fn test_12bit_packing_correctness() {
    // Verify the 12-bit packing produces correct byte counts
    let compressor = Compressor12::train(&[b"test"]);

    // Single byte input -> 1 code -> 2 bytes (trailing odd code)
    let compressed = compressor.compress(b"x");
    assert_eq!(compressed.len(), 2);

    // Two byte input -> 2 codes -> 3 bytes (one pair)
    let compressed = compressor.compress(b"xy");
    assert_eq!(compressed.len(), 3);

    let decompressor = compressor.decompressor();
    assert_eq!(decompressor.decompress(&compressor.compress(b"x")), b"x");
    assert_eq!(decompressor.decompress(&compressor.compress(b"xy")), b"xy");
    assert_eq!(
        decompressor.decompress(&compressor.compress(b"xyz")),
        b"xyz"
    );
}

#[test]
fn test_long_shared_substrings() {
    // Data with long shared substrings - FSST-12's sweet spot
    let mut corpus: Vec<Vec<u8>> = Vec::new();
    for i in 0..500 {
        let s = format!(
            "https://api.example.com/v2/users/{}/profile?format=json&lang=en&session=abc{}",
            i % 100,
            i % 50,
        );
        corpus.push(s.into_bytes());
    }
    let refs: Vec<&[u8]> = corpus.iter().map(|v| v.as_slice()).collect();
    let compressor = Compressor12::train(&refs);
    let decompressor = compressor.decompressor();

    let mut total_raw = 0;
    let mut total_compressed = 0;
    for input in &refs {
        let compressed = compressor.compress(input);
        let decompressed = decompressor.decompress(&compressed);
        assert_eq!(&decompressed, *input);
        total_raw += input.len();
        total_compressed += compressed.len();
    }

    // Should achieve meaningful compression
    assert!(
        total_compressed < total_raw,
        "should compress: {} < {}",
        total_compressed,
        total_raw
    );
}
