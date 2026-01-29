// Test to verify the order of elements through transpose -> delta -> bitpack pipeline

#![allow(clippy::use_debug)]

use fastlanes::{BitPacking, Delta, Transpose};

/// Test the complete pipeline and print indices at each step
#[test]
fn test_fastlanes_order() {
    // Use u16 for easier visualization (64 lanes, 16 rows)
    const LANES: usize = 64;

    // Create linear input: value = index (so we can track where each value came from)
    let linear: [u16; 1024] = std::array::from_fn(|i| i as u16);

    println!("\n=== Step 0: Linear Input ===");
    println!("First 16 values: {:?}", &linear[0..16]);
    println!(
        "Values at 0, 64, 128, 192: {:?}",
        [linear[0], linear[64], linear[128], linear[192]]
    );

    // Step 1: Transpose
    let mut transposed = [0u16; 1024];
    Transpose::transpose(&linear, &mut transposed);

    println!("\n=== Step 1: After Transpose ===");
    println!("First 16 values: {:?}", &transposed[0..16]);
    println!("transposed[0] came from linear[{}]", transposed[0]);
    println!("transposed[1] came from linear[{}]", transposed[1]);
    println!("transposed[64] came from linear[{}]", transposed[64]);
    println!("transposed[128] came from linear[{}]", transposed[128]);

    // Verify lane 0 elements in transposed array
    println!("\nLane 0 in transposed (positions where lane 0 data lives):");
    // According to iterate! macro, lane 0 visits indices: 0, 128, 256, ... for rows 0-7
    // then 64, 192, 320, ... for rows 8-15
    let lane0_positions = [
        0, 128, 256, 384, 512, 640, 768, 896, 64, 192, 320, 448, 576, 704, 832, 960,
    ];
    for (row, &pos) in lane0_positions.iter().enumerate() {
        println!(
            "  row {}: transposed[{}] = {} (came from linear[{}])",
            row, pos, transposed[pos], transposed[pos]
        );
    }

    // Step 2: Delta on transposed data
    let mut deltas = [0u16; 1024];
    let bases: [u16; LANES] = std::array::from_fn(|lane| transposed[lane]); // First element of each lane
    Delta::delta::<LANES>(&transposed, &bases, &mut deltas);

    println!("\n=== Step 2: After Delta ===");
    println!("Bases (first {} values): {:?}", 8, &bases[0..8]);
    println!("First 16 delta values: {:?}", &deltas[0..16]);

    println!("\nLane 0 deltas (at same positions as transposed):");
    for (row, &pos) in lane0_positions.iter().enumerate() {
        println!("  row {}: deltas[{}] = {}", row, pos, deltas[pos]);
    }

    // Step 3: BitPack the deltas
    // Use bit_width = 10 (enough for deltas up to ~1000)
    const W: usize = 10;
    const B: usize = 1024 * W / 16; // = 640
    let mut packed = [0u16; B];
    BitPacking::pack::<W, B>(&deltas, &mut packed);

    println!("\n=== Step 3: After BitPack ===");
    println!("Packed length: {} u16 values", packed.len());
    println!("First 16 packed values: {:?}", &packed[0..16]);

    // Step 4: Unpack
    let mut unpacked = [0u16; 1024];
    BitPacking::unpack::<W, B>(&packed, &mut unpacked);

    println!("\n=== Step 4: After Unpack ===");
    println!("First 16 unpacked values: {:?}", &unpacked[0..16]);
    assert_eq!(deltas, unpacked, "Unpack should restore deltas exactly");
    println!("✓ Unpack matches deltas");

    // Step 5: Undelta
    let mut undelta_result = [0u16; 1024];
    Delta::undelta::<LANES>(&unpacked, &bases, &mut undelta_result);

    println!("\n=== Step 5: After Undelta ===");
    println!("First 16 values: {:?}", &undelta_result[0..16]);
    assert_eq!(
        transposed, undelta_result,
        "Undelta should restore transposed exactly"
    );
    println!("✓ Undelta matches transposed");

    // Step 6: Untranspose
    let mut final_linear = [0u16; 1024];
    Transpose::untranspose(&undelta_result, &mut final_linear);

    println!("\n=== Step 6: After Untranspose ===");
    println!("First 16 values: {:?}", &final_linear[0..16]);
    assert_eq!(
        linear, final_linear,
        "Full round-trip should restore original"
    );
    println!("✓ Full round-trip successful!");
}

/// Test what happens if we DON'T transpose before delta+bitpack
#[test]
fn test_without_transpose() {
    const LANES: usize = 64;

    let linear: [u16; 1024] = std::array::from_fn(|i| i as u16);

    // Skip transpose, apply delta directly to linear data
    let mut deltas = [0u16; 1024];
    let bases: [u16; LANES] = std::array::from_fn(|lane| linear[lane]);
    Delta::delta::<LANES>(&linear, &bases, &mut deltas);

    println!("\n=== Without Transpose: Delta on Linear ===");
    println!("Bases: {:?}", &bases[0..8]);
    println!("First 16 deltas: {:?}", &deltas[0..16]);

    // The delta values will be wrong because Delta::delta uses iterate!
    // which expects transposed layout
    println!("\nLane 0 deltas (iterate! pattern on LINEAR data):");
    let lane0_positions = [
        0, 128, 256, 384, 512, 640, 768, 896, 64, 192, 320, 448, 576, 704, 832, 960,
    ];
    for (row, &pos) in lane0_positions.iter().enumerate() {
        println!(
            "  row {}: deltas[{}] = {} (linear[{}] was {})",
            row, pos, deltas[pos], pos, linear[pos]
        );
    }

    // Round trip
    let mut undelta_result = [0u16; 1024];
    Delta::undelta::<LANES>(&deltas, &bases, &mut undelta_result);

    // This should still work because delta/undelta use same pattern
    assert_eq!(
        linear, undelta_result,
        "Round trip should work even without transpose"
    );
    println!("\n✓ Round trip works (but deltas are not optimal for compression)");
}

/// Test to see what the iterate! macro index pattern actually is
#[test]
fn test_iterate_pattern() {
    use fastlanes::FL_ORDER;

    println!("\n=== iterate! Index Pattern ===");
    println!("FL_ORDER = {:?}", FL_ORDER);

    fn index(row: usize, lane: usize) -> usize {
        let o = row / 8;
        let s = row % 8;
        (FL_ORDER[o] * 16) + (s * 128) + lane
    }

    println!("\nLane 0 indices:");
    for row in 0..16 {
        println!("  row {:2}: index = {:4}", row, index(row, 0));
    }

    println!("\nLane 1 indices:");
    for row in 0..16 {
        println!("  row {:2}: index = {:4}", row, index(row, 1));
    }

    println!("\nRow 0 indices (all lanes 0-7):");
    for lane in 0..8 {
        println!("  lane {}: index = {:4}", lane, index(0, lane));
    }
}

/// Test what happens when you compose Delta (explicit transpose) + BitPack (implicit transpose)
#[test]
fn test_composition_problem() {
    const LANES: usize = 64;
    const W: usize = 10;
    const B: usize = 1024 * W / 16;

    // Create linear input
    let linear: [u16; 1024] = std::array::from_fn(|i| i as u16);

    println!("\n=== Composition Problem Test ===");

    // === CORRECT WAY: Transpose once, then delta, then linear-pack ===
    // (But fastlanes BitPacking::pack uses FL_ORDER, so this is wrong)

    // Step 1: Transpose
    let mut transposed = [0u16; 1024];
    Transpose::transpose(&linear, &mut transposed);

    // Step 2: Delta on transposed
    let bases: [u16; LANES] = std::array::from_fn(|lane| transposed[lane]);
    let mut deltas = [0u16; 1024];
    Delta::delta::<LANES>(&transposed, &bases, &mut deltas);

    // Step 3: BitPack the deltas (THIS IS THE PROBLEM - it uses FL_ORDER again!)
    let mut packed = [0u16; B];
    BitPacking::pack::<W, B>(&deltas, &mut packed);

    // Now unpack
    let mut unpacked = [0u16; 1024];
    BitPacking::unpack::<W, B>(&packed, &mut unpacked);

    // Check if unpacked matches deltas
    println!("After pack+unpack, do we get deltas back?");
    println!("  deltas[0..8]: {:?}", &deltas[0..8]);
    println!("  unpacked[0..8]: {:?}", &unpacked[0..8]);

    // They should match because pack/unpack are inverses
    assert_eq!(deltas, unpacked, "pack/unpack should be inverses");
    println!("✓ pack/unpack are inverses");

    // But wait - the issue is that BitPacking reads in FL_ORDER.
    // If deltas are already in FL_ORDER (from transpose+delta),
    // then pack reads them in FL_ORDER order, which is... actually correct!
    //
    // Let me trace through:
    // - deltas[0] is lane 0, row 0 delta (in transposed layout)
    // - pack! reads deltas[index(0, 0)] = deltas[0] = lane 0, row 0
    // - This is correct!

    // The iterate!/pack!/unpack! macros use index(row, lane) which gives FL_ORDER positions.
    // After transpose, the data IS in FL_ORDER layout.
    // So delta writes to FL_ORDER positions.
    // Then pack reads from FL_ORDER positions.
    // It all matches!

    // Let's verify full round-trip
    let mut undelta_result = [0u16; 1024];
    Delta::undelta::<LANES>(&unpacked, &bases, &mut undelta_result);

    let mut final_linear = [0u16; 1024];
    Transpose::untranspose(&undelta_result, &mut final_linear);

    assert_eq!(
        linear, final_linear,
        "Full round-trip with bitpacking should work"
    );
    println!("✓ Full round-trip with bitpacking works!");

    // === NOW TEST: What if we DON'T transpose before delta? ===
    println!("\n--- Without initial transpose ---");

    // Delta directly on linear (BAD)
    let bases_linear: [u16; LANES] = std::array::from_fn(|lane| linear[lane]);
    let mut deltas_linear = [0u16; 1024];
    Delta::delta::<LANES>(&linear, &bases_linear, &mut deltas_linear);

    // Pack
    let mut packed_linear = [0u16; B];
    BitPacking::pack::<W, B>(&deltas_linear, &mut packed_linear);

    // Unpack
    let mut unpacked_linear = [0u16; 1024];
    BitPacking::unpack::<W, B>(&packed_linear, &mut unpacked_linear);

    // Undelta
    let mut undelta_linear = [0u16; 1024];
    Delta::undelta::<LANES>(&unpacked_linear, &bases_linear, &mut undelta_linear);

    // Check: do we get back linear?
    println!("Without transpose, do we get original back?");
    println!("  linear[0..8]: {:?}", &linear[0..8]);
    println!("  recovered[0..8]: {:?}", &undelta_linear[0..8]);

    // This DOES NOT work! The data gets scrambled.
    // FL_ORDER operations don't cancel - you NEED the transpose.
    if linear != undelta_linear {
        println!("✗ Round-trip WITHOUT transpose FAILS (data is scrambled)");
        println!("  Expected[440..450]: {:?}", &linear[440..450]);
        println!("  Got[440..450]: {:?}", &undelta_linear[440..450]);
    } else {
        println!("✓ Round-trip works (unexpected!)");
    }

    println!("\nDelta values comparison:");
    println!(
        "  With transpose - deltas[128] (lane 0, row 1): {}",
        deltas[128]
    );
    println!("  Without transpose - deltas[128]: {}", deltas_linear[128]);
}

/// Test fused vs unfused delta+bitpacking kernels
#[test]
fn test_fused_vs_unfused_kernels() {
    use fastlanes::FL_ORDER;

    const LANES: usize = 64;
    const W: usize = 10;
    const B: usize = 1024 * W / 16; // = 640

    // Create linear input
    let linear: [u16; 1024] = std::array::from_fn(|i| i as u16);

    println!("\n=== Fused vs Unfused Kernels ===");

    // === Compression path ===
    // Step 1: Transpose (same for both)
    let mut transposed = [0u16; 1024];
    Transpose::transpose(&linear, &mut transposed);

    // Step 2: Delta (same for both)
    let bases: [u16; LANES] = std::array::from_fn(|lane| transposed[lane]);
    let mut deltas = [0u16; 1024];
    Delta::delta::<LANES>(&transposed, &bases, &mut deltas);

    // Step 3: BitPack (same for both)
    let mut packed = [0u16; B];
    BitPacking::pack::<W, B>(&deltas, &mut packed);

    println!("Compressed {} u16 values into {} packed values", 1024, B);
    println!("Compression ratio: {:.2}x", 1024.0 / B as f64);

    // === Decompression: UNFUSED path ===
    println!("\n--- Unfused decompression ---");
    let mut intermediate_unpacked = [0u16; 1024]; // EXTRA ALLOCATION
    BitPacking::unpack::<W, B>(&packed, &mut intermediate_unpacked);

    let mut unfused_transposed = [0u16; 1024];
    Delta::undelta::<LANES>(&intermediate_unpacked, &bases, &mut unfused_transposed);

    let mut unfused_result = [0u16; 1024];
    Transpose::untranspose(&unfused_transposed, &mut unfused_result);

    println!("Operations: unpack → undelta → untranspose");
    println!("Memory: needs TWO intermediate 1024-element buffers");

    // === Decompression: FUSED path ===
    println!("\n--- Fused decompression ---");
    let mut fused_result = [0u16; 1024]; // NO intermediate allocation
    Delta::undelta_pack::<LANES, W, B>(&packed, &bases, &mut fused_result);

    // Note: fused result is in transposed layout, needs untranspose
    let mut fused_final = [0u16; 1024];
    Transpose::untranspose(&fused_result, &mut fused_final);

    println!("Operations: undelta_pack (fused) → untranspose");
    println!("Memory: NO intermediate buffer needed");

    // Verify both produce same result
    assert_eq!(
        unfused_result, fused_final,
        "Fused and unfused should match"
    );
    assert_eq!(linear, fused_final, "Should recover original");
    println!("\n✓ Both paths produce identical results");

    // === What the fused kernel saves ===
    println!("\n--- Savings Analysis ---");
    println!("Unfused decompression on 1024 elements:");
    println!("  1. BitPacking::unpack: 640 reads, 1024 writes (to intermediate buffer 1)");
    println!("  2. Delta::undelta: 1024 reads, 1024 writes (to intermediate buffer 2)");
    println!("  3. Transpose::untranspose: 1024 reads, 1024 writes");
    println!("  Memory: needs 2 intermediate 1024-element buffers");
    println!("\nFused decompression:");
    println!("  1. Delta::undelta_pack: 640 reads, 1024 writes");
    println!("  2. Transpose::untranspose: 1024 reads, 1024 writes");
    println!("  Memory: needs 1 intermediate 1024-element buffer");
    println!("\nSavings: ~33% reduction in memory ops, 50% less intermediate memory!");

    // The real benefit is cache efficiency
    println!("\n--- Cache Efficiency ---");
    println!("Unfused: Each unpacked value may be evicted from cache before undelta reads it");
    println!("Fused: Each value is processed immediately after unpacking, while in registers");

    // Show how iterate! pattern affects this
    println!("\n--- Why FL_ORDER Iteration Order Matters ---");
    println!("FL_ORDER = {:?}", FL_ORDER);
    println!("For lane 0, the iteration visits indices:");
    fn index(row: usize, lane: usize) -> usize {
        let o = row / 8;
        let s = row % 8;
        (FL_ORDER[o] * 16) + (s * 128) + lane
    }
    for row in 0..4 {
        println!("  row {}: index {}", row, index(row, 0));
    }
    println!("  ...");
    println!("This non-sequential access pattern means intermediate buffer");
    println!("causes cache misses in unfused path.");
}

/// Compare Vortex's delta compress with raw fastlanes
#[test]
fn test_vortex_delta_compress() {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ToCanonical};
    use vortex_buffer::Buffer;

    // Create a primitive array
    let values: Buffer<u16> = (0u16..2048).collect();
    let parray = PrimitiveArray::new(values, Validity::NonNullable);

    println!("\n=== Vortex Delta Compress ===");
    println!("Input length: {}", parray.len());

    // Use Vortex's DeltaArray::try_from_primitive_array (public API)
    use crate::DeltaArray;
    let delta_array = DeltaArray::try_from_primitive_array(&parray).expect("delta compress failed");

    println!("DeltaArray created successfully");
    println!("  bases dtype: {:?}", delta_array.bases().dtype());
    println!(
        "  deltas encoding: {:?}",
        delta_array.deltas().encoding_id()
    );

    // Decompress and verify
    let decompressed = delta_array.to_primitive();

    println!("  decompressed length: {}", decompressed.len());

    // Verify round-trip
    for i in 0..10 {
        let original = i as u16;
        let recovered = decompressed.as_slice::<u16>()[i];
        println!("  [{}] original={}, recovered={}", i, original, recovered);
        assert_eq!(original, recovered);
    }
    println!("✓ Vortex delta round-trip successful");
}
