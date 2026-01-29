# Constrained Arbitrary Arrays - Implementation Progress

## Completed

- [x] Add ArrayConstraints and ConstraintKind types
- [x] Add ArbitraryConstrained trait
- [x] Implement constrained generation for PrimitiveArray (base case)
- [x] Implement ArbitraryConstrained for ConstantArray
- [x] Implement ArbitraryConstrained for SequenceArray
- [x] Implement ArbitraryConstrained for DeltaArray
- [x] Implement ArbitraryConstrained for FoRArray
- [x] Implement ArbitraryConstrained for BitPackedArray
- [x] Implement ArbitrarySparseArray with constrained indices
- [x] Update ArbitraryRunEndArray to use constrained generation
- [x] Update ArbitraryDictArray to use constrained generation
- [x] Implement ArbitraryRLEArray (fastlanes)

## Pending

- [ ] Implement ArbitraryZigZagArray
- [ ] Implement ArbitraryByteBoolArray
- [ ] Implement ArbitraryFSSTArray
- [ ] Implement ArbitraryDateTimePartsArray
- [ ] Implement ArbitraryALPArray
- [ ] Implement ArbitraryALPRDArray
- [ ] Implement ArbitraryPcoArray
- [ ] Implement ArbitraryZstdArray
- [ ] Implement ArbitraryMaskedArray
- [ ] Implement ArbitraryExtensionArray
- [ ] Implement ArbitraryDecimalBytePartsArray
