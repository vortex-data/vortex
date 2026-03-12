// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright SpiralDB Developers

//! Functions and types used for building a [`Compressor`] from a corpus of text.
//!
//! This module implements the logic from Algorithm 3 of the [FSST Paper].
//!
//! [FSST Paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::Code;
use crate::Compressor;
use crate::FSST_CODE_BASE;
use crate::FSST_CODE_MASK;
use crate::Symbol;
use crate::advance_8byte_word;
use crate::compare_masked;
use crate::lossy_pht::LossyPHT;

/// Bitmap that only works for values up to 512
#[derive(Clone, Copy, Debug, Default)]
struct CodesBitmap {
    codes: [u64; 8],
}

assert_sizeof!(CodesBitmap => 64);

impl CodesBitmap {
    /// Set the indicated bit. Must be between 0 and [`FSST_CODE_MASK`][crate::FSST_CODE_MASK].
    pub(crate) fn set(&mut self, index: usize) {
        debug_assert!(
            index <= FSST_CODE_MASK as usize,
            "code cannot exceed {FSST_CODE_MASK}"
        );

        let map = index >> 6;
        self.codes[map] |= 1 << (index % 64);
    }

    /// Check if `index` is present in the bitmap
    pub(crate) fn is_set(&self, index: usize) -> bool {
        debug_assert!(
            index <= FSST_CODE_MASK as usize,
            "code cannot exceed {FSST_CODE_MASK}"
        );

        let map = index >> 6;
        self.codes[map] & (1 << (index % 64)) != 0
    }

    /// Get all codes set in this bitmap
    pub(crate) fn codes(&self) -> CodesIterator<'_> {
        CodesIterator {
            inner: self,
            index: 0,
            block: self.codes[0],
            reference: 0,
        }
    }

    /// Clear the bitmap of all entries.
    pub(crate) fn clear(&mut self) {
        self.codes[0] = 0;
        self.codes[1] = 0;
        self.codes[2] = 0;
        self.codes[3] = 0;
        self.codes[4] = 0;
        self.codes[5] = 0;
        self.codes[6] = 0;
        self.codes[7] = 0;
    }
}

struct CodesIterator<'a> {
    inner: &'a CodesBitmap,
    index: usize,
    block: u64,
    reference: usize,
}

impl Iterator for CodesIterator<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        // If current is zero, advance to next non-zero block
        while self.block == 0 {
            self.index += 1;
            if self.index >= 8 {
                return None;
            }
            self.block = self.inner.codes[self.index];
            self.reference = self.index * 64;
        }

        // Find the next set bit in the current block.
        let position = self.block.trailing_zeros() as usize;
        let code = self.reference + position;

        if code >= 511 {
            return None;
        }

        // The next iteration will calculate with reference to the returned code + 1
        self.reference = code + 1;
        self.block = if position == 63 {
            0
        } else {
            self.block >> (1 + position)
        };

        Some(code as u16)
    }
}

#[derive(Debug, Clone)]
struct Counter {
    /// Frequency count for each code.
    counts1: Vec<usize>,

    /// Frequency count for each code-pair.
    counts2: Vec<usize>,

    /// Bitmap index for codes that appear in counts1
    code1_index: CodesBitmap,

    /// Bitmap index of pairs that have been set.
    ///
    /// `pair_index[code1].codes()` yields an iterator that can
    /// be used to find all possible codes that follow `codes1`.
    pair_index: Vec<CodesBitmap>,
}

const COUNTS1_SIZE: usize = (FSST_CODE_MASK + 1) as usize;

// NOTE: in Rust, creating a 1D vector of length N^2 is ~4x faster than creating a 2-D vector,
//  because `vec!` has a specialization for zero.
//
// We also include +1 extra row at the end so that we can do writes into the counters without a branch
// for the first iteration.
const COUNTS2_SIZE: usize = COUNTS1_SIZE * COUNTS1_SIZE;

impl Counter {
    fn new() -> Self {
        let mut counts1 = Vec::with_capacity(COUNTS1_SIZE);
        let mut counts2 = Vec::with_capacity(COUNTS2_SIZE);
        // SAFETY: all accesses to the vector go through the bitmap to ensure no uninitialized
        //  data is ever read from these vectors.
        unsafe {
            counts1.set_len(COUNTS1_SIZE);
            counts2.set_len(COUNTS2_SIZE);
        }

        Self {
            counts1,
            counts2,
            code1_index: CodesBitmap::default(),
            pair_index: vec![CodesBitmap::default(); COUNTS1_SIZE],
        }
    }

    #[inline]
    fn record_count1(&mut self, code1: u16) {
        // If not set, we want to start at one.
        let base = if self.code1_index.is_set(code1 as usize) {
            self.counts1[code1 as usize]
        } else {
            0
        };

        self.counts1[code1 as usize] = base + 1;
        self.code1_index.set(code1 as usize);
    }

    #[inline]
    fn record_count2(&mut self, code1: u16, code2: u16) {
        debug_assert!(code1 == FSST_CODE_MASK || self.code1_index.is_set(code1 as usize));
        debug_assert!(self.code1_index.is_set(code2 as usize));

        let idx = (code1 as usize) * COUNTS1_SIZE + (code2 as usize);
        if self.pair_index[code1 as usize].is_set(code2 as usize) {
            self.counts2[idx] += 1;
        } else {
            self.counts2[idx] = 1;
        }
        self.pair_index[code1 as usize].set(code2 as usize);
    }

    #[inline]
    fn count1(&self, code1: u16) -> usize {
        debug_assert!(self.code1_index.is_set(code1 as usize));

        self.counts1[code1 as usize]
    }

    #[inline]
    fn count2(&self, code1: u16, code2: u16) -> usize {
        debug_assert!(self.code1_index.is_set(code1 as usize));
        debug_assert!(self.code1_index.is_set(code2 as usize));
        debug_assert!(self.pair_index[code1 as usize].is_set(code2 as usize));

        let idx = (code1 as usize) * 512 + (code2 as usize);
        self.counts2[idx]
    }

    /// Returns an ordered iterator over the codes that were observed
    /// in a call to [`Self::count1`].
    fn first_codes(&self) -> CodesIterator<'_> {
        self.code1_index.codes()
    }

    /// Returns an iterator over the codes that have been observed
    /// to follow `code1`.
    ///
    /// This is the set of all values `code2` where there was
    /// previously a call to `self.record_count2(code1, code2)`.
    fn second_codes(&self, code1: u16) -> CodesIterator<'_> {
        self.pair_index[code1 as usize].codes()
    }

    /// Clear the counters.
    /// Note that this just touches the bitmaps and sets them all to invalid.
    fn clear(&mut self) {
        self.code1_index.clear();
        for index in &mut self.pair_index {
            index.clear();
        }
    }
}

/// Entrypoint for building a new `Compressor`.
pub struct CompressorBuilder {
    /// Table mapping codes to symbols.
    ///
    /// The entries 0-255 are setup in some other way here
    symbols: Vec<Symbol>,

    /// The number of entries in the symbol table that have been populated, not counting
    /// the escape values.
    n_symbols: u8,

    /// Counts for number of symbols of each length.
    ///
    /// `len_histogram[len-1]` = count of the symbols of length `len`.
    len_histogram: [u8; 8],

    /// Inverted index mapping 1-byte symbols to codes.
    ///
    /// This is only used for building, not used by the final `Compressor`.
    codes_one_byte: Vec<Code>,

    /// Inverted index mapping 2-byte symbols to codes
    codes_two_byte: Vec<Code>,

    /// Lossy perfect hash table for looking up codes to symbols that are 3 bytes or more
    lossy_pht: LossyPHT,
}

impl CompressorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        // NOTE: `vec!` has a specialization for building a new vector of `0u64`. Because Symbol and u64
        //  have the same bit pattern, we can allocate as u64 and transmute. If we do `vec![Symbol::EMPTY; N]`,
        // that will create a new Vec and call `Symbol::EMPTY.clone()` `N` times which is considerably slower.
        let symbols = vec![0u64; 511];

        // SAFETY: transmute safety assured by the compiler.
        let symbols: Vec<Symbol> = unsafe { std::mem::transmute(symbols) };

        let mut table = Self {
            symbols,
            n_symbols: 0,
            len_histogram: [0; 8],
            codes_two_byte: Vec::with_capacity(65_536),
            codes_one_byte: Vec::with_capacity(512),
            lossy_pht: LossyPHT::new(),
        };

        // Populate the escape byte entries.
        for byte in 0..=255 {
            let symbol = Symbol::from_u8(byte);
            table.symbols[byte as usize] = symbol;
        }

        // Fill codes_one_byte with pseudocodes for each byte.
        for byte in 0..=255 {
            // Push pseudocode for single-byte escape.
            table.codes_one_byte.push(Code::new_escape(byte));
        }

        // Fill codes_two_byte with pseudocode of first byte
        for idx in 0..=65_535 {
            table.codes_two_byte.push(Code::new_escape(idx as u8));
        }

        table
    }
}

impl Default for CompressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressorBuilder {
    /// Attempt to insert a new symbol at the end of the table.
    ///
    /// # Panics
    ///
    /// Panics if the table is already full.
    ///
    /// # Returns
    ///
    /// Returns true if the symbol was inserted successfully, or false if it conflicted
    /// with an existing symbol.
    pub fn insert(&mut self, symbol: Symbol, len: usize) -> bool {
        assert!(self.n_symbols < 255, "cannot insert into full symbol table");
        assert_eq!(len, symbol.len(), "provided len must equal symbol.len()");

        if len == 2 {
            // Check if this 2-byte symbol is already in the table.
            if self.codes_two_byte[symbol.first2() as usize].extended_code() >= FSST_CODE_BASE {
                return false;
            }
            self.codes_two_byte[symbol.first2() as usize] =
                Code::new_symbol_building(self.n_symbols, 2);
        } else if len == 1 {
            // Check if this 1-byte symbol is already in the table.
            if self.codes_one_byte[symbol.first_byte() as usize].extended_code() >= FSST_CODE_BASE {
                return false;
            }
            self.codes_one_byte[symbol.first_byte() as usize] =
                Code::new_symbol_building(self.n_symbols, 1);
        } else {
            // Symbols of 3 or more bytes go into the hash table
            if !self.lossy_pht.insert(symbol, len, self.n_symbols) {
                return false;
            }
        }

        // Increment length histogram.
        self.len_histogram[len - 1] += 1;

        // Insert successfully stored symbol at end of the symbol table
        // Note the rescaling from range [0-254] -> [256, 510].
        self.symbols[256 + (self.n_symbols as usize)] = symbol;
        self.n_symbols += 1;
        true
    }

    /// Clear all set items from the compressor.
    ///
    /// This is considerably faster than building a new Compressor from scratch for each
    /// iteration of the `train` loop.
    fn clear(&mut self) {
        // Eliminate every observed code from the table.
        for code in 0..(256 + self.n_symbols as usize) {
            let symbol = self.symbols[code];
            if symbol.len() == 1 {
                // Reset the entry from the codes_one_byte array.
                self.codes_one_byte[symbol.first_byte() as usize] =
                    Code::new_escape(symbol.first_byte());
            } else if symbol.len() == 2 {
                // Reset the entry from the codes_two_byte array.
                self.codes_two_byte[symbol.first2() as usize] =
                    Code::new_escape(symbol.first_byte());
            } else {
                // Clear the hashtable entry
                self.lossy_pht.remove(symbol);
            }
        }

        // Reset len histogram
        for i in 0..=7 {
            self.len_histogram[i] = 0;
        }

        self.n_symbols = 0;
    }

    /// Finalizing the table is done once building is complete to prepare for efficient
    /// compression.
    ///
    /// When we finalize the table, the following modifications are made in-place:
    ///
    /// 1. The codes are renumbered so that all symbols are ordered by length (order 23456781).
    ///    During this process, the two byte symbols are separated into a byte_lim and a suffix_lim,
    ///    so we know that we don't need to check the suffix limitations instead.
    /// 2. The 1-byte symbols index is merged into the 2-byte symbols index to allow for use of only
    ///    a single index in front of the hash table.
    ///
    /// # Returns
    ///
    /// Returns the `suffix_lim`, which is the index of the two-byte code before where we know
    /// there are no longer suffixies in the symbol table.
    ///
    /// Also returns the lengths vector, which is of length `n_symbols` and contains the
    /// length for each of the values.
    fn finalize(&mut self) -> (u8, Vec<u8>) {
        // Create a cumulative sum of each of the elements of the input line numbers.
        // Do a map that includes the previously seen value as well.
        // Regroup symbols based on their lengths.
        // Space at the end of the symbol table reserved for the one-byte codes.
        let byte_lim = self.n_symbols - self.len_histogram[0];

        // Start code for each length.
        // Length 1: at the end of symbol table.
        // Length 2: starts at 0. Split into before/after suffixLim.
        let mut codes_by_length = [0u8; 8];
        codes_by_length[0] = byte_lim;
        codes_by_length[1] = 0;

        // codes for lengths 3..=8 start where the previous ones end.
        for i in 1..7 {
            codes_by_length[i + 1] = codes_by_length[i] + self.len_histogram[i];
        }

        // no_suffix_code is the lowest code for a symbol that does not have a longer 3+ byte
        // suffix in the table.
        // This value starts at 0 and extends up.
        let mut no_suffix_code = 0;

        // The codes that do not have a suffix begin just before the range of the 3-byte codes.
        let mut has_suffix_code = codes_by_length[2];

        // Assign each symbol a new code ordered by lengths, in the order
        // 2(no suffix) | 2 (suffix) | 3 | 4 | 5 | 6 | 7 | 8 | 1
        let mut new_codes = [0u8; FSST_CODE_BASE as usize];

        let mut symbol_lens = [0u8; FSST_CODE_BASE as usize];

        for i in 0..(self.n_symbols as usize) {
            let symbol = self.symbols[256 + i];
            let len = symbol.len();
            if len == 2 {
                let has_suffix = self
                    .symbols
                    .iter()
                    .skip(FSST_CODE_BASE as usize)
                    .enumerate()
                    .any(|(k, other)| i != k && symbol.first2() == other.first2());

                if has_suffix {
                    // Symbols that have a longer suffix are inserted at the end of the 2-byte range
                    has_suffix_code -= 1;
                    new_codes[i] = has_suffix_code;
                } else {
                    // Symbols that do not have a longer suffix are inserted at the start of
                    // the 2-byte range.
                    new_codes[i] = no_suffix_code;
                    no_suffix_code += 1;
                }
            } else {
                // Assign new code based on the next code available for the given length symbol
                new_codes[i] = codes_by_length[len - 1];
                codes_by_length[len - 1] += 1;
            }

            // Write the symbol into the front half of the symbol table.
            // We are reusing the space that was previously occupied by escapes.
            self.symbols[new_codes[i] as usize] = symbol;
            symbol_lens[new_codes[i] as usize] = len as u8;
        }

        // Truncate the symbol table to only include the "true" symbols.
        self.symbols.truncate(self.n_symbols as usize);

        // Rewrite the codes_one_byte table to point at the new code values.
        // Replace pseudocodes with escapes.
        for byte in 0..=255 {
            let one_byte = self.codes_one_byte[byte];
            if one_byte.extended_code() >= FSST_CODE_BASE {
                let new_code = new_codes[one_byte.code() as usize];
                self.codes_one_byte[byte] = Code::new_symbol(new_code, 1);
            } else {
                // After finalize: codes_one_byte contains the unused value
                self.codes_one_byte[byte] = Code::UNUSED;
            }
        }

        // Rewrite the codes_two_byte table to point at the new code values.
        // Replace pseudocodes with escapes.
        for two_bytes in 0..=65_535 {
            let two_byte = self.codes_two_byte[two_bytes];
            if two_byte.extended_code() >= FSST_CODE_BASE {
                let new_code = new_codes[two_byte.code() as usize];
                self.codes_two_byte[two_bytes] = Code::new_symbol(new_code, 2);
            } else {
                // The one-byte code for the given code number here...
                self.codes_two_byte[two_bytes] = self.codes_one_byte[two_bytes & 0xFF];
            }
        }

        // Reset values in the hash table as well.
        self.lossy_pht.renumber(&new_codes);

        // Pre-compute the lengths
        let mut lengths = Vec::with_capacity(self.n_symbols as usize);
        for symbol in &self.symbols {
            lengths.push(symbol.len() as u8);
        }

        (has_suffix_code, lengths)
    }

    /// Build into the final hash table.
    pub fn build(mut self) -> Compressor {
        // finalize the symbol table by inserting the codes_twobyte values into
        // the relevant parts of the `codes_onebyte` set.

        let (has_suffix_code, lengths) = self.finalize();

        Compressor {
            symbols: self.symbols,
            lengths,
            n_symbols: self.n_symbols,
            has_suffix_code,
            codes_two_byte: self.codes_two_byte,
            lossy_pht: self.lossy_pht,
        }
    }
}

/// The number of generations used for training. This is taken from the [FSST paper].
///
/// [FSST paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
#[cfg(not(miri))]
const GENERATIONS: [usize; 5] = [8usize, 38, 68, 98, 128];
#[cfg(miri)]
const GENERATIONS: [usize; 3] = [8usize, 38, 128];

const FSST_SAMPLETARGET: usize = 1 << 14;
const FSST_SAMPLEMAX: usize = 1 << 15;
const FSST_SAMPLELINE: usize = 512;

/// Create a sample from a set of strings in the input.
///
/// Sample is constructing by copying "chunks" from the `str_in`s into the `sample_buf`, the
/// returned slices are pointers into the `sample_buf`.
///
/// SAFETY: sample_buf must be >= FSST_SAMPLEMAX bytes long. Providing something less may cause unexpected failures.
#[allow(clippy::ptr_arg)]
fn make_sample<'a, 'b: 'a>(sample_buf: &'a mut Vec<u8>, str_in: &Vec<&'b [u8]>) -> Vec<&'a [u8]> {
    assert!(
        sample_buf.capacity() >= FSST_SAMPLEMAX,
        "sample_buf.len() < FSST_SAMPLEMAX"
    );

    let mut sample: Vec<&[u8]> = Vec::new();

    let tot_size: usize = str_in.iter().map(|s| s.len()).sum();
    if tot_size < FSST_SAMPLETARGET {
        return str_in.clone();
    }

    let mut sample_rnd = fsst_hash(4637947);
    let sample_lim = FSST_SAMPLETARGET;
    let mut sample_buf_offset: usize = 0;

    while sample_buf_offset < sample_lim {
        sample_rnd = fsst_hash(sample_rnd);
        let line_nr = (sample_rnd as usize) % str_in.len();

        // Find the first non-empty chunk starting at line_nr, wrapping around if
        // necessary.
        let Some(line) = (line_nr..str_in.len())
            .chain(0..line_nr)
            .map(|line_nr| str_in[line_nr])
            .find(|line| !line.is_empty())
        else {
            return sample;
        };

        let chunks = 1 + ((line.len() - 1) / FSST_SAMPLELINE);
        sample_rnd = fsst_hash(sample_rnd);
        let chunk = FSST_SAMPLELINE * ((sample_rnd as usize) % chunks);

        let len = FSST_SAMPLELINE.min(line.len() - chunk);

        sample_buf.extend_from_slice(&line[chunk..chunk + len]);

        // SAFETY: this is the data we just placed into `sample_buf` in the line above.
        let slice =
            unsafe { std::slice::from_raw_parts(sample_buf.as_ptr().add(sample_buf_offset), len) };

        sample.push(slice);

        sample_buf_offset += len;
    }

    sample
}

/// Hash function used in various components of the library.
///
/// This is equivalent to the FSST_HASH macro from the C++ implementation.
#[inline]
pub(crate) fn fsst_hash(value: u64) -> u64 {
    value.wrapping_mul(2971215073) ^ value.wrapping_shr(15)
}

impl Compressor {
    /// Build and train a `Compressor` from a sample corpus of text.
    ///
    /// This function implements the generational algorithm described in the [FSST paper] Section
    /// 4.3. Starting with an empty symbol table, it iteratively compresses the corpus, then attempts
    /// to merge symbols when doing so would yield better compression than leaving them unmerged. The
    /// resulting table will have at most 255 symbols (the 256th symbol is reserved for the escape
    /// code).
    ///
    /// [FSST paper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
    pub fn train(values: &Vec<&[u8]>) -> Self {
        let mut builder = CompressorBuilder::new();

        if values.is_empty() {
            return builder.build();
        }

        let mut counters = Counter::new();
        let mut sample_memory = Vec::with_capacity(FSST_SAMPLEMAX);
        let mut pqueue = BinaryHeap::with_capacity(65_536);

        let sample = make_sample(&mut sample_memory, values);
        for sample_frac in GENERATIONS {
            for (i, line) in sample.iter().enumerate() {
                if sample_frac < 128 && ((fsst_hash(i as u64) & 127) as usize) > sample_frac {
                    continue;
                }

                builder.compress_count(line, &mut counters);
            }

            // Clear the heap before we use it again
            pqueue.clear();
            builder.optimize(&counters, sample_frac, &mut pqueue);
            counters.clear();
        }

        builder.build()
    }
}

impl CompressorBuilder {
    /// Find the longest symbol using the hash table and the codes_one_byte and codes_two_byte indexes.
    fn find_longest_symbol(&self, word: u64) -> Code {
        // Probe the hash table first to see if we have a long match
        let entry = self.lossy_pht.lookup(word);
        let ignored_bits = entry.ignored_bits;

        // If the entry is valid, return the code
        if !entry.is_unused() && compare_masked(word, entry.symbol.to_u64(), ignored_bits) {
            return entry.code;
        }

        // Try and match first two bytes
        let twobyte = self.codes_two_byte[word as u16 as usize];
        if twobyte.extended_code() >= FSST_CODE_BASE {
            return twobyte;
        }

        // Fall back to single-byte match
        self.codes_one_byte[word as u8 as usize]
    }

    /// Compress the text using the current symbol table. Count the code occurrences
    /// and code-pair occurrences, calculating total gain using the current compressor.
    ///
    /// NOTE: this is largely an unfortunate amount of copy-paste from `compress`, just to make sure
    /// we can do all the counting in a single pass.
    fn compress_count(&self, sample: &[u8], counter: &mut Counter) -> usize {
        let mut gain = 0;
        if sample.is_empty() {
            return gain;
        }

        let mut in_ptr = sample.as_ptr();

        // SAFETY: `end` will point just after the end of the `plaintext` slice.
        let in_end = unsafe { in_ptr.byte_add(sample.len()) };
        let in_end_sub8 = in_end as usize - 8;

        let mut prev_code: u16 = FSST_CODE_MASK;

        while (in_ptr as usize) < (in_end_sub8) {
            // SAFETY: ensured in-bounds by loop condition.
            let word: u64 = unsafe { std::ptr::read_unaligned(in_ptr as *const u64) };
            let code = self.find_longest_symbol(word);
            let code_u16 = code.extended_code();

            // Gain increases by the symbol length if a symbol matches, or 0
            // if an escape is emitted.
            gain += (code.len() as usize) - ((code_u16 < 256) as usize);

            // Record the single and pair counts
            counter.record_count1(code_u16);
            counter.record_count2(prev_code, code_u16);

            // Also record the count for just extending by a single byte, but only if
            // the symbol is not itself a single byte.
            if code.len() > 1 {
                let code_first_byte = self.symbols[code_u16 as usize].first_byte() as u16;
                counter.record_count1(code_first_byte);
                counter.record_count2(prev_code, code_first_byte);
            }

            // SAFETY: pointer bound is checked in loop condition before any access is made.
            in_ptr = unsafe { in_ptr.byte_add(code.len() as usize) };

            prev_code = code_u16;
        }

        let remaining_bytes = unsafe { in_end.byte_offset_from(in_ptr) };
        assert!(
            remaining_bytes.is_positive(),
            "in_ptr exceeded in_end, should not be possible"
        );
        let remaining_bytes = remaining_bytes as usize;

        // Load the last `remaining_byte`s of data into a final world. We then replicate the loop above,
        // but shift data out of this word rather than advancing an input pointer and potentially reading
        // unowned memory
        let mut bytes = [0u8; 8];
        unsafe {
            // SAFETY: it is safe to read up to remaining_bytes from in_ptr, and remaining_bytes
            //  will be <= 8 bytes.
            std::ptr::copy_nonoverlapping(in_ptr, bytes.as_mut_ptr(), remaining_bytes);
        }
        let mut last_word = u64::from_le_bytes(bytes);

        let mut remaining_bytes = remaining_bytes;

        while remaining_bytes > 0 {
            // SAFETY: ensured in-bounds by loop condition.
            let code = self.find_longest_symbol(last_word);
            let code_u16 = code.extended_code();

            // Gain increases by the symbol length if a symbol matches, or 0
            // if an escape is emitted.
            gain += (code.len() as usize) - ((code_u16 < 256) as usize);

            // Record the single and pair counts
            counter.record_count1(code_u16);
            counter.record_count2(prev_code, code_u16);

            // Also record the count for just extending by a single byte, but only if
            // the symbol is not itself a single byte.
            if code.len() > 1 {
                let code_first_byte = self.symbols[code_u16 as usize].first_byte() as u16;
                counter.record_count1(code_first_byte);
                counter.record_count2(prev_code, code_first_byte);
            }

            // Advance our last_word "input pointer" by shifting off the covered values.
            let advance = code.len() as usize;
            remaining_bytes -= advance;
            last_word = advance_8byte_word(last_word, advance);

            prev_code = code_u16;
        }

        gain
    }

    /// Using a set of counters and the existing set of symbols, build a new
    /// set of symbols/codes that optimizes the gain over the distribution in `counter`.
    fn optimize(
        &mut self,
        counters: &Counter,
        sample_frac: usize,
        pqueue: &mut BinaryHeap<Candidate>,
    ) {
        for code1 in counters.first_codes() {
            let symbol1 = self.symbols[code1 as usize];
            let symbol1_len = symbol1.len();
            let count = counters.count1(code1);

            // From the c++ impl:
            // "improves both compression speed (less candidates), but also quality!!"
            if count < (5 * sample_frac / 128) {
                continue;
            }

            let mut gain = count * symbol1_len;
            // NOTE: use heuristic from C++ implementation to boost the gain of single-byte symbols.
            // This helps to reduce exception counts.
            if code1 < 256 {
                gain *= 8;
            }

            pqueue.push(Candidate {
                symbol: symbol1,
                gain,
            });

            // Skip merges on last round, or when symbol cannot be extended.
            if sample_frac >= 128 || symbol1_len == 8 {
                continue;
            }

            for code2 in counters.second_codes(code1) {
                let symbol2 = self.symbols[code2 as usize];

                // If merging would yield a symbol of length greater than 8, skip.
                if symbol1_len + symbol2.len() > 8 {
                    continue;
                }
                let new_symbol = symbol1.concat(symbol2);
                let gain = counters.count2(code1, code2) * new_symbol.len();

                pqueue.push(Candidate {
                    symbol: new_symbol,
                    gain,
                })
            }
        }

        // clear self in advance of inserting the symbols.
        self.clear();

        // Pop the 255 best symbols.
        let mut n_symbols = 0;
        while !pqueue.is_empty() && n_symbols < 255 {
            let candidate = pqueue.pop().unwrap();
            if self.insert(candidate.symbol, candidate.symbol.len()) {
                n_symbols += 1;
            }
        }
    }
}

/// A candidate for inclusion in a symbol table.
///
/// This is really only useful for the `optimize` step of training.
#[derive(Copy, Clone, Debug)]
struct Candidate {
    gain: usize,
    symbol: Symbol,
}

impl Candidate {
    fn comparable_form(&self) -> (usize, usize) {
        (self.gain, self.symbol.len())
    }
}

impl Eq for Candidate {}

impl PartialEq<Self> for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.comparable_form().eq(&other.comparable_form())
    }
}

impl PartialOrd<Self> for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_ord = (self.gain, self.symbol.len());
        let other_ord = (other.gain, other.symbol.len());

        self_ord.cmp(&other_ord)
    }
}

#[cfg(test)]
mod test {
    use crate::Compressor;
    use crate::ESCAPE_CODE;
    use crate::builder::CodesBitmap;

    #[test]
    fn test_builder() {
        // Train a Compressor on the toy string
        let text = b"hello hello hello hello hello";

        // count of 5 is the cutoff for including a symbol in the table.
        let table = Compressor::train(&vec![text, text, text, text, text]);

        // Use the table to compress a string, see the values
        let compressed = table.compress(text);

        // Ensure that the compressed string has no escape bytes
        assert!(compressed.iter().all(|b| *b != ESCAPE_CODE));

        // Ensure that we can compress a string with no values seen at training time, with escape bytes
        let compressed = table.compress("xyz123".as_bytes());
        let decompressed = table.decompressor().decompress(&compressed);
        assert_eq!(&decompressed, b"xyz123");
        assert_eq!(
            compressed,
            vec![
                ESCAPE_CODE,
                b'x',
                ESCAPE_CODE,
                b'y',
                ESCAPE_CODE,
                b'z',
                ESCAPE_CODE,
                b'1',
                ESCAPE_CODE,
                b'2',
                ESCAPE_CODE,
                b'3',
            ]
        );
    }

    #[test]
    fn test_bitmap() {
        let mut map = CodesBitmap::default();
        map.set(10);
        map.set(100);
        map.set(500);

        let codes: Vec<u16> = map.codes().collect();
        assert_eq!(codes, vec![10u16, 100, 500]);

        // empty case
        let map = CodesBitmap::default();
        assert!(map.codes().collect::<Vec<_>>().is_empty());

        // edge case: first bit in each block is set
        let mut map = CodesBitmap::default();
        (0..8).for_each(|i| map.set(64 * i));
        assert_eq!(
            map.codes().collect::<Vec<_>>(),
            (0u16..8).map(|i| 64 * i).collect::<Vec<_>>(),
        );

        // Full bitmap case. There are only 512 values, so test them all
        let mut map = CodesBitmap::default();
        for i in 0..512 {
            map.set(i);
        }
        assert_eq!(
            map.codes().collect::<Vec<_>>(),
            (0u16..511u16).collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic(expected = "code cannot exceed")]
    fn test_bitmap_invalid() {
        let mut map = CodesBitmap::default();
        map.set(512);
    }

    #[test]
    fn test_no_duplicate_symbols() {
        // Train on data that is likely to produce duplicate 1-byte and 2-byte candidates.
        let text = b"aababcabcdabcde";
        let corpus: Vec<&[u8]> = std::iter::repeat_n(text.as_slice(), 100).collect();
        let compressor = Compressor::train(&corpus);

        let symbols = compressor.symbol_table();
        let lengths = compressor.symbol_lengths();

        // Collect all 1-byte symbols and check for duplicates.
        let one_byte: Vec<u8> = symbols
            .iter()
            .zip(lengths.iter())
            .filter(|&(_, &len)| len == 1)
            .map(|(sym, _)| sym.first_byte())
            .collect();
        let mut one_byte_sorted = one_byte.clone();
        one_byte_sorted.sort();
        one_byte_sorted.dedup();
        assert_eq!(
            one_byte.len(),
            one_byte_sorted.len(),
            "duplicate 1-byte symbols found"
        );

        // Collect all 2-byte symbols and check for duplicates.
        let two_byte: Vec<u16> = symbols
            .iter()
            .zip(lengths.iter())
            .filter(|&(_, &len)| len == 2)
            .map(|(sym, _)| sym.first2())
            .collect();
        let mut two_byte_sorted = two_byte.clone();
        two_byte_sorted.sort();
        two_byte_sorted.dedup();
        assert_eq!(
            two_byte.len(),
            two_byte_sorted.len(),
            "duplicate 2-byte symbols found"
        );
    }
}
