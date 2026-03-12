#![doc = "Pure-Rust implementation of Fast Static Symbol Tables algorithm for string compression"]
#![cfg(target_endian = "little")]

/// Throw a compiler error if a type isn't guaranteed to have a specific size in bytes.
macro_rules! assert_sizeof {
    ($typ:ty => $size_in_bytes:expr) => {
        const _: [u8; $size_in_bytes] = [0; std::mem::size_of::<$typ>()];
    };
}

use std::fmt::Debug;
use std::fmt::Formatter;
use std::mem::MaybeUninit;

use lossy_pht::LossyPHT;

mod builder;
mod lossy_pht;

pub use builder::*;

/// `Symbol`s are small (up to 8-byte) segments of strings, stored in a [`Compressor`][`crate::Compressor`] and
/// identified by an 8-bit code.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Symbol(u64);

assert_sizeof!(Symbol => 8);

impl Symbol {
    /// Zero value for `Symbol`.
    pub const ZERO: Self = Self::zero();

    /// Constructor for a `Symbol` from an 8-element byte slice.
    pub fn from_slice(slice: &[u8; 8]) -> Self {
        let num: u64 = u64::from_le_bytes(*slice);

        Self(num)
    }

    /// Return a zero symbol
    const fn zero() -> Self {
        Self(0)
    }

    /// Create a new single-byte symbol
    pub fn from_u8(value: u8) -> Self {
        Self(value as u64)
    }
}

impl Symbol {
    /// Calculate the length of the symbol in bytes. Always a value between 1 and 8.
    ///
    /// Each symbol has the capacity to hold up to 8 bytes of data, but the symbols
    /// can contain fewer bytes, padded with 0x00. There is a special case of a symbol
    /// that holds the byte 0x00. In that case, the symbol contains `0x0000000000000000`
    /// but we want to interpret that as a one-byte symbol containing `0x00`.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(self) -> usize {
        let numeric = self.0;
        // For little-endian platforms, this counts the number of *trailing* zeros
        let null_bytes = (numeric.leading_zeros() >> 3) as usize;

        // Special case handling of a symbol with all-zeros. This is actually
        // a 1-byte symbol containing 0x00.
        let len = size_of::<Self>() - null_bytes;
        if len == 0 { 1 } else { len }
    }

    /// Returns the Symbol's inner representation.
    #[inline]
    pub fn to_u64(self) -> u64 {
        self.0
    }

    /// Get the first byte of the symbol as a `u8`.
    ///
    /// If the symbol is empty, this will return the zero byte.
    #[inline]
    pub fn first_byte(self) -> u8 {
        self.0 as u8
    }

    /// Get the first two bytes of the symbol as a `u16`.
    ///
    /// If the Symbol is one or zero bytes, this will return `0u16`.
    #[inline]
    pub fn first2(self) -> u16 {
        self.0 as u16
    }

    /// Get the first three bytes of the symbol as a `u64`.
    ///
    /// If the Symbol is one or zero bytes, this will return `0u64`.
    #[inline]
    pub fn first3(self) -> u64 {
        self.0 & 0xFF_FF_FF
    }

    /// Return a new `Symbol` by logically concatenating ourselves with another `Symbol`.
    pub fn concat(self, other: Self) -> Self {
        assert!(
            self.len() + other.len() <= 8,
            "cannot build symbol with length > 8"
        );

        let self_len = self.len();

        Self((other.0 << (8 * self_len)) | self.0)
    }
}

impl Debug for Symbol {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;

        let slice = &self.0.to_le_bytes()[0..self.len()];
        for c in slice.iter().map(|c| *c as char) {
            if ('!'..='~').contains(&c) {
                write!(f, "{c}")?;
            } else if c == '\n' {
                write!(f, " \\n ")?;
            } else if c == '\t' {
                write!(f, " \\t ")?;
            } else if c == ' ' {
                write!(f, " SPACE ")?;
            } else {
                write!(f, " 0x{:X?} ", c as u8)?
            }
        }

        write!(f, "]")
    }
}

/// A packed type containing a code value, as well as metadata about the symbol referred to by
/// the code.
///
/// Logically, codes can range from 0-255 inclusive. This type holds both the 8-bit code as well as
/// other metadata bit-packed into a `u16`.
///
/// The bottom 8 bits contain EITHER a code for a symbol stored in the table, OR a raw byte.
///
/// The interpretation depends on the 9th bit: when toggled off, the value stores a raw byte, and when
/// toggled on, it stores a code. Thus if you examine the bottom 9 bits of the `u16`, you have an extended
/// code range, where the values 0-255 are raw bytes, and the values 256-510 represent codes 0-254. 511 is
/// a placeholder for the invalid code here.
///
/// Bits 12-15 store the length of the symbol (values ranging from 0-8).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Code(u16);

/// Code used to indicate bytes that are not in the symbol table.
///
/// When compressing a string that cannot fully be expressed with the symbol table, the compressed
/// output will contain an `ESCAPE` byte followed by a raw byte. At decompression time, the presence
/// of `ESCAPE` indicates that the next byte should be appended directly to the result instead of
/// being looked up in the symbol table.
pub const ESCAPE_CODE: u8 = 255;

/// Number of bits in the `ExtendedCode` that are used to dictate a code value.
pub const FSST_CODE_BITS: usize = 9;

/// First bit of the "length" portion of an extended code.
pub const FSST_LEN_BITS: usize = 12;

/// Maximum code value in the extended code range.
pub const FSST_CODE_MAX: u16 = 1 << FSST_CODE_BITS;

/// Maximum value for the extended code range.
///
/// When truncated to u8 this is code 255, which is equivalent to [`ESCAPE_CODE`].
pub const FSST_CODE_MASK: u16 = FSST_CODE_MAX - 1;

/// First code in the symbol table that corresponds to a non-escape symbol.
pub const FSST_CODE_BASE: u16 = 256;

#[allow(clippy::len_without_is_empty)]
impl Code {
    /// Code for an unused slot in a symbol table or index.
    ///
    /// This corresponds to the maximum code with a length of 1.
    pub const UNUSED: Self = Code(FSST_CODE_MASK + (1 << 12));

    /// Create a new code for a symbol of given length.
    fn new_symbol(code: u8, len: usize) -> Self {
        Self(code as u16 + ((len as u16) << FSST_LEN_BITS))
    }

    /// Code for a new symbol during the building phase.
    ///
    /// The code is remapped from 0..254 to 256...510.
    fn new_symbol_building(code: u8, len: usize) -> Self {
        Self(code as u16 + 256 + ((len as u16) << FSST_LEN_BITS))
    }

    /// Create a new code corresponding for an escaped byte.
    fn new_escape(byte: u8) -> Self {
        Self((byte as u16) + (1 << FSST_LEN_BITS))
    }

    #[inline]
    fn code(self) -> u8 {
        self.0 as u8
    }

    #[inline]
    fn extended_code(self) -> u16 {
        self.0 & 0b111_111_111
    }

    #[inline]
    fn len(self) -> u16 {
        self.0 >> FSST_LEN_BITS
    }
}

impl Debug for Code {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainingCode")
            .field("code", &(self.0 as u8))
            .field("is_escape", &(self.0 < 256))
            .field("len", &(self.0 >> 12))
            .finish()
    }
}

/// Decompressor uses a symbol table to take a stream of 8-bit codes into a string.
#[derive(Clone)]
pub struct Decompressor<'a> {
    /// Slice mapping codes to symbols.
    pub(crate) symbols: &'a [Symbol],

    /// Slice containing the length of each symbol in the `symbols` slice.
    pub(crate) lengths: &'a [u8],
}

impl<'a> Decompressor<'a> {
    /// Returns a new decompressor that uses the provided symbol table.
    ///
    /// # Panics
    ///
    /// If the provided symbol table has length greater than 256
    pub fn new(symbols: &'a [Symbol], lengths: &'a [u8]) -> Self {
        assert!(
            symbols.len() < FSST_CODE_BASE as usize,
            "symbol table cannot have size exceeding 255"
        );

        Self { symbols, lengths }
    }

    /// Returns an upper bound on the size of the decompressed data.
    pub fn max_decompression_capacity(&self, compressed: &[u8]) -> usize {
        size_of::<Symbol>() * (compressed.len() + 1)
    }

    /// Decompress a slice of codes into a provided buffer.
    ///
    /// The provided `decoded` buffer must be at least the size of the decoded data, plus
    /// an additional 7 bytes.
    ///
    /// ## Panics
    ///
    /// If the caller fails to provide sufficient capacity in the decoded buffer. An upper bound
    /// on the required capacity can be obtained by calling [`Self::max_decompression_capacity`].
    ///
    /// ## Example
    ///
    /// ```
    /// use fsst::{Symbol, Compressor, CompressorBuilder};
    /// let compressor = {
    ///     let mut builder = CompressorBuilder::new();
    ///     builder.insert(Symbol::from_slice(&[b'h', b'e', b'l', b'l', b'o', b'o', b'o', b'o']), 8);
    ///     builder.build()
    /// };
    ///
    /// let decompressor = compressor.decompressor();
    ///
    /// let mut decompressed = Vec::with_capacity(8 + 7);
    ///
    /// let len = decompressor.decompress_into(&[0], decompressed.spare_capacity_mut());
    /// assert_eq!(len, 8);
    /// unsafe { decompressed.set_len(len) };
    /// assert_eq!(&decompressed, "helloooo".as_bytes());
    /// ```
    pub fn decompress_into(&self, compressed: &[u8], decoded: &mut [MaybeUninit<u8>]) -> usize {
        // Ensure the target buffer is at least half the size of the input buffer.
        // This is the theortical smallest a valid target can be, and occurs when
        // every input code is an escape.
        assert!(
            decoded.len() >= compressed.len() / 2,
            "decoded is smaller than lower-bound decompressed size"
        );

        unsafe {
            let mut in_ptr = compressed.as_ptr();
            let _in_begin = in_ptr;
            let in_end = in_ptr.add(compressed.len());

            let mut out_ptr: *mut u8 = decoded.as_mut_ptr().cast();
            let out_begin = out_ptr.cast_const();
            let out_end = decoded.as_ptr().add(decoded.len()).cast::<u8>();

            macro_rules! store_next_symbol {
                ($code:expr) => {{
                    out_ptr
                        .cast::<u64>()
                        .write_unaligned(self.symbols.get_unchecked($code as usize).to_u64());
                    out_ptr = out_ptr.add(*self.lengths.get_unchecked($code as usize) as usize);
                }};
            }

            // First we try loading 8 bytes at a time.
            if decoded.len() >= 8 * size_of::<Symbol>() && compressed.len() >= 8 {
                // Extract the loop condition since the compiler fails to do so
                let block_out_end = out_end.sub(8 * size_of::<Symbol>());
                let block_in_end = in_end.sub(8);

                while out_ptr.cast_const() <= block_out_end && in_ptr < block_in_end {
                    // Note that we load a little-endian u64 here.
                    let next_block = in_ptr.cast::<u64>().read_unaligned();
                    let escape_mask = (next_block & 0x8080808080808080)
                        & ((((!next_block) & 0x7F7F7F7F7F7F7F7F) + 0x7F7F7F7F7F7F7F7F)
                            ^ 0x8080808080808080);

                    // If there are no escape codes, we write each symbol one by one.
                    if escape_mask == 0 {
                        let code = (next_block & 0xFF) as u8;
                        store_next_symbol!(code);
                        let code = ((next_block >> 8) & 0xFF) as u8;
                        store_next_symbol!(code);
                        let code = ((next_block >> 16) & 0xFF) as u8;
                        store_next_symbol!(code);
                        let code = ((next_block >> 24) & 0xFF) as u8;
                        store_next_symbol!(code);
                        let code = ((next_block >> 32) & 0xFF) as u8;
                        store_next_symbol!(code);
                        let code = ((next_block >> 40) & 0xFF) as u8;
                        store_next_symbol!(code);
                        let code = ((next_block >> 48) & 0xFF) as u8;
                        store_next_symbol!(code);
                        let code = ((next_block >> 56) & 0xFF) as u8;
                        store_next_symbol!(code);
                        in_ptr = in_ptr.add(8);
                    } else {
                        // Otherwise, find the first escape code and write the symbols up to that point.
                        let first_escape_pos = escape_mask.trailing_zeros() >> 3; // Divide bits to bytes
                        debug_assert!(first_escape_pos < 8);
                        match first_escape_pos {
                            7 => {
                                let code = (next_block & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 8) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 16) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 24) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 32) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 40) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 48) & 0xFF) as u8;
                                store_next_symbol!(code);

                                in_ptr = in_ptr.add(7);
                            }
                            6 => {
                                let code = (next_block & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 8) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 16) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 24) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 32) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 40) & 0xFF) as u8;
                                store_next_symbol!(code);

                                let escaped = ((next_block >> 56) & 0xFF) as u8;
                                out_ptr.write(escaped);
                                out_ptr = out_ptr.add(1);

                                in_ptr = in_ptr.add(8);
                            }
                            5 => {
                                let code = (next_block & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 8) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 16) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 24) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 32) & 0xFF) as u8;
                                store_next_symbol!(code);

                                let escaped = ((next_block >> 48) & 0xFF) as u8;
                                out_ptr.write(escaped);
                                out_ptr = out_ptr.add(1);

                                in_ptr = in_ptr.add(7);
                            }
                            4 => {
                                let code = (next_block & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 8) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 16) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 24) & 0xFF) as u8;
                                store_next_symbol!(code);

                                let escaped = ((next_block >> 40) & 0xFF) as u8;
                                out_ptr.write(escaped);
                                out_ptr = out_ptr.add(1);

                                in_ptr = in_ptr.add(6);
                            }
                            3 => {
                                let code = (next_block & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 8) & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 16) & 0xFF) as u8;
                                store_next_symbol!(code);

                                let escaped = ((next_block >> 32) & 0xFF) as u8;
                                out_ptr.write(escaped);
                                out_ptr = out_ptr.add(1);

                                in_ptr = in_ptr.add(5);
                            }
                            2 => {
                                let code = (next_block & 0xFF) as u8;
                                store_next_symbol!(code);
                                let code = ((next_block >> 8) & 0xFF) as u8;
                                store_next_symbol!(code);

                                let escaped = ((next_block >> 24) & 0xFF) as u8;
                                out_ptr.write(escaped);
                                out_ptr = out_ptr.add(1);

                                in_ptr = in_ptr.add(4);
                            }
                            1 => {
                                let code = (next_block & 0xFF) as u8;
                                store_next_symbol!(code);

                                let escaped = ((next_block >> 16) & 0xFF) as u8;
                                out_ptr.write(escaped);
                                out_ptr = out_ptr.add(1);

                                in_ptr = in_ptr.add(3);
                            }
                            0 => {
                                // Otherwise, we actually need to decompress the next byte
                                // Extract the second byte from the u32
                                let escaped = ((next_block >> 8) & 0xFF) as u8;
                                in_ptr = in_ptr.add(2);
                                out_ptr.write(escaped);
                                out_ptr = out_ptr.add(1);
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }

            // Otherwise, fall back to 1-byte reads.
            while out_end.offset_from(out_ptr) > size_of::<Symbol>() as isize && in_ptr < in_end {
                let code = in_ptr.read();
                in_ptr = in_ptr.add(1);

                if code == ESCAPE_CODE {
                    out_ptr.write(in_ptr.read());
                    in_ptr = in_ptr.add(1);
                    out_ptr = out_ptr.add(1);
                } else {
                    store_next_symbol!(code);
                }
            }

            assert_eq!(
                in_ptr, in_end,
                "decompression should exhaust input before output"
            );

            out_ptr.offset_from(out_begin) as usize
        }
    }

    /// Decompress a byte slice that was previously returned by a compressor using the same symbol
    /// table into a new vector of bytes.
    pub fn decompress(&self, compressed: &[u8]) -> Vec<u8> {
        let mut decoded = Vec::with_capacity(self.max_decompression_capacity(compressed) + 7);

        let len = self.decompress_into(compressed, decoded.spare_capacity_mut());
        // SAFETY: len bytes have now been initialized by the decompressor.
        unsafe { decoded.set_len(len) };
        decoded
    }
}

/// A compressor that uses a symbol table to greedily compress strings.
///
/// The `Compressor` is the central component of FSST. You can create a compressor either by
/// default (i.e. an empty compressor), or by [training][`Self::train`] it on an input corpus of text.
///
/// Example usage:
///
/// ```
/// use fsst::{Symbol, Compressor, CompressorBuilder};
/// let compressor = {
///     let mut builder = CompressorBuilder::new();
///     builder.insert(Symbol::from_slice(&[b'h', b'e', b'l', b'l', b'o', 0, 0, 0]), 5);
///     builder.build()
/// };
///
/// let compressed = compressor.compress("hello".as_bytes());
/// assert_eq!(compressed, vec![0u8]);
/// ```
#[derive(Clone)]
pub struct Compressor {
    /// Table mapping codes to symbols.
    pub(crate) symbols: Vec<Symbol>,

    /// Length of each symbol, values range from 1-8.
    pub(crate) lengths: Vec<u8>,

    /// The number of entries in the symbol table that have been populated, not counting
    /// the escape values.
    pub(crate) n_symbols: u8,

    /// Inverted index mapping 2-byte symbols to codes
    codes_two_byte: Vec<Code>,

    /// Limit of no suffixes.
    has_suffix_code: u8,

    /// Lossy perfect hash table for looking up codes to symbols that are 3 bytes or more
    lossy_pht: LossyPHT,
}

/// The core structure of the FSST codec, holding a mapping between `Symbol`s and `Code`s.
///
/// The symbol table is trained on a corpus of data in the form of a single byte array, building up
/// a mapping of 1-byte "codes" to sequences of up to 8 plaintext bytes, or "symbols".
impl Compressor {
    /// Using the symbol table, runs a single cycle of compression on an input word, writing
    /// the output into `out_ptr`.
    ///
    /// # Returns
    ///
    /// This function returns a tuple of (advance_in, advance_out) with the number of bytes
    /// for the caller to advance the input and output pointers.
    ///
    /// `advance_in` is the number of bytes to advance the input pointer before the next call.
    ///
    /// `advance_out` is the number of bytes to advance `out_ptr` before the next call.
    ///
    /// # Safety
    ///
    /// `out_ptr` must never be NULL or otherwise point to invalid memory.
    pub unsafe fn compress_word(&self, word: u64, out_ptr: *mut u8) -> (usize, usize) {
        // Speculatively write the first byte of `word` at offset 1. This is necessary if it is an escape, and
        // if it isn't, it will be overwritten anyway.
        //
        // SAFETY: caller ensures out_ptr is not null
        let first_byte = word as u8;
        // SAFETY: out_ptr is not null
        unsafe { out_ptr.byte_add(1).write_unaligned(first_byte) };

        // First, check the two_bytes table
        let code_twobyte = self.codes_two_byte[word as u16 as usize];

        if code_twobyte.code() < self.has_suffix_code {
            // 2 byte code without having to worry about longer matches.
            // SAFETY: out_ptr is not null.
            unsafe { std::ptr::write(out_ptr, code_twobyte.code()) };

            // Advance input by symbol length (2) and output by a single code byte
            (2, 1)
        } else {
            // Probe the hash table
            let entry = self.lossy_pht.lookup(word);

            // Now, downshift the `word` and the `entry` to see if they align.
            let ignored_bits = entry.ignored_bits;
            if entry.code != Code::UNUSED
                && compare_masked(word, entry.symbol.to_u64(), ignored_bits)
            {
                // Advance the input by the symbol length (variable) and the output by one code byte
                // SAFETY: out_ptr is not null.
                unsafe { std::ptr::write(out_ptr, entry.code.code()) };
                (entry.code.len() as usize, 1)
            } else {
                // SAFETY: out_ptr is not null
                unsafe { std::ptr::write(out_ptr, code_twobyte.code()) };

                // Advance the input by the symbol length (variable) and the output by either 1
                // byte (if was one-byte code) or two bytes (escape).
                (
                    code_twobyte.len() as usize,
                    // Predicated version of:
                    //
                    // if entry.code >= 256 {
                    //      2
                    // } else {
                    //      1
                    // }
                    1 + (code_twobyte.extended_code() >> 8) as usize,
                )
            }
        }
    }

    /// Compress many lines in bulk.
    pub fn compress_bulk(&self, lines: &Vec<&[u8]>) -> Vec<Vec<u8>> {
        let mut res = Vec::new();

        for line in lines {
            res.push(self.compress(line));
        }

        res
    }

    /// Compress a string, writing its result into a target buffer.
    ///
    /// The target buffer is a byte vector that must have capacity large enough
    /// to hold the encoded data.
    ///
    /// When this call returns, `values` will hold the compressed bytes and have
    /// its length set to the length of the compressed text.
    ///
    /// ```
    /// use fsst::{Compressor, CompressorBuilder, Symbol};
    ///
    /// let mut compressor = CompressorBuilder::new();
    /// assert!(compressor.insert(Symbol::from_slice(b"aaaaaaaa"), 8));
    ///
    /// let compressor = compressor.build();
    ///
    /// let mut compressed_values = Vec::with_capacity(1_024);
    ///
    /// // SAFETY: we have over-sized compressed_values.
    /// unsafe {
    ///     compressor.compress_into(b"aaaaaaaa", &mut compressed_values);
    /// }
    ///
    /// assert_eq!(compressed_values, vec![0u8]);
    /// ```
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure the provided buffer is large enough to hold
    /// all encoded data.
    pub unsafe fn compress_into(&self, plaintext: &[u8], values: &mut Vec<u8>) {
        let mut in_ptr = plaintext.as_ptr();
        let mut out_ptr = values.as_mut_ptr();

        // SAFETY: `end` will point just after the end of the `plaintext` slice.
        let in_end = unsafe { in_ptr.byte_add(plaintext.len()) };
        let in_end_sub8 = in_end as usize - 8;
        // SAFETY: `end` will point just after the end of the `values` allocation.
        let out_end = unsafe { out_ptr.byte_add(values.capacity()) };

        while (in_ptr as usize) <= in_end_sub8 && out_ptr < out_end {
            // SAFETY: pointer ranges are checked in the loop condition
            unsafe {
                // Load a full 8-byte word of data from in_ptr.
                // SAFETY: caller asserts in_ptr is not null. we may read past end of pointer though.
                let word: u64 = std::ptr::read_unaligned(in_ptr as *const u64);
                let (advance_in, advance_out) = self.compress_word(word, out_ptr);
                in_ptr = in_ptr.byte_add(advance_in);
                out_ptr = out_ptr.byte_add(advance_out);
            };
        }

        let remaining_bytes = unsafe { in_end.byte_offset_from(in_ptr) };
        assert!(
            out_ptr < out_end || remaining_bytes == 0,
            "output buffer sized too small"
        );

        let remaining_bytes = remaining_bytes as usize;

        // Load the last `remaining_byte`s of data into a final world. We then replicate the loop above,
        // but shift data out of this word rather than advancing an input pointer and potentially reading
        // unowned memory.
        let mut bytes = [0u8; 8];
        // SAFETY: remaining_bytes <= 8
        unsafe { std::ptr::copy_nonoverlapping(in_ptr, bytes.as_mut_ptr(), remaining_bytes) };
        let mut last_word = u64::from_le_bytes(bytes);

        while in_ptr < in_end && out_ptr < out_end {
            // Load a full 8-byte word of data from in_ptr.
            // SAFETY: caller asserts in_ptr is not null
            let (advance_in, advance_out) = unsafe { self.compress_word(last_word, out_ptr) };
            // SAFETY: pointer ranges are checked in the loop condition
            unsafe {
                in_ptr = in_ptr.add(advance_in);
                out_ptr = out_ptr.add(advance_out);
            }

            last_word = advance_8byte_word(last_word, advance_in);
        }

        // in_ptr should have exceeded in_end
        assert!(
            in_ptr >= in_end,
            "exhausted output buffer before exhausting input, there is a bug in SymbolTable::compress()"
        );

        assert!(out_ptr <= out_end, "output buffer sized too small");

        // SAFETY: out_ptr is derived from the `values` allocation.
        let bytes_written = unsafe { out_ptr.offset_from(values.as_ptr()) };
        assert!(
            bytes_written >= 0,
            "out_ptr ended before it started, not possible"
        );

        // SAFETY: we have initialized `bytes_written` values in the output buffer.
        unsafe { values.set_len(bytes_written as usize) };
    }

    /// Use the symbol table to compress the plaintext into a sequence of codes and escapes.
    pub fn compress(&self, plaintext: &[u8]) -> Vec<u8> {
        if plaintext.is_empty() {
            return Vec::new();
        }

        let mut buffer = Vec::with_capacity(plaintext.len() * 2);

        // SAFETY: the largest compressed size would be all escapes == 2*plaintext_len
        unsafe { self.compress_into(plaintext, &mut buffer) };

        buffer
    }

    /// Access the decompressor that can be used to decompress strings emitted from this
    /// `Compressor` instance.
    pub fn decompressor(&self) -> Decompressor<'_> {
        Decompressor::new(self.symbol_table(), self.symbol_lengths())
    }

    /// Returns a readonly slice of the current symbol table.
    ///
    /// The returned slice will have length of `n_symbols`.
    pub fn symbol_table(&self) -> &[Symbol] {
        &self.symbols[0..self.n_symbols as usize]
    }

    /// Returns a readonly slice where index `i` contains the
    /// length of the symbol represented by code `i`.
    ///
    /// Values range from 1-8.
    pub fn symbol_lengths(&self) -> &[u8] {
        &self.lengths[0..self.n_symbols as usize]
    }

    /// Rebuild a compressor from an existing symbol table.
    ///
    /// This will not attempt to optimize or re-order the codes.
    pub fn rebuild_from(symbols: impl AsRef<[Symbol]>, symbol_lens: impl AsRef<[u8]>) -> Self {
        let symbols = symbols.as_ref();
        let symbol_lens = symbol_lens.as_ref();

        assert_eq!(
            symbols.len(),
            symbol_lens.len(),
            "symbols and lengths differ"
        );
        assert!(
            symbols.len() <= 255,
            "symbol table len must be <= 255, was {}",
            symbols.len()
        );
        validate_symbol_order(symbol_lens);

        // Insert the symbols in their given order into the FSST lookup structures.
        let symbols = symbols.to_vec();
        let lengths = symbol_lens.to_vec();
        let mut lossy_pht = LossyPHT::new();

        let mut codes_one_byte = vec![Code::UNUSED; 256];

        // Insert all of the one byte symbols first.
        for (code, (&symbol, &len)) in symbols.iter().zip(lengths.iter()).enumerate() {
            if len == 1 {
                codes_one_byte[symbol.first_byte() as usize] = Code::new_symbol(code as u8, 1);
            }
        }

        // Initialize the codes_two_byte table to be all escapes
        let mut codes_two_byte = vec![Code::UNUSED; 65_536];

        // Insert the two byte symbols, possibly overwriting slots for one-byte symbols and escapes.
        for (code, (&symbol, &len)) in symbols.iter().zip(lengths.iter()).enumerate() {
            match len {
                2 => {
                    codes_two_byte[symbol.first2() as usize] = Code::new_symbol(code as u8, 2);
                }
                3.. => {
                    assert!(
                        lossy_pht.insert(symbol, len as usize, code as u8),
                        "rebuild symbol insertion into PHT must succeed"
                    );
                }
                _ => { /* Covered by the 1-byte loop above. */ }
            }
        }

        // Build the finished codes_two_byte table, subbing in unused positions with the
        // codes_one_byte value similar to what we do in CompressBuilder::finalize.
        for (symbol, code) in codes_two_byte.iter_mut().enumerate() {
            if *code == Code::UNUSED {
                *code = codes_one_byte[symbol & 0xFF];
            }
        }

        // Find the position of the first 2-byte code that has a suffix later in the table
        let mut has_suffix_code = 0u8;
        for (code, (&symbol, &len)) in symbols.iter().zip(lengths.iter()).enumerate() {
            if len != 2 {
                break;
            }
            let rest = &symbols[code..];
            if rest
                .iter()
                .any(|&other| other.len() > 2 && symbol.first2() == other.first2())
            {
                has_suffix_code = code as u8;
                break;
            }
        }

        Compressor {
            n_symbols: symbols.len() as u8,
            symbols,
            lengths,
            codes_two_byte,
            lossy_pht,
            has_suffix_code,
        }
    }
}

#[inline]
pub(crate) fn advance_8byte_word(word: u64, bytes: usize) -> u64 {
    // shift the word off the low-end, because little endian means the first
    // char is stored in the LSB.
    //
    // Note that even though this looks like it branches, Rust compiles this to a
    // conditional move instruction. See `<https://godbolt.org/z/Pbvre65Pq>`
    if bytes == 8 { 0 } else { word >> (8 * bytes) }
}

fn validate_symbol_order(symbol_lens: &[u8]) {
    // Ensure that the symbol table is ordered by length, 23456781
    let mut expected = 2;
    for (idx, &len) in symbol_lens.iter().enumerate() {
        if expected == 1 {
            assert_eq!(
                len, 1,
                "symbol code={idx} should be one byte, was {len} bytes"
            );
        } else {
            if len == 1 {
                expected = 1;
            }

            // we're in the non-zero portion.
            assert!(
                len >= expected,
                "symbol code={idx} breaks violates FSST symbol table ordering"
            );
            expected = len;
        }
    }
}

#[inline]
pub(crate) fn compare_masked(left: u64, right: u64, ignored_bits: u16) -> bool {
    let mask = u64::MAX >> ignored_bits;
    (left & mask) == right
}

#[cfg(test)]
mod test {
    use std::iter;
    use std::mem;

    use super::*;
    #[test]
    fn test_stuff() {
        let compressor = {
            let mut builder = CompressorBuilder::new();
            builder.insert(Symbol::from_slice(b"helloooo"), 8);
            builder.build()
        };

        let decompressor = compressor.decompressor();

        let mut decompressed = Vec::with_capacity(8 + 7);

        let len = decompressor.decompress_into(&[0], decompressed.spare_capacity_mut());
        assert_eq!(len, 8);
        unsafe { decompressed.set_len(len) };
        assert_eq!(&decompressed, "helloooo".as_bytes());
    }

    #[test]
    fn test_symbols_good() {
        let symbols_u64: &[u64] = &[
            24931, 25698, 25442, 25699, 25186, 25444, 24932, 25188, 25185, 25441, 25697, 25700,
            24929, 24930, 25443, 25187, 6513249, 6512995, 6578786, 6513761, 6513507, 6382434,
            6579042, 6512994, 6447460, 6447969, 6382178, 6579041, 6512993, 6448226, 6513250,
            6579297, 6513506, 6447459, 6513764, 6447458, 6578529, 6382180, 6513762, 6447714,
            6579299, 6513508, 6382436, 6513763, 6578532, 6381924, 6448228, 6579300, 6381921,
            6382690, 6382179, 6447713, 6447972, 6513505, 6447457, 6382692, 6513252, 6578785,
            6578787, 6578531, 6448225, 6382177, 6382433, 6578530, 6448227, 6381922, 6578788,
            6579044, 6382691, 6512996, 6579043, 6579298, 6447970, 6447716, 6447971, 6381923,
            6447715, 97, 98, 100, 99, 97, 98, 99, 100,
        ];
        let symbols: &[Symbol] = unsafe { mem::transmute(symbols_u64) };
        let lens: Vec<u8> = iter::repeat_n(2u8, 16)
            .chain(iter::repeat_n(3u8, 61))
            .chain(iter::repeat_n(1u8, 8))
            .collect();

        let compressor = Compressor::rebuild_from(symbols, lens);
        let built_symbols: &[u64] = unsafe { mem::transmute(compressor.symbol_table()) };
        assert_eq!(built_symbols, symbols_u64);
    }

    #[should_panic(expected = "assertion `left == right` failed")]
    #[test]
    fn test_symbols_bad() {
        let symbols: &[u64] = &[
            24931, 25698, 25442, 25699, 25186, 25444, 24932, 25188, 25185, 25441, 25697, 25700,
            24929, 24930, 25443, 25187, 6513249, 6512995, 6578786, 6513761, 6513507, 6382434,
            6579042, 6512994, 6447460, 6447969, 6382178, 6579041, 6512993, 6448226, 6513250,
            6579297, 6513506, 6447459, 6513764, 6447458, 6578529, 6382180, 6513762, 6447714,
            6579299, 6513508, 6382436, 6513763, 6578532, 6381924, 6448228, 6579300, 6381921,
            6382690, 6382179, 6447713, 6447972, 6513505, 6447457, 6382692, 6513252, 6578785,
            6578787, 6578531, 6448225, 6382177, 6382433, 6578530, 6448227, 6381922, 6578788,
            6579044, 6382691, 6512996, 6579043, 6579298, 6447970, 6447716, 6447971, 6381923,
            6447715, 97, 98, 100, 99, 97, 98, 99, 100,
        ];
        let lens: Vec<u8> = iter::repeat_n(2u8, 16)
            .chain(iter::repeat_n(3u8, 61))
            .chain(iter::repeat_n(1u8, 8))
            .collect();

        let mut builder = CompressorBuilder::new();
        for (symbol, len) in symbols.iter().zip(lens.iter()) {
            let symbol = Symbol::from_slice(&symbol.to_le_bytes());
            builder.insert(symbol, *len as usize);
        }
        let compressor = builder.build();
        let built_symbols: &[u64] = unsafe { mem::transmute(compressor.symbol_table()) };
        assert_eq!(built_symbols, symbols);
    }
}
