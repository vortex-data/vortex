// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnsMetadata;
use crate::tans::Encoder;
use crate::tans::MAX_ANS_SIZE_LOG;
use crate::tans::MIN_ANS_SIZE_LOG;
use crate::tans::Symbol;
use crate::tans::Weight;
use crate::tans::decode_symbols;
use crate::tans::encode_symbols;
use crate::tans::quantize_weights;

/// A tANS-encoded Vortex array of `u8` symbols.
///
/// Stores a small-alphabet `Primitive<u8>` stream compressed with
/// single-state tANS. The compressed byte stream lives in buffer 0;
/// the alphabet (a sparse subset of `0..=255`) and per-symbol weights
/// live in the metadata.
///
/// # Random access
///
/// `scalar_at(i)` is **O(N)** for this array: tANS decode is inherently
/// sequential, so element-level random access requires a full decode.
/// Adding batch checkpoints to amortize is a P6 concern (see
/// `encodings/pco/DESIGN.md`). For the standalone array, `execute`
/// is the supported access pattern.
pub type AnsArray = Array<Ans>;

const NUM_SLOTS: usize = 0;
const NUM_BUFFERS: usize = 1;
const ENCODED_BUFFER_NAME: &str = "encoded";

/// Marker type implementing [`VTable`] for [`Ans`].
#[derive(Clone, Debug)]
pub struct Ans;

/// Per-array data for [`AnsArray`].
///
/// `alphabet` maps a dense symbol id (`0..alphabet.len()`) to the
/// original `u8` value emitted at decode time. `weights` is the
/// frequency-derived weight table whose sum equals `1 << ans_size_log`.
#[derive(Clone, Debug)]
pub struct AnsData {
    /// tANS table size log; `state ∈ [1 << ans_size_log, 2 << ans_size_log)`.
    ans_size_log: u8,
    /// Number of symbols emitted at decode time.
    n_symbols: u64,
    /// Distinct symbols in the order they entered the alphabet.
    alphabet: Vec<u8>,
    /// Per-symbol weight; `weights.len() == alphabet.len()`; sum equals
    /// `1 << ans_size_log` (or `1` when `alphabet.len() <= 1`).
    weights: Vec<Weight>,
    /// Final tANS state after encoding (kept in `[table_size, 2*table_size)`).
    final_state: u32,
    /// Number of bits the bit-writer actually consumed.
    bit_len: u64,
    /// The compressed bit stream (LSB-first; bytes round up).
    encoded: ByteBuffer,
}

impl AnsData {
    pub(crate) fn new(
        ans_size_log: u8,
        n_symbols: u64,
        alphabet: Vec<u8>,
        weights: Vec<Weight>,
        final_state: u32,
        bit_len: u64,
        encoded: ByteBuffer,
    ) -> Self {
        Self {
            ans_size_log,
            n_symbols,
            alphabet,
            weights,
            final_state,
            bit_len,
            encoded,
        }
    }

    /// tANS table size log.
    pub fn ans_size_log(&self) -> u8 {
        self.ans_size_log
    }

    /// Number of symbols stored.
    pub fn n_symbols(&self) -> u64 {
        self.n_symbols
    }

    /// Dense -> u8 alphabet.
    pub fn alphabet(&self) -> &[u8] {
        &self.alphabet
    }

    /// Per-symbol weight table (sum is `1 << ans_size_log`).
    pub fn weights(&self) -> &[Weight] {
        &self.weights
    }

    /// Final tANS state after encoding.
    pub fn final_state(&self) -> u32 {
        self.final_state
    }

    /// Bit length of the encoded stream.
    pub fn bit_len(&self) -> u64 {
        self.bit_len
    }

    /// Encoded byte buffer.
    pub fn encoded(&self) -> &ByteBuffer {
        &self.encoded
    }
}

impl Display for AnsData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n_symbols: {}, alphabet_size: {}, ans_size_log: {}, encoded_bytes: {}",
            self.n_symbols,
            self.alphabet.len(),
            self.ans_size_log,
            self.encoded.len(),
        )
    }
}

impl ArrayHash for AnsData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.ans_size_log.hash(state);
        self.n_symbols.hash(state);
        self.alphabet.hash(state);
        self.weights.hash(state);
        self.final_state.hash(state);
        self.bit_len.hash(state);
        self.encoded.as_slice().hash(state);
    }
}

impl ArrayEq for AnsData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.ans_size_log == other.ans_size_log
            && self.n_symbols == other.n_symbols
            && self.alphabet == other.alphabet
            && self.weights == other.weights
            && self.final_state == other.final_state
            && self.bit_len == other.bit_len
            && self.encoded.as_slice() == other.encoded.as_slice()
    }
}

impl VTable for Ans {
    type TypedArrayData = AnsData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.ans");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        validate_parts(data, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        NUM_BUFFERS
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.data().encoded.clone()),
            _ => vortex_panic!("AnsArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some(ENCODED_BUFFER_NAME.to_string()),
            _ => vortex_panic!("AnsArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let data = array.data();
        Ok(Some(
            AnsMetadata {
                ans_size_log: data.ans_size_log as u32,
                n_symbols: data.n_symbols,
                alphabet: data.alphabet.iter().map(|b| *b as u32).collect(),
                weights: data.weights.clone(),
                final_state: data.final_state,
                bit_len: data.bit_len,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = AnsMetadata::decode(metadata)?;
        if children.len() != NUM_SLOTS {
            vortex_bail!("Expected {NUM_SLOTS} children, got {}", children.len());
        }
        if buffers.len() != NUM_BUFFERS {
            vortex_bail!("Expected {NUM_BUFFERS} buffers, got {}", buffers.len());
        }
        ensure_u8_dtype(dtype)?;
        let ans_size_log = decode_size_log(metadata.ans_size_log)?;
        let alphabet = decode_alphabet(&metadata.alphabet)?;
        let weights = metadata.weights.clone();
        let encoded = buffers[0].clone().try_to_host_sync()?;
        let data = AnsData::new(
            ans_size_log,
            metadata.n_symbols,
            alphabet,
            weights,
            metadata.final_state,
            metadata.bit_len,
            encoded,
        );
        validate_parts(&data, dtype, len)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("AnsArray slot_name index {idx} out of bounds")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let _ = ctx;
        let data = array.data();
        let values = decode_all(data)?;
        Ok(ExecutionResult::done(
            PrimitiveArray::new(Buffer::from(values), Validity::NonNullable).into_array(),
        ))
    }
}

impl ValidityVTable<Ans> for Ans {
    fn validity(_array: ArrayView<'_, Ans>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

/// Decode the full stream to a `Vec<u8>` in input order.
fn decode_all(data: &AnsData) -> VortexResult<Vec<u8>> {
    if data.n_symbols == 0 {
        return Ok(Vec::new());
    }
    if data.alphabet.len() <= 1 {
        // Degenerate: a single distinct symbol, so the stream is a
        // run of that byte. No bits are stored.
        let byte = *data.alphabet.first().ok_or_else(|| {
            vortex_error::vortex_err!("AnsArray with zero alphabet but nonzero len")
        })?;
        return Ok(vec![byte; data.n_symbols as usize]);
    }
    let symbols = decode_symbols(
        data.encoded.as_slice(),
        data.n_symbols,
        data.final_state,
        data.ans_size_log,
        &data.weights,
    )?;
    let mut out = Vec::with_capacity(symbols.len());
    for s in symbols {
        let idx = s as usize;
        if idx >= data.alphabet.len() {
            vortex_bail!(
                "AnsArray decoded symbol {idx} out of alphabet (size {})",
                data.alphabet.len(),
            );
        }
        out.push(data.alphabet[idx]);
    }
    Ok(out)
}

fn ensure_u8_dtype(dtype: &DType) -> VortexResult<()> {
    let ptype = PType::try_from(dtype)?;
    if ptype != PType::U8 {
        vortex_bail!("AnsArray only supports u8 in this phase, got {ptype}");
    }
    if dtype.is_nullable() {
        vortex_bail!("AnsArray is non-nullable in this phase");
    }
    Ok(())
}

fn decode_size_log(raw: u32) -> VortexResult<u8> {
    // size_log of 0 is valid for a single-symbol alphabet (degenerate).
    if raw > MAX_ANS_SIZE_LOG as u32 {
        vortex_bail!(
            "AnsArray ans_size_log {raw} exceeds max {}",
            MAX_ANS_SIZE_LOG,
        );
    }
    Ok(raw as u8)
}

fn decode_alphabet(raw: &[u32]) -> VortexResult<Vec<u8>> {
    if raw.len() > 256 {
        vortex_bail!("AnsArray alphabet has {} entries, exceeds 256", raw.len());
    }
    let mut out = Vec::with_capacity(raw.len());
    for &v in raw {
        if v > u8::MAX as u32 {
            vortex_bail!("AnsArray alphabet entry {v} does not fit in u8");
        }
        out.push(v as u8);
    }
    Ok(out)
}

fn validate_parts(data: &AnsData, dtype: &DType, len: usize) -> VortexResult<()> {
    ensure_u8_dtype(dtype)?;
    vortex_ensure!(
        usize::try_from(data.n_symbols)
            .map(|n| n == len)
            .unwrap_or(false),
        "AnsArray n_symbols {} does not match array len {len}",
        data.n_symbols,
    );
    vortex_ensure!(
        data.weights.len() == data.alphabet.len(),
        "AnsArray weights len {} != alphabet len {}",
        data.weights.len(),
        data.alphabet.len(),
    );
    if data.alphabet.is_empty() {
        vortex_ensure!(
            data.n_symbols == 0,
            "AnsArray empty alphabet but n_symbols={}",
            data.n_symbols,
        );
    } else if data.alphabet.len() == 1 {
        // No tANS table is built; nothing to validate beyond alphabet.
    } else {
        let expected_sum: Weight = 1u32 << data.ans_size_log;
        let actual_sum: Weight = data.weights.iter().copied().sum();
        vortex_ensure!(
            actual_sum == expected_sum,
            "AnsArray weight sum {} != 1 << ans_size_log = {}",
            actual_sum,
            expected_sum,
        );
        for &w in &data.weights {
            vortex_ensure!(w > 0, "AnsArray weights must all be > 0");
        }
    }
    Ok(())
}

/// Extension methods on any typed reference to an [`AnsArray`].
pub trait AnsArrayExt: TypedArrayRef<Ans> {
    /// tANS table size log.
    fn ans_size_log(&self) -> u8 {
        AnsData::ans_size_log(self)
    }

    /// Number of symbols.
    fn n_symbols(&self) -> u64 {
        AnsData::n_symbols(self)
    }

    /// Dense -> u8 alphabet.
    fn alphabet(&self) -> &[u8] {
        AnsData::alphabet(self)
    }

    /// Per-symbol weights.
    fn weights(&self) -> &[Weight] {
        AnsData::weights(self)
    }

    /// Encoded byte buffer.
    fn encoded(&self) -> &ByteBuffer {
        AnsData::encoded(self)
    }

    /// Final tANS state.
    fn final_state(&self) -> u32 {
        AnsData::final_state(self)
    }
}

impl<T: TypedArrayRef<Ans>> AnsArrayExt for T {}

impl Ans {
    /// Construct an [`AnsArray`] from already-validated parts.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        ans_size_log: u8,
        alphabet: Vec<u8>,
        weights: Vec<Weight>,
        final_state: u32,
        bit_len: u64,
        encoded: ByteBuffer,
        n_symbols: usize,
    ) -> VortexResult<AnsArray> {
        let dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let n_symbols_u64 = u64::try_from(n_symbols)
            .map_err(|_| vortex_error::vortex_err!("array length {n_symbols} exceeds u64"))?;
        let data = AnsData::new(
            ans_size_log,
            n_symbols_u64,
            alphabet,
            weights,
            final_state,
            bit_len,
            encoded,
        );
        validate_parts(&data, &dtype, n_symbols)?;
        // SAFETY: validate_parts above checked all type/length invariants.
        Ok(unsafe { Array::from_parts_unchecked(ArrayParts::new(Ans, dtype, n_symbols, data)) })
    }

    /// Encode a `Primitive<u8>` array with tANS.
    ///
    /// `ans_size_log` controls table size (`2^ans_size_log` states; a
    /// typical value is `12`). Must be in `4..=14`. The encoder builds
    /// a dense alphabet from the symbols seen and quantizes their
    /// frequencies to fit the table.
    pub fn encode(
        parray: ArrayView<'_, Primitive>,
        ans_size_log: u8,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<AnsArray> {
        if !(MIN_ANS_SIZE_LOG..=MAX_ANS_SIZE_LOG).contains(&ans_size_log) {
            vortex_bail!(
                "AnsArray ans_size_log must be in {MIN_ANS_SIZE_LOG}..={MAX_ANS_SIZE_LOG}, got {ans_size_log}"
            );
        }
        let _ = ctx;
        let ptype = PrimitiveArrayExt::ptype(&parray);
        if ptype != PType::U8 {
            vortex_bail!("AnsArray::encode requires u8 input, got {ptype}");
        }
        let validity = PrimitiveArrayExt::validity(&parray);
        if !matches!(validity, Validity::NonNullable) {
            vortex_bail!("AnsArray::encode requires non-nullable input in this phase");
        }
        let parray = parray.into_owned();
        let buf = parray.into_buffer::<u8>();
        let bytes = buf.as_slice();
        let n = bytes.len();

        // Empty input: empty alphabet, empty buffer.
        if n == 0 {
            return Self::try_new(
                ans_size_log,
                Vec::new(),
                Vec::new(),
                1u32 << ans_size_log,
                0,
                ByteBuffer::empty(),
                0,
            );
        }

        // Build the dense alphabet in first-occurrence order and count
        // per-symbol occurrences.
        let mut byte_to_symbol = [u32::MAX; 256];
        let mut alphabet: Vec<u8> = Vec::new();
        let mut counts: Vec<u32> = Vec::new();
        for &b in bytes {
            let slot = &mut byte_to_symbol[b as usize];
            if *slot == u32::MAX {
                *slot = alphabet.len() as u32;
                alphabet.push(b);
                counts.push(1);
            } else {
                counts[*slot as usize] += 1;
            }
        }

        // Degenerate: single distinct byte. We still build a "weights"
        // table containing a single 1, mark size_log = 0, and emit
        // an empty bit stream.
        if alphabet.len() == 1 {
            return Self::try_new(
                ans_size_log,
                alphabet,
                vec![1],
                1u32,
                0,
                ByteBuffer::empty(),
                n,
            );
        }

        let (chosen_size_log, weights) = quantize_weights(&counts, n, ans_size_log);
        if chosen_size_log < MIN_ANS_SIZE_LOG && alphabet.len() > 1 {
            // We honor the caller's choice even if the auto-quantizer
            // could shrink the table. Pad back up by doubling.
            // In practice, quantize_weights only returns < ans_size_log
            // when all weights share a common power-of-2 factor, which
            // happens for highly uniform alphabets. We renormalize back
            // up so we always serialize at the requested size_log.
        }

        // Convert input bytes into dense symbol ids.
        let symbols: Vec<Symbol> = bytes.iter().map(|&b| byte_to_symbol[b as usize]).collect();

        let encoder = Encoder::new(chosen_size_log, &weights)?;
        let (encoded_bytes, bit_len, final_state) = encode_symbols(&symbols, &encoder)?;
        let encoded = Buffer::<u8>::from(encoded_bytes).into_byte_buffer();

        Self::try_new(
            chosen_size_log,
            alphabet,
            weights,
            final_state,
            bit_len,
            encoded,
            n,
        )
    }
}

impl OperationsVTable<Ans> for Ans {
    fn scalar_at(
        array: ArrayView<'_, Ans>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Sequential decode is the only path that exists for a tANS
        // stream without batch checkpoints. We document the O(N) cost
        // on the type; this is acceptable for the standalone array in
        // P5. P6 will add batch boundaries inside the layered stack.
        let data = array.data();
        let values = decode_all(data)?;
        let v = *values
            .get(index)
            .ok_or_else(|| vortex_error::vortex_err!("AnsArray index {index} out of bounds"))?;
        Ok(Scalar::primitive(v, Nullability::NonNullable))
    }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::*;

    fn primitive_u8(values: Vec<u8>) -> PrimitiveArray {
        PrimitiveArray::new(Buffer::from(values), Validity::NonNullable)
    }

    fn round_trip(values: Vec<u8>, ans_size_log: u8) -> VortexResult<AnsArray> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = primitive_u8(values.clone());
        let encoded = Ans::encode(parray.as_view(), ans_size_log, &mut ctx)?;
        let decoded = encoded
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let expected = primitive_u8(values);
        assert_arrays_eq!(decoded, expected);
        Ok(encoded)
    }

    #[test]
    fn round_trip_uniform_alphabet() -> VortexResult<()> {
        // Uniform symbols in 0..16, N = 10_000.
        let mut rng = SmallRng::seed_from_u64(0xA5);
        let values: Vec<u8> = (0..10_000).map(|_| (rng.random::<u8>()) & 0x0F).collect();
        round_trip(values, 12)?;
        Ok(())
    }

    #[test]
    fn round_trip_skewed_alphabet() -> VortexResult<()> {
        // Zipf-ish: symbol k drawn with probability ~ 1/(k+1), over 0..16.
        let mut rng = SmallRng::seed_from_u64(0xBEEF);
        let mut values: Vec<u8> = Vec::with_capacity(10_000);
        let weights: Vec<f64> = (0..16).map(|k| 1.0 / (k as f64 + 1.0)).collect();
        let total: f64 = weights.iter().sum();
        let cdf: Vec<f64> = weights
            .iter()
            .scan(0.0, |s, w| {
                *s += w / total;
                Some(*s)
            })
            .collect();
        for _ in 0..10_000 {
            let r: f64 = rng.random();
            let mut sym = 15u8;
            for (k, &c) in cdf.iter().enumerate() {
                if r < c {
                    sym = k as u8;
                    break;
                }
            }
            values.push(sym);
        }
        let encoded = round_trip(values.clone(), 12)?;
        // Skewed data should compress to <1 byte/symbol.
        assert!(
            encoded.encoded().len() < values.len(),
            "no compression on skewed input: {} bytes for {} symbols",
            encoded.encoded().len(),
            values.len(),
        );
        Ok(())
    }

    #[test]
    fn round_trip_full_byte_alphabet() -> VortexResult<()> {
        let mut rng = SmallRng::seed_from_u64(0xC0DE);
        let values: Vec<u8> = (0..10_000).map(|_| rng.random::<u8>()).collect();
        round_trip(values, 12)?;
        Ok(())
    }

    #[rstest]
    #[case::log4(4)]
    #[case::log8(8)]
    #[case::log12(12)]
    fn round_trip_size_log_variations(#[case] ans_size_log: u8) -> VortexResult<()> {
        // Use 8 distinct symbols (fits in size_log >= 3). Skew the
        // distribution so quantization is non-trivial.
        let mut rng = SmallRng::seed_from_u64(0xFACE);
        let values: Vec<u8> = (0..2_000)
            .map(|_| {
                let r: u32 = rng.random();
                // Symbol 0 most likely, then 1..=7 progressively rarer.
                let bucket = (r % 64) as u8;
                if bucket < 32 {
                    0
                } else if bucket < 48 {
                    1
                } else if bucket < 56 {
                    2
                } else if bucket < 60 {
                    3
                } else if bucket < 62 {
                    4
                } else if bucket < 63 {
                    5
                } else {
                    6
                }
            })
            .collect();
        round_trip(values, ans_size_log)?;
        Ok(())
    }

    #[test]
    fn singleton_round_trip() -> VortexResult<()> {
        round_trip(vec![42u8], 12)?;
        Ok(())
    }

    #[test]
    fn empty_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = primitive_u8(Vec::new());
        let encoded = Ans::encode(parray.as_view(), 12, &mut ctx)?;
        assert_eq!(encoded.len(), 0);
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(decoded.len(), 0);
        Ok(())
    }

    #[test]
    fn scalar_at_matches_canonical() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut rng = SmallRng::seed_from_u64(0xCAFE);
        let values: Vec<u8> = (0..1_000).map(|_| rng.random::<u8>() & 0x0F).collect();
        let parray = primitive_u8(values.clone());
        let encoded = Ans::encode(parray.as_view(), 12, &mut ctx)?;
        let arr = encoded.into_array();
        let decoded = arr
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u8>();
        let mut idx_rng = SmallRng::seed_from_u64(0xD00D);
        for _ in 0..64 {
            let i = idx_rng.random_range(0..values.len());
            let s = arr.execute_scalar(i, &mut ctx)?;
            assert_eq!(s, Scalar::from(decoded.as_slice()[i]));
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_size_log() {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = primitive_u8(vec![0u8, 1, 2, 3]);
        let lo = Ans::encode(parray.as_view(), MIN_ANS_SIZE_LOG - 1, &mut ctx);
        assert!(lo.is_err(), "expected error for size_log too low");
        let hi = Ans::encode(parray.as_view(), MAX_ANS_SIZE_LOG + 1, &mut ctx);
        assert!(hi.is_err(), "expected error for size_log too high");
    }

    #[test]
    fn singleton_alphabet_round_trip() -> VortexResult<()> {
        // All-zero input: alphabet of 1, no bits written.
        let values = vec![0u8; 500];
        round_trip(values, 12)?;
        Ok(())
    }
}
