// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Branchless shift-packed DFA for short contains matching (`LIKE '%needle%'`, needle ≤ 7).

use fsst::Symbol;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::build_escape_folded_table;
use super::compose_packed;
use super::pack_shift_table;
use super::shift_extract;

/// Branchless escape-folded DFA for short needles (len <= 7).
///
/// Folds escape handling into the state space so that `matches()` is
/// completely branchless (except for loop control). The state layout is:
/// - States 0..N-1: normal match-progress states
/// - State N: accept (sticky for all inputs)
/// - States N+1..2N: escape states (state `s+N+1` means "was in state `s`,
///   just consumed ESCAPE_CODE")
///
/// Total states: 2N+1. With 4-bit packing, max N=7.
///
/// Uses a decomposed hierarchical lookup that processes 4 code bytes per
/// loop iteration with only ~3 KB of tables:
///
/// 1. **Equivalence class table** (256 B): maps each code byte to a class
///    id. Bytes with identical transition u64s share a class -- typically
///    only ~6-10 classes exist (needle chars + escape + "miss-all").
/// 2. **Pair-compose table** (~N^2 B): maps `(class0, class1)` to a 2-byte
///    palette index. Typically ~36 entries.
/// 3. **4-byte compose table** (~M^2 x 8 B): maps `(palette0, palette1)` to
///    the composed packed u64 for all 4 bytes. Typically ~81 entries = 648 B.
///
/// Each loop iteration: 4 class lookups (parallel, 256 B table) -> 2
/// pair-compose lookups (parallel, ~36 B table) -> 1 compose lookup
/// (~648 B table) -> 1 shift+mask. All tables fit in L1 cache.
pub(crate) struct BranchlessShiftDfa {
    /// Maps each code byte to its equivalence class. Bytes with the same
    /// packed transition u64 share a class. (256 bytes)
    eq_class: [u8; 256],
    /// Maps `(class0 * n_classes + class1)` -> 2-byte palette index.
    pair_compose: Vec<u8>,
    /// Number of equivalence classes (stride for pair_compose).
    n_classes: usize,
    /// Maps `(palette0 * n_palette + palette1)` -> composed packed u64
    /// for 4 bytes.
    compose_4b: Vec<u64>,
    /// Number of unique 2-byte palette entries (stride for compose_4b).
    n_palette: usize,
    /// 1-byte fallback transitions for trailing bytes.
    transitions_1b: [u64; 256],
    /// 2-byte palette for the remainder path (2-3 trailing bytes).
    palette_2b: Vec<u64>,
    accept_state: u8,
}

impl BranchlessShiftDfa {
    const BITS: u32 = 4;
    /// Maximum needle length: need 2N+1 states to fit in 16 slots (4 bits).
    /// 2*7+1 = 15 <= 16, so max N = 7.
    pub(crate) const MAX_NEEDLE_LEN: usize = 7;

    pub(crate) fn new(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        needle: &[u8],
    ) -> VortexResult<Self> {
        let n = needle.len();
        if n > Self::MAX_NEEDLE_LEN {
            vortex_bail!(
                "needle length {} exceeds maximum {} for branchless shift DFA",
                n,
                Self::MAX_NEEDLE_LEN
            );
        }

        #[expect(clippy::cast_possible_truncation, reason = "n ≤ MAX_NEEDLE_LEN (7)")]
        let accept_state = n as u8;
        let total_states = 2 * accept_state + 1;

        let fused = build_escape_folded_table(symbols, symbol_lengths, needle);
        let transitions_1b = pack_shift_table(&fused, total_states, Self::BITS);

        // Build equivalence classes: group bytes with identical transition u64.
        let mut eq_class = [0u8; 256];
        let mut class_representatives: Vec<u64> = Vec::new();
        for byte_val in 0..256usize {
            let t = transitions_1b[byte_val];
            let cls = class_representatives
                .iter()
                .position(|&v| v == t)
                .unwrap_or_else(|| {
                    class_representatives.push(t);
                    class_representatives.len() - 1
                });
            #[expect(clippy::cast_possible_truncation, reason = "≤ 256 equivalence classes")]
            {
                eq_class[byte_val] = cls as u8;
            }
        }
        let n_classes = class_representatives.len();

        // Build pair-compose: for each (class0, class1), compose the two
        // 1-byte transitions and deduplicate into a 2-byte palette.
        let (pair_compose, palette_2b) =
            Self::build_pair_compose(&class_representatives, n_classes, total_states);

        // Build 4-byte composition: compose_4b[p0 * n + p1] gives the packed
        // u64 for applying palette_2b[p0] then palette_2b[p1] in sequence.
        let n_palette = palette_2b.len();
        let compose_4b = Self::build_compose_4b(&palette_2b, total_states);

        Ok(Self {
            eq_class,
            pair_compose,
            n_classes,
            compose_4b,
            n_palette,
            transitions_1b,
            palette_2b,
            accept_state,
        })
    }

    /// Build the pair-compose table and 2-byte palette from equivalence
    /// class representatives.
    fn build_pair_compose(
        class_reps: &[u64],
        n_classes: usize,
        total_states: u8,
    ) -> (Vec<u8>, Vec<u64>) {
        let mut pair_compose = vec![0u8; n_classes * n_classes];
        let mut palette_2b: Vec<u64> = Vec::new();

        for c0 in 0..n_classes {
            for c1 in 0..n_classes {
                let packed =
                    compose_packed(class_reps[c0], class_reps[c1], total_states, Self::BITS);
                let idx = palette_2b
                    .iter()
                    .position(|&v| v == packed)
                    .unwrap_or_else(|| {
                        palette_2b.push(packed);
                        palette_2b.len() - 1
                    });
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "palette size ≤ n_classes² ≤ 256"
                )]
                {
                    pair_compose[c0 * n_classes + c1] = idx as u8;
                }
            }
        }
        (pair_compose, palette_2b)
    }

    /// Compose pairs of 2-byte palette entries into a 4-byte lookup table.
    fn build_compose_4b(palette_2b: &[u64], total_states: u8) -> Vec<u64> {
        let n = palette_2b.len();
        let mut compose = vec![0u64; n * n];
        for p0 in 0..n {
            for p1 in 0..n {
                compose[p0 * n + p1] =
                    compose_packed(palette_2b[p0], palette_2b[p1], total_states, Self::BITS);
            }
        }
        compose
    }

    /// Process remaining bytes after the interleaved common prefix.
    #[inline]
    fn finish_tail(&self, mut state: u8, codes: &[u8]) -> u8 {
        let chunks = codes.chunks_exact(4);
        let rem = chunks.remainder();

        for chunk in chunks {
            // SAFETY: chunk[i] is u8, eq_class has 256 entries — index always in bounds.
            let ec0 = unsafe { *self.eq_class.get_unchecked(usize::from(chunk[0])) };
            let ec1 = unsafe { *self.eq_class.get_unchecked(usize::from(chunk[1])) };
            let ec2 = unsafe { *self.eq_class.get_unchecked(usize::from(chunk[2])) };
            let ec3 = unsafe { *self.eq_class.get_unchecked(usize::from(chunk[3])) };
            let p0 = unsafe {
                *self
                    .pair_compose
                    .get_unchecked(usize::from(ec0) * self.n_classes + usize::from(ec1))
            };
            let p1 = unsafe {
                *self
                    .pair_compose
                    .get_unchecked(usize::from(ec2) * self.n_classes + usize::from(ec3))
            };
            let packed = unsafe {
                *self
                    .compose_4b
                    .get_unchecked(usize::from(p0) * self.n_palette + usize::from(p1))
            };
            state = shift_extract(packed, state, Self::BITS);
        }

        if rem.len() >= 2 {
            let ec0 = self.eq_class[usize::from(rem[0])];
            let ec1 = self.eq_class[usize::from(rem[1])];
            let p = self.pair_compose[usize::from(ec0) * self.n_classes + usize::from(ec1)];
            let packed = self.palette_2b[usize::from(p)];
            state = shift_extract(packed, state, Self::BITS);
            if rem.len() == 3 {
                let packed = self.transitions_1b[usize::from(rem[2])];
                state = shift_extract(packed, state, Self::BITS);
            }
        } else if rem.len() == 1 {
            let packed = self.transitions_1b[usize::from(rem[0])];
            state = shift_extract(packed, state, Self::BITS);
        }

        state
    }

    /// Branchless matching processing four code bytes per iteration.
    #[inline(never)]
    pub(crate) fn matches(&self, codes: &[u8]) -> bool {
        self.finish_tail(0, codes) == self.accept_state
    }
}
