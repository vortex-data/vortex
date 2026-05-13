#include <onpair/encoding/training/trainer.h>
#include <onpair/encoding/training/dynamic_threshold.h>
#include <onpair/core/dictionary_view.h>
#include <onpair/core/types.h>
#include <boost/unordered/unordered_flat_map.hpp>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <variant>

namespace onpair::encoding {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// sort_dictionary — internal helper
//
// Sorts the dictionary tokens lexicographically by their byte sequences,
// rebuilds the flat bytes/offsets representation in the new order, and
// reconstructs the LPM with remapped token IDs.
// ─────────────────────────────────────────────────────────────────────────────

void sort_dictionary(TrainResult& result)
{
    const DictionaryView src(result.dict);
    const size_t N = src.num_tokens();

    // ── Build sorted permutation ──────────────────────────────────────────
    // perm[new_id] = old_id: the token that occupies position new_id after sort.
    std::vector<Token> perm(N);
    std::iota(perm.begin(), perm.end(), Token(0));
    std::sort(perm.begin(), perm.end(), [&](const Token& a, const Token& b) {
        const int cmp = std::memcmp(src.data(a), src.data(b),
                                    std::min(src.token_size(a), src.token_size(b)));
        return cmp != 0 ? cmp < 0 : src.token_size(a) < src.token_size(b);
    });

    // ── Rebuild dictionary in sorted order ────────────────────────────────
    Dictionary sorted_dict;
    sorted_dict.bytes.reserve(src.bytes_used());
    sorted_dict.offsets.reserve(N + 1);
    sorted_dict.offsets.push_back(0);

    for (size_t new_id = 0; new_id < N; ++new_id) {
        const Token    old_id = perm[new_id];
        const ByteSpan sp     = src.span(old_id);
        sorted_dict.bytes.insert(sorted_dict.bytes.end(),
                                 src.raw_bytes() + sp.begin,
                                 src.raw_bytes() + sp.end);
        sorted_dict.offsets.push_back(uint32_t(sorted_dict.bytes.size()));
    }
    result.dict = std::move(sorted_dict);

    // ── Rebuild LPM with new (sorted) token IDs ──────────────────────────
    result.lpm = LongestPrefixMatcher::from_dictionary(DictionaryView(result.dict));
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// train()
//
// Discovers merge tokens via frequency-threshold scanning, then
// sorts the dictionary lexicographically before returning.
// ─────────────────────────────────────────────────────────────────────────────

TrainResult train(const uint8_t* data,
                  const uint32_t* offsets,
                  size_t n,
                  const TrainingConfig& cfg)
{
    TrainResult result;

    // ── Initialise with the 256 single-byte base tokens ───────────────────────
    // Token i = byte value i.
    // offsets layout: offsets[0]=0, offsets[i+1]-offsets[i]=len(i).
    // Note: result.lpm is default-constructed and already has all 256 single-byte
    // tokens pre-inserted (IDs 0–255); only the dictionary needs explicit init.
    const size_t dict_capacity = max_dict_size(cfg.bits);
    result.dict.offsets.reserve(dict_capacity + 1);
    result.dict.bytes.reserve(dict_capacity * MAX_TOKEN_SIZE);
    result.dict.offsets.push_back(0);

    for (uint16_t i = 0; i <= 255; ++i) {
        const uint8_t b = uint8_t(i);
        result.dict.bytes.push_back(b);
        result.dict.offsets.push_back(uint32_t(result.dict.bytes.size()));
    }

    // ── Threshold setup ───────────────────────────────────────────────────────
    uint8_t threshold;
    std::optional<DynamicThresholdController> dyn;

    if (const auto* ft = std::get_if<FixedThreshold>(&cfg.threshold)) {
        threshold = ft->value;
    } else {
        const auto& dt = std::get<DynamicThreshold>(cfg.threshold);
        size_t total_bytes = offsets[n]; // offsets[n] == total byte count across all strings
        // Capacity = multi-byte tokens available (total - 256 base tokens).
        const size_t capacity = dict_capacity - 256;
        dyn.emplace(capacity, total_bytes, dt.sample_fraction);
        threshold = dyn->get();
    }

    // ── Shuffle training order ────────────────────────────────────────────────
    std::mt19937_64 rng(cfg.seed.value_or(std::random_device{}()));
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), rng);

    // ── Pair frequency map ────────────────────────────────────────────────────
    // Key packs two Token values into one uint32_t — avoids a custom hash.
    boost::unordered_flat_map<uint32_t, uint8_t> freq;

    // ── Main training loop ────────────────────────────────────────────────────
    bool full_dictionary = false;
    bool budget_exhausted = false;

    for (uint32_t idx : order) {
        if (full_dictionary || budget_exhausted) break;

        const uint8_t* str = data + offsets[idx];
        const size_t   len = offsets[idx + 1] - offsets[idx];
        if (len == 0) continue;

        // Greedy parse of the current string.
        auto first = result.lpm.find_longest_match(str, len);
        Token  prev_id  = first.first;
        size_t prev_len = first.second;
        size_t pos      = prev_len;

        if (dyn) {
            dyn->on_bytes_scanned(prev_len);
            budget_exhausted = dyn->budget_exhausted();
            if (budget_exhausted) break;
        }

        while (pos < len) {
            auto m = result.lpm.find_longest_match(str + pos, len - pos);
            const Token  curr_id  = m.first;
            const size_t curr_len = m.second;

            if (dyn) {
                dyn->on_bytes_scanned(curr_len);
                budget_exhausted = dyn->budget_exhausted();
                if (budget_exhausted) break;
            }

            const size_t pair_len = prev_len + curr_len;

            // Pairs larger than MAX_TOKEN_SIZE can never be merged.
            // Skip the hashmap entirely — keeps freq small and cache-hot.
            if (pair_len <= MAX_TOKEN_SIZE) {
                const uint32_t key = (uint32_t(prev_id) << 16) | uint32_t(curr_id);
                auto& f = freq[key];
                f += (f < 255);  // saturating increment — branchless, no wrap-around
                if (f >= threshold) {
                    // Merge: create new token for this pair.
                    const Token new_id = result.lpm.insert(str + pos - prev_len, pair_len);
                    result.dict.bytes.insert(result.dict.bytes.end(),
                                             str + pos - prev_len,
                                             str + pos + curr_len);
                    result.dict.offsets.push_back(uint32_t(result.dict.bytes.size()));

                    if (result.lpm.size() == dict_capacity) {
                        full_dictionary = true;
                        break;
                    }

                    if (dyn) {
                        dyn->on_entry_created();
                        threshold = dyn->get();
                    }

                    freq.erase(key);
                    prev_id  = new_id;
                    prev_len = pair_len;
                    pos += curr_len;
                    continue;
                }
            }
            // Pair too long or frequency below threshold: no merge.
            prev_id  = curr_id;
            prev_len = curr_len;
            pos += curr_len;
        }
    }

    // ── Sort lexicographically before returning ───────────────────────────
    sort_dictionary(result);

    result.dict.pad_for_decoder();

    return result;
}

} // namespace onpair::encoding
