#pragma once
#include <onpair/column/column_view.h>
#include <onpair/core/dictionary.h>
#include <onpair/core/store.h>
#include <onpair/encoding/training/config.h>
#include <concepts>
#include <cstddef>
#include <iosfwd>
#include <ranges>
#include <string_view>
#include <vector>

namespace onpair {

// ─────────────────────────────────────────────────────────────────────────────
// OnPairColumn
// ─────────────────────────────────────────────────────────────────────────────
// Owning, move-only compressed column.  Produced by compress(); consumed by
// view().  Serialisable.

class OnPairColumn {
public:
    using Config = encoding::TrainingConfig;

    // ── Compression ───────────────────────────────────────────────────────────

    // Accepts any range whose elements are convertible to std::string_view.
    template<std::ranges::input_range Range>
        requires std::convertible_to<std::ranges::range_value_t<Range>, std::string_view>
    static OnPairColumn compress(Range&& strings, const Config& cfg = {});

    // Arrow-style: flat byte buffer + offsets array of size n+1.
    static OnPairColumn compress(const char* data, const uint32_t* offsets,
                                 size_t n, const Config& cfg = {});

    // ── Access ────────────────────────────────────────────────────────────────
    OnPairColumnView view() const noexcept { return OnPairColumnView(*this); }

    // ── Metadata ──────────────────────────────────────────────────────────────
    size_t num_strings() const noexcept { return view().num_strings(); }
    size_t bytes_used()  const noexcept { return view().bytes_used();  }
    BitWidth bits()      const noexcept { return view().bits();        }

    // ── Serialisation ─────────────────────────────────────────────────────────
    void write_to(std::ostream& out) const;
    static OnPairColumn read_from(std::istream& in);

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    OnPairColumn()                               = default;
    OnPairColumn(OnPairColumn&&)                 = default;
    OnPairColumn& operator=(OnPairColumn&&)      = default;
    OnPairColumn(const OnPairColumn&)            = delete;
    OnPairColumn& operator=(const OnPairColumn&) = delete;

private:
    Dictionary dict_;
    Store      store_;

    static OnPairColumn compress_raw(const uint8_t*  data,
                                     const uint32_t* offsets,
                                     size_t          n,
                                     const Config&   cfg);

    friend class OnPairColumnView;
};

// ─── OnPairColumn::compress<Range> (template definition) ─────────────────────

template<std::ranges::input_range Range>
    requires std::convertible_to<std::ranges::range_value_t<Range>, std::string_view>
OnPairColumn OnPairColumn::compress(Range&& strings, const Config& cfg)
{
    std::vector<uint8_t>  data;
    std::vector<uint32_t> offsets;

    if constexpr (std::ranges::sized_range<Range>)
        offsets.reserve(std::ranges::size(strings) + 1);
    offsets.push_back(0);

    for (const auto& s : strings) {
        const std::string_view sv = s;
        data.insert(data.end(),
                    reinterpret_cast<const uint8_t*>(sv.data()),
                    reinterpret_cast<const uint8_t*>(sv.data()) + sv.size());
        offsets.push_back(static_cast<uint32_t>(data.size()));
    }

    const size_t n = offsets.size() - 1;
    return compress_raw(data.data(), offsets.data(), n, cfg);
}

// ─── OnPairColumnView constructor (needs OnPairColumn to be complete) ────────
inline OnPairColumnView::OnPairColumnView(const OnPairColumn& col) noexcept
    : sv_(col.store_), dv_(col.dict_) {}

} // namespace onpair
