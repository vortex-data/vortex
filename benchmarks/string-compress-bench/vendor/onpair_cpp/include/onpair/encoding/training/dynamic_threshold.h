#pragma once
#include <algorithm>
#include <cstddef>

// ─────────────────────────────────────────────────────────────────────────────
// DynamicThresholdController — encoding-internal controller.
//
// Adaptively adjusts the pair-merge frequency threshold during training so that
// the dictionary fills to capacity within a configurable scan budget.
//
// Strategy: every ~capacity/128 new entries, compare the recent entry-creation
// rate against the rate needed to fill the remaining capacity in the remaining
// budget.  If we're creating entries too fast (ratio > 2), raise the threshold;
// if too slow (ratio < 0.5), lower it.
// ─────────────────────────────────────────────────────────────────────────────

namespace onpair::encoding {

class DynamicThresholdController {
public:
    // `capacity`        = number of multi-byte tokens that can still be created
    //                     (total dict capacity minus the 256 base tokens)
    // `total_bytes`     = total bytes in the training input
    // `scan_fraction`   = fraction of total_bytes to scan before stopping
    DynamicThresholdController(size_t capacity, size_t total_bytes, double scan_fraction)
        : capacity_(capacity)
        , scan_budget_(static_cast<size_t>(total_bytes * scan_fraction))
        , check_interval_(std::max<size_t>(capacity / 128, 64))
        , threshold_(2)
        , next_checkpoint_(check_interval_)
    {}

    uint8_t get()             const noexcept { return threshold_; }
    bool   budget_exhausted() const noexcept { return bytes_scanned_ > scan_budget_; }

    void on_bytes_scanned(size_t n) noexcept { bytes_scanned_ += n; }

    void on_entry_created() noexcept {
        ++entries_created_;
        if (entries_created_ >= next_checkpoint_) rebalance();
    }

private:
    void rebalance() noexcept {
        const size_t delta_e = entries_created_ - entries_at_check_;
        const size_t delta_b = bytes_scanned_   - bytes_at_check_;

        const double recent_rate = delta_b > 0
            ? double(delta_e) / double(delta_b) : 1e9;

        const size_t e_rem = capacity_ > entries_created_
            ? capacity_ - entries_created_ : 1;
        const size_t b_rem = scan_budget_ > bytes_scanned_
            ? scan_budget_ - bytes_scanned_ : 1;

        const double target_rate = double(e_rem) / double(b_rem);
        const double ratio = target_rate > 0 ? recent_rate / target_rate : 1e9;

        if      (ratio > 2.0 && threshold_ < 255) ++threshold_;
        else if (ratio < 0.5) threshold_ = (threshold_ > 2) ? threshold_ - 1 : 2;

        entries_at_check_ = entries_created_;
        bytes_at_check_   = bytes_scanned_;
        next_checkpoint_  = entries_created_ + check_interval_;
    }

    const size_t capacity_;
    const size_t scan_budget_;
    const size_t check_interval_;

    uint8_t threshold_       = 2;
    size_t entries_created_  = 0;
    size_t bytes_scanned_    = 0;
    size_t entries_at_check_ = 0;
    size_t bytes_at_check_   = 0;
    size_t next_checkpoint_;
};

} // namespace onpair::encoding
