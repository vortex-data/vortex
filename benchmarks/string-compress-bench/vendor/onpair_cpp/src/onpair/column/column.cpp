#include <onpair/column/column.h>
#include <onpair/encoding/training/trainer.h>
#include <onpair/encoding/parsing/parser.h>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace onpair {

// ─────────────────────────────────────────────────────────────────────────────
// compress_raw  (the single implementation that both public overloads reach)
// ─────────────────────────────────────────────────────────────────────────────

OnPairColumn OnPairColumn::compress_raw(const uint8_t*  data,
                                         const uint32_t* offsets,
                                         size_t          n,
                                         const Config&   cfg)
{
    OnPairColumn col;

    encoding::TrainResult trained = encoding::train(data, offsets, n, cfg);
    encoding::parse(data, offsets, n, trained.lpm, cfg.bits, col.store_);
    col.dict_ = std::move(trained.dict);

    return col;
}

// ─────────────────────────────────────────────────────────────────────────────
// Arrow-style public overload
// ─────────────────────────────────────────────────────────────────────────────

OnPairColumn OnPairColumn::compress(const char*     data,
                                     const uint32_t* offsets,
                                     size_t          n,
                                     const Config&   cfg)
{
    return compress_raw(reinterpret_cast<const uint8_t*>(data), offsets, n, cfg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialisation
// ─────────────────────────────────────────────────────────────────────────────

namespace {

template<typename T>
void write_pod(std::ostream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template<typename T>
void write_vec(std::ostream& out, const std::vector<T>& v) {
    const uint32_t sz = static_cast<uint32_t>(v.size());
    write_pod(out, sz);
    if (sz) out.write(reinterpret_cast<const char*>(v.data()),
                      sz * sizeof(T));
}

template<typename T>
T read_pod(std::istream& in) {
    T v; in.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!in) throw std::runtime_error("OnPair: truncated file");
    return v;
}

template<typename T>
std::vector<T> read_vec(std::istream& in) {
    const uint32_t sz = read_pod<uint32_t>(in);
    std::vector<T> v(sz);
    if (sz) in.read(reinterpret_cast<char*>(v.data()), sz * sizeof(T));
    if (!in) throw std::runtime_error("OnPair: truncated file");
    return v;
}

} // namespace

// Binary format:
//   "ONPAIR01"            8 bytes  magic + version
//   bit_width             1 byte
//   dict.bytes            uint32 count + data
//   dict.offsets          uint32 count + uint32 data
//   store.packed          uint32 count + uint64 data  (sentinel word excluded)
//   store.boundaries      uint32 count + uint32 data

static constexpr char MAGIC[8] = {'O','N','P','A','I','R','0','1'};

void OnPairColumn::write_to(std::ostream& out) const {
    out.write(MAGIC, 8);

    write_pod(out, store_.bit_width);

    // Write only the true token bytes (offsets.back()), not the trailing
    // decoder-padding added by pad_for_decoder().  read_from() re-adds it.
    const uint32_t true_bytes = dict_.offsets.empty() ? 0u : dict_.offsets.back();
    write_pod(out, true_bytes);
    if (true_bytes) out.write(reinterpret_cast<const char*>(dict_.bytes.data()), true_bytes);
    write_vec(out, dict_.offsets);
    // Write packed words without the trailing sentinel added by BitWriter::flush().
    // read_from() re-adds it.
    {
        const uint32_t real_words = store_.packed.empty()
            ? 0u : static_cast<uint32_t>(store_.packed.size()) - 1u;
        write_pod(out, real_words);
        if (real_words)
            out.write(reinterpret_cast<const char*>(store_.packed.data()),
                      real_words * sizeof(uint64_t));
    }
    write_vec(out, store_.boundaries);
}

OnPairColumn OnPairColumn::read_from(std::istream& in) {
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, MAGIC, 8) != 0)
        throw std::runtime_error("OnPair: invalid magic / wrong version");

    const uint8_t bit_width = read_pod<uint8_t>(in);
    if (!is_valid_bits(bit_width))
        throw std::runtime_error("OnPair: invalid bit_width in file");

    OnPairColumn col;
    col.dict_.bytes   = read_vec<uint8_t>(in);
    col.dict_.offsets = read_vec<uint32_t>(in);
    col.dict_.pad_for_decoder();  // restore decoder-padding stripped by write_to()

    col.store_.bit_width  = bit_width;
    col.store_.packed = read_vec<uint64_t>(in);
    if (!col.store_.packed.empty())
        col.store_.packed.push_back(0);  // restore sentinel for safe over-read
    col.store_.boundaries = read_vec<uint32_t>(in);

    return col;
}

} // namespace onpair
