#pragma once

#include "vortex_common.hpp"
#include "vortex_expr.hpp"

namespace vortex {

class LayoutReader {
public:
	explicit LayoutReader(vx_layout_reader *reader) : reader(reader) {
	}

	~LayoutReader() {
		vx_layout_reader_free(reader);
	}

	static std::shared_ptr<LayoutReader> CreateFromFile(vortex::FileReader *file) {
		auto reader = Try([&](auto err) { return vx_layout_reader_create(file->file, err); });
		return std::make_shared<LayoutReader>(reader);
	}

	vx_array_iterator *Scan(const vx_file_scan_options *options) {
		return Try([&](auto err) { return vx_layout_reader_scan(this->reader, options, err); });
	}

	vx_layout_reader *reader;
};

} // namespace vortex
