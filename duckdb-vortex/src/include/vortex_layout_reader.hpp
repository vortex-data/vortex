#pragma once

#include "vortex_common.hpp"
#include "expr/expr.hpp"

class VortexLayoutReader {
public:
	explicit VortexLayoutReader(vx_layout_reader *reader) : reader(reader) {
	}

	~VortexLayoutReader() {
		vx_layout_reader_free(reader);
	}

	static std::shared_ptr<VortexLayoutReader> CreateFromFile(VortexFileReader *file) {
		auto reader = Try([&](auto err) { return vx_layout_reader_create(file->file, err); });
		return std::make_shared<VortexLayoutReader>(reader);
	}

	vx_array_stream *Scan(const vx_file_scan_options *options) {
		return Try([&](auto err) { return vx_layout_reader_scan(this->reader, options, err); });
	}

	vx_layout_reader *reader;
};
