#pragma once

#include "vortex.hpp"
#include "duckdb/storage/object_cache.hpp"

namespace vortex {

class VortexSession : public duckdb::ObjectCacheEntry {
public:
	VortexSession() : session(vx_session_create()) {
	}

	~VortexSession() override {
		vx_session_free(session);
	}

	vx_session *session;

	static std::string ObjectType() {
		return "vortex_session_cache_metadata";
	}

	std::string GetObjectType() override {
		return ObjectType();
	}
};

} // namespace vortex