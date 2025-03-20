protoc --cpp_out=gen ../vortex-proto/proto/*.proto -I ../vortex-proto/proto

mv gen/*.pb.h gen/include