// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <iostream>
#include <string_view>
#include <vortex/data_source.hpp>

using vortex::DataSource;
using vortex::DataType;
using vortex::DataTypeVariant;
using vortex::Session;

static void print_dtype(const DataType &dtype);

static std::string_view ptype_name(vortex::PType p) {
    using enum vortex::PType;
    switch (p) {
    case U8:
        return "uint8_t";
    case U16:
        return "uint16_t";
    case U32:
        return "uint32_t";
    case U64:
        return "uint64_t";
    case I8:
        return "int8_t";
    case I16:
        return "int16_t";
    case I32:
        return "int32_t";
    case I64:
        return "int64_t";
    case F16:
        return "float16";
    case F32:
        return "float";
    case F64:
        return "double";
    }
    return "?";
}

static void print_struct(const DataType &dtype) {
    std::cout << "struct(\n";
    for (const auto &[name, dtype] : dtype.fields()) {
        std::cout << "    " << name << " = ";
        print_dtype(dtype);
    }
    std::cout << ")";
}

static void print_dtype(const DataType &dtype) {
    using enum DataTypeVariant;
    switch (dtype.variant()) {
    case Null:
        std::cout << "null";
        break;
    case Bool:
        std::cout << "bool";
        break;
    case Utf8:
        std::cout << "utf8";
        break;
    case Binary:
        std::cout << "binary";
        break;
    case Extension:
        std::cout << "extension";
        break;
    case Primitive:
        std::cout << "primitive(" << ptype_name(dtype.primitive_type()) << ")";
        break;
    case Struct:
        print_struct(dtype);
        break;
    case List:
        std::cout << "list(";
        print_dtype(dtype.list_element());
        std::cout << ")";
        break;
    case FixedSizeList:
        std::cout << "fixed_list(";
        print_dtype(dtype.fixed_size_list_element());
        std::cout << ")";
        break;
    case Decimal:
        std::cout << "decimal(precision=" << static_cast<unsigned>(dtype.decimal_precision())
                  << ", scale=" << static_cast<int>(dtype.decimal_scale()) << ")";
        break;
    }
    std::cout << (dtype.nullable() ? '?' : ' ') << '\n';
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: dtype <file glob>\n";
        return 1;
    }

    Session session;
    DataSource ds = DataSource::open(session, {argv[1]});
    DataType dt = ds.dtype();
    std::cout << "dtype: ";
    print_dtype(dt);
    return 0;
}
