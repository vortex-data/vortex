// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <memory>

namespace vortex {

class Expr {
public:
    Expr() = delete;
    Expr(Expr &&other) noexcept;
    Expr &operator=(Expr &&other) noexcept;
    ~Expr();

    Expr(const Expr &) = delete;
    Expr &operator=(const Expr &) = delete;

    static Expr get_item(std::string field, Expr expr);

private:
    struct Impl;
    explicit Expr(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace vortex