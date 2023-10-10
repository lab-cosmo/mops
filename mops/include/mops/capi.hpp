#ifndef MOPS_CAPI_HELPERS_HPP
#define MOPS_CAPI_HELPERS_HPP

#include <stdexcept>
#include <string>

#include "mops/exports.h"

namespace mops {
    void MOPS_EXPORT store_error_message(std::string message);
    const std::string& MOPS_EXPORT get_last_error_message();
}

#define MOPS_CATCH_EXCEPTIONS(__code__)                                                     \
    do {                                                                                    \
        try { __code__; return 0; }                                                         \
        catch (const std::exception& e) { mops::store_error_message(e.what()); return 1; }  \
        catch (...) { mops::store_error_message("unknown error"); return 2; }               \
    } while (false)

#endif
