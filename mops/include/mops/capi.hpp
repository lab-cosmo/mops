#ifndef MOPS_CAPI_HELPERS_HPP
#define MOPS_CAPI_HELPERS_HPP

#include <string>

#include "mops/exports.h"

namespace mops {
void store_error_message(std::string message);
MOPS_EXPORT const std::string& get_last_error_message();
} // namespace mops

#define MOPS_CATCH_EXCEPTIONS_BEGIN try {

#define MOPS_CATCH_EXCEPTIONS_END                                                                  \
    return 0;                                                                                      \
    }                                                                                              \
    catch (const std::exception& e) {                                                              \
        mops::store_error_message(e.what());                                                       \
        return 1;                                                                                  \
    }                                                                                              \
    catch (...) {                                                                                  \
        mops::store_error_message("unknown error");                                                \
        return 2;                                                                                  \
    }

#endif
