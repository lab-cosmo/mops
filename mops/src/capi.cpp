#include "mops/capi.hpp"
#include "mops.h"

thread_local std::string LAST_ERROR_MESSAGE;

void mops::store_error_message(std::string message) { LAST_ERROR_MESSAGE = std::move(message); }

const std::string& mops::get_last_error_message() { return LAST_ERROR_MESSAGE; }

extern "C" const char* mops_get_last_error_message() {
    return mops::get_last_error_message().c_str();
}
