#ifndef MOPS_H
#define MOPS_H

#include "mops/exports.h"     // IWYU pragma: export
#include "mops/opsa.h"        // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif

/// TODO
const char* MOPS_EXPORT mops_get_last_error_message();

#ifdef __cplusplus
}
#endif

#endif
