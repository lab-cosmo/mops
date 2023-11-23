#ifndef MOPS_H
#define MOPS_H

#include "mops/exports.h"     // IWYU pragma: export
#include "mops/hpe.h"         // IWYU pragma: export
#include "mops/opsa.h"        // IWYU pragma: export
#include "mops/sap.h"         // IWYU pragma: export
#include "mops/opsaw.h"       // IWYU pragma: export
#include "mops/sasaw.h"       // IWYU pragma: export

#ifdef __cplusplus
extern "C" {
#endif

/// TODO
MOPS_EXPORT const char *mops_get_last_error_message();

#ifdef __cplusplus
}
#endif

#endif
