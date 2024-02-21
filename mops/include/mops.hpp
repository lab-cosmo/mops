#ifndef MOPS_HPP
#define MOPS_HPP

#include "mops/exports.h" // IWYU pragma: export

#include "mops/capi.hpp"  // IWYU pragma: export
#include "mops/hpe.hpp"   // IWYU pragma: export
#include "mops/opsa.hpp"  // IWYU pragma: export
#include "mops/opsaw.hpp" // IWYU pragma: export
#include "mops/sap.hpp"   // IWYU pragma: export
#include "mops/sasaw.hpp" // IWYU pragma: export

#ifdef MOPS_CUDA_ENABLED
#include "mops/cuda_first_occurences.hpp"
#include "mops/cuda_opsa.hpp"
#endif

#endif
