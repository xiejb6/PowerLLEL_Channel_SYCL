#pragma once

#include "mod_type.hh"

namespace mod_monitor {
void initMonitor();
void outputMonitor(int nt, fp u_crf, const Array3DH3 &u, const Array3DH3 &v,
                   const Array3DH3 &w, const Array3DH1 &p);
void freeMonitor();
} // namespace mod_monitor
