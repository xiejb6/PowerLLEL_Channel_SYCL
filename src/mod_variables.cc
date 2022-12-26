#include "mod_variables.hh"

namespace mod_variables {
void allocVariables(std::array<int, 3> sz) {
  u.allocate(sz);
  v.allocate(sz);
  w.allocate(sz);
  u1.allocate(sz);
  v1.allocate(sz);
  w1.allocate(sz);
  p.allocate(sz);
}

void freeVariables() {
  u.deallocate();
  v.deallocate();
  w.deallocate();
  u1.deallocate();
  v1.deallocate();
  w1.deallocate();
  p.deallocate();
}
} // namespace mod_variables