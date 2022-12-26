#include "mod_mpi.hh"
#include "decomp_2d.hh"
#include "mod_parameters.hh"
#include "mod_type.hh"

#include <type_traits>

namespace mod_mpi {
namespace {
std::array<MPI_Datatype, 6> createHaloMPIType(std::array<int, 6> nhalo,
                                              MPI_Datatype oldtype) {
  std::array<MPI_Datatype, 6> halotype;
  // halo comm in the west/east direction
  int bcount = (sz[1] + nhalo[2] + nhalo[3]) * (sz[2] + nhalo[4] + nhalo[5]);
  int bsize = nhalo[0];
  int bstride = sz[0] + nhalo[0] + nhalo[1];
  MPI_Type_vector(bcount, bsize, bstride, oldtype, &halotype[0]);
  MPI_Type_commit(&halotype[0]);
  bsize = nhalo[1];
  MPI_Type_vector(bcount, bsize, bstride, oldtype, &halotype[1]);
  MPI_Type_commit(&halotype[1]);

  // halo comm in the south/north direction
  bcount = (sz[0] + nhalo[0] + nhalo[1]) * (sz[2] + nhalo[4] + nhalo[5]) * nhalo[2];
  bsize = 1;
  bstride = 1;
  MPI_Type_vector(bcount, bsize, bstride, oldtype, &halotype[2]);
  MPI_Type_commit(&halotype[2]);
  bcount = (sz[0] + nhalo[0] + nhalo[1]) * (sz[2] + nhalo[4] + nhalo[5]) * nhalo[3];
  MPI_Type_vector(bcount, bsize, bstride, oldtype, &halotype[3]);
  MPI_Type_commit(&halotype[3]);

  // halo comm in the bottom/top direction
  bcount =
      (sz[0] + nhalo[0] + nhalo[1]) * (sz[1] + nhalo[2] + nhalo[3]) * nhalo[4];
  bsize = 1;
  bstride = 1;
  MPI_Type_vector(bcount, bsize, bstride, oldtype, &halotype[4]);
  MPI_Type_commit(&halotype[4]);
  bcount =
      (sz[0] + nhalo[0] + nhalo[1]) * (sz[1] + nhalo[2] + nhalo[3]) * nhalo[5];
  MPI_Type_vector(bcount, bsize, bstride, oldtype, &halotype[5]);
  MPI_Type_commit(&halotype[5]);
  return halotype;
}
} // namespace

void initMPI() {
  std::array<bool, 3> periodic_bc = {true, true, false};
  decomp_2d_init(nx, ny, nz, p_row, p_col, periodic_bc);

  // staring/ending index and size of data held by current processor
  // x-pencil
  xst = decomp_main.xst;
  xen = decomp_main.xen;
  xsz = decomp_main.xsz;
  // y-pencil
  yst = decomp_main.yst;
  yen = decomp_main.yen;
  ysz = decomp_main.ysz;
  // z-pencil
  zst = decomp_main.zst;
  zen = decomp_main.zen;
  zsz = decomp_main.zsz;

  comm_cart_xpen = decomp_2d_comm_cart_x;
  comm_cart_ypen = decomp_2d_comm_cart_y;
  comm_cart_zpen = decomp_2d_comm_cart_z;

  // Find the MPI ranks of neighboring pencils
  //  first dimension 1=west, 2=east, 3=south, 4=north, 5=bottom, 6=top
  // second dimension 1=x-pencil, 2=y-pencil, 3=z-pencil
  // x-pencil
  neighbor_xyz[0][0] = MPI_PROC_NULL;
  neighbor_xyz[0][1] = MPI_PROC_NULL;
  MPI_Cart_shift(comm_cart_xpen, 0, 1, &neighbor_xyz[0][2],
                 &neighbor_xyz[0][3]);
  MPI_Cart_shift(comm_cart_xpen, 1, 1, &neighbor_xyz[0][4],
                 &neighbor_xyz[0][5]);
  // y-pencil
  MPI_Cart_shift(comm_cart_ypen, 0, 1, &neighbor_xyz[1][0],
                 &neighbor_xyz[1][1]);
  neighbor_xyz[1][2] = MPI_PROC_NULL;
  neighbor_xyz[1][3] = MPI_PROC_NULL;
  MPI_Cart_shift(comm_cart_ypen, 1, 1, &neighbor_xyz[1][4],
                 &neighbor_xyz[1][5]);
  // z-pencil
  MPI_Cart_shift(comm_cart_zpen, 0, 1, &neighbor_xyz[2][0],
                 &neighbor_xyz[2][1]);
  MPI_Cart_shift(comm_cart_zpen, 1, 1, &neighbor_xyz[2][2],
                 &neighbor_xyz[2][3]);
  neighbor_xyz[2][4] = MPI_PROC_NULL;
  neighbor_xyz[2][5] = MPI_PROC_NULL;

  // the coordinary of each process in the x-pencil decomposition
  MPI_Cart_coords(comm_cart_xpen, myrank, 2, coord_xpen.data());
  coord_ypen = coord_xpen;
  coord_zpen = coord_xpen;

  // x-pencil
  st = xst;
  en = xen;
  sz = xsz;
  comm_cart = comm_cart_xpen;
  neighbor = neighbor_xyz[0];
  coord_pen = coord_xpen;

  halotype_vel = createHaloMPIType(Array3DH3::nhalo, MPI_REAL_FP);
  halotype_one = createHaloMPIType(Array3DH1::nhalo, MPI_REAL_FP);
}

void freeMPI() {
  decomp_2d_finalize();
  for (int i = 0; i < 6; ++i) {
    MPI_Type_free(&halotype_vel[i]);
    MPI_Type_free(&halotype_one[i]);
  }
}
} // namespace mod_mpi