#include "mod_hdf5.hh"
#include "mod_type.hh"

#include <H5FDmpi.h>
#include <array>
// #include <bits/stdint-intn.h>
#include <cstddef>
#include <fstream>
#include <type_traits>

#include <fmt/core.h>

namespace mod_hdf5 {
namespace {
const auto h5_real =
    std::is_same_v<fp, double> ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
constexpr int xfer_size_limit = 2147483647;
constexpr int xfer_size_batch = 1073741824;
constexpr int64_t div_ceiling(int64_t val, int64_t upper) {
  return (val + upper - 1) / upper;
}
} // namespace

using mod_mpi::myrank;

void initIO(MPI_Comm comm_in) { H5open(); }

void freeIO() { H5close(); }

hid_t createFile(std::string_view filename) {
  // Setup file access property list with parallel I/O access
  hid_t plistid = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plistid, MPI_COMM_WORLD, MPI_INFO_NULL);

  // Create the file collectively
  hid_t fileid =
      H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, plistid);
  H5Pclose(plistid);
  return fileid;
}

hid_t openFile(std::string_view filename) {
  do {
    std::ifstream file(filename.data());
    if (!file) {
      if (myrank == 0)
        fmt::print("PowerLLEL.ERROR.openFile: {} doesn't exist!\n", filename);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  } while (0);

  // Setup file access property list with parallel I/O access
  hid_t plistid = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plistid, MPI_COMM_WORLD, MPI_INFO_NULL);

  // Open an existing file
  hid_t fileid = H5Fopen(filename.data(), H5F_ACC_RDONLY, plistid);
  H5Pclose(plistid);
  return fileid;
}

void closeFile(hid_t fileid) { H5Fclose(fileid); }

void readAttribute(hid_t fileid, std::string_view tag, int &var) {
  // Create property list for independent dataset read
  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
  // Open an existing dataset
  hid_t dsetid = H5Dopen(fileid, tag.data(), H5P_DEFAULT);
  // Read the dataset independently.
  H5Dread(dsetid, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, plistid, &var);
  // Close the dataset
  H5Dclose(dsetid);
  // Close the property list
  H5Pclose(plistid);
}

void readAttribute(hid_t fileid, std::string_view tag, fp &var) {
  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
  hid_t dsetid = H5Dopen(fileid, tag.data(), H5P_DEFAULT);
  H5Dread(dsetid, h5_real, H5S_ALL, H5S_ALL, plistid, &var);
  H5Dclose(dsetid);
  H5Pclose(plistid);
}

void writeAttribute(hid_t fileid, std::string_view tag, int var) {
  // create the data space for the dataset
  hsize_t dims_1d = 1;
  hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
  // create the dataset with default properties
  hid_t dsetid = H5Dcreate(fileid, tag.data(), H5T_NATIVE_INT, filespace,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Create property list for independent dataset write
  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
  // write the dataset
  if (myrank == 0) {
    H5Dwrite(dsetid, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, plistid, &var);
  }
  // close the property list
  H5Pclose(plistid);
  // close the dataset
  H5Dclose(dsetid);
  // terminate access to the dataspace
  H5Sclose(filespace);
}

void writeAttribute(hid_t fileid, std::string_view tag, fp var) {
  hsize_t dims_1d = 1;
  hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
  hid_t dsetid = H5Dcreate(fileid, tag.data(), h5_real, filespace, H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);
  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
  if (myrank == 0) {
    H5Dwrite(dsetid, h5_real, H5S_ALL, H5S_ALL, plistid, &var);
  }
  H5Pclose(plistid);
  H5Dclose(dsetid);
  H5Sclose(filespace);
}

void write1d(hid_t fileid, std::string_view tag, const Array1DH1 &var) {
  hsize_t dims_1d = var.size();
  hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
  hid_t dsetid = H5Dcreate(fileid, tag.data(), h5_real, filespace, H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);
  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
  if (myrank == 0) {
    H5Dwrite(dsetid, h5_real, H5S_ALL, H5S_ALL, plistid, &var(1));
  }
  H5Pclose(plistid);
  H5Dclose(dsetid);
  H5Sclose(filespace);
}

void write1d(hid_t fileid, std::string_view tag, const Array1DH1 &var,
             bool is_involved) {
  hsize_t dims_1d = var.size();
  hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
  hid_t dsetid = H5Dcreate(fileid, tag.data(), h5_real, filespace, H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);
  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
  if (is_involved) {
    H5Dwrite(dsetid, h5_real, H5S_ALL, H5S_ALL, plistid, &var(1));
  }
  H5Pclose(plistid);
  H5Dclose(dsetid);
  H5Sclose(filespace);
}

void write1d(hid_t fileid, std::string_view tag, const Array1DH1 &var,
             bool is_involved, int sz_global, int st) {
  hsize_t dims_1d = sz_global;
  hid_t filespace = H5Screate_simple(1, &dims_1d, NULL);
  hid_t dsetid = H5Dcreate(fileid, tag.data(), h5_real, filespace, H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);
  dims_1d = var.size();
  hid_t memspace = H5Screate_simple(1, &dims_1d, NULL);
  hsize_t offset = st - 1;
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, &offset, NULL, &dims_1d, NULL);

  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_INDEPENDENT);
  if (is_involved) {
    H5Dwrite(dsetid, h5_real, memspace, filespace, plistid, &var(1));
  }
  H5Pclose(plistid);
  H5Dclose(dsetid);
  H5Sclose(filespace);
  H5Sclose(memspace);
}

template <typename T>
void read3d(hid_t fileid, std::string_view tag, std::array<int, 3> st, T &var) {
  const auto &sz = var.size();
  // Open an existing dataset
  hid_t dsetid = H5Dopen(fileid, tag.data(), H5P_DEFAULT);
  // Get the data space for the whole dataset
  hid_t filespace = H5Dget_space(dsetid);

  std::array<hsize_t, 3> dims_3d;
  dims_3d[0] = sz[2] + var.nhalo[4] + var.nhalo[5];
  dims_3d[1] = sz[1] + var.nhalo[2] + var.nhalo[3];
  dims_3d[2] = sz[0] + var.nhalo[0] + var.nhalo[1];
  hid_t memspace = H5Screate_simple(3, dims_3d.data(), NULL);

  int64_t xfer_size = sz[0] * sz[1] * sz[2] * sizeof(fp);
  int nbatch = 1;
  hsize_t batch_size = sz[2];
  if (xfer_size > xfer_size_limit) {
    nbatch = div_ceiling(xfer_size, xfer_size_batch);
    batch_size = div_ceiling(sz[2], nbatch);
  }

  std::array<hsize_t, 3> blocksize;
  blocksize[0] = batch_size;
  blocksize[1] = sz[1];
  blocksize[2] = sz[0];

  // Create property list for independent/collective dataset read
  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_COLLECTIVE);

  std::array<hsize_t, 3> offset;
  offset[0] = var.nhalo[4];
  offset[1] = var.nhalo[2];
  offset[2] = var.nhalo[0];
  std::array<hsize_t, 3> offset_f;
  offset_f[0] = st[2] - 1;
  offset_f[1] = st[1] - 1;
  offset_f[2] = st[0] - 1;

  if (nbatch == 1) {
    // Select hyperslab in the memory
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset.data(), NULL,
                        blocksize.data(), NULL);

    // Select hyperslab in the file
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_f.data(), NULL,
                        blocksize.data(), NULL);

    // Read the dataset
    H5Dread(dsetid, h5_real, memspace, filespace, plistid, var.data());
  } else {
    for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
      // Reset the memspace & filespace size of the last batch
      if (ibatch == nbatch - 1) {
        blocksize[0] = sz[2] - ibatch * batch_size;
      }

      H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset.data(), NULL,
                          blocksize.data(), NULL);
      offset[0] += batch_size;

      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_f.data(), NULL,
                          blocksize.data(), NULL);
      offset_f[0] += batch_size;

      H5Dread(dsetid, h5_real, memspace, filespace, plistid, var.data());
    }
  }

  // Close the property list
  H5Pclose(plistid);
  // Close the dataset
  H5Dclose(dsetid);
  // Close dataspaces
  H5Sclose(filespace);
  H5Sclose(memspace);
}

template <typename T>
void write3d(hid_t fileid, std::string_view tag, std::array<int, 3> sz_global,
             std::array<int, 3> st, const T &var) {
  const auto &sz = var.size();
  // Create the data space for the whole dataset
  std::array<hsize_t, 3> dims_3d;
  dims_3d[0] = sz_global[2];
  dims_3d[1] = sz_global[1];
  dims_3d[2] = sz_global[0];
  hid_t filespace = H5Screate_simple(3, dims_3d.data(), NULL);

  dims_3d[0] = sz[2] + var.nhalo[4] + var.nhalo[5];
  dims_3d[1] = sz[1] + var.nhalo[2] + var.nhalo[3];
  dims_3d[2] = sz[0] + var.nhalo[0] + var.nhalo[1];
  hid_t memspace = H5Screate_simple(3, dims_3d.data(), NULL);

  int64_t xfer_size = sz[0] * sz[1] * sz[2] * sizeof(fp);
  int nbatch = 1;
  hsize_t batch_size = sz[2];
  if (xfer_size > xfer_size_limit) {
    nbatch = div_ceiling(xfer_size, xfer_size_batch);
    batch_size = div_ceiling(sz[2], nbatch);
  }

  std::array<hsize_t, 3> blocksize;
  blocksize[0] = batch_size;
  blocksize[1] = sz[1];
  blocksize[2] = sz[0];

  // Create datasets with default properties
  hid_t dsetid = H5Dcreate(fileid, tag.data(), h5_real, filespace, H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);
  // Create property list for independent/collective dataset write
  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_COLLECTIVE);

  std::array<hsize_t, 3> offset;
  offset[0] = var.nhalo[4];
  offset[1] = var.nhalo[2];
  offset[2] = var.nhalo[0];
  std::array<hsize_t, 3> offset_f;
  offset_f[0] = st[2] - 1;
  offset_f[1] = st[1] - 1;
  offset_f[2] = st[0] - 1;

  if (nbatch == 1) {
    // Select hyperslab in the memory
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset.data(), NULL,
                        blocksize.data(), NULL);

    // Select hyperslab in the file
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_f.data(), NULL,
                        blocksize.data(), NULL);

    // Write the dataset
    H5Dwrite(dsetid, h5_real, memspace, filespace, plistid, var.data());
  } else {
    for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
      // Reset the memspace & filespace size of the last batch
      if (ibatch == nbatch - 1) {
        blocksize[0] = sz[2] - ibatch * batch_size;
      }

      H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset.data(), NULL,
                          blocksize.data(), NULL);
      offset[0] += batch_size;

      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_f.data(), NULL,
                          blocksize.data(), NULL);
      offset_f[0] += batch_size;

      H5Dwrite(dsetid, h5_real, memspace, filespace, plistid, var.data());
    }
  }

  // Close the property list
  H5Pclose(plistid);
  // Close the dataset
  H5Dclose(dsetid);
  // Close dataspaces
  H5Sclose(filespace);
  H5Sclose(memspace);
}

template <typename T>
void write3d(hid_t fileid, std::string_view tag, std::array<int, 3> sz_global,
             std::array<int, 3> st, const T &var, bool is_involved) {
  const auto &sz = var.size();
  // Create the data space for the whole dataset
  std::array<hsize_t, 3> dims_3d;
  dims_3d[0] = sz_global[2];
  dims_3d[1] = sz_global[1];
  dims_3d[2] = sz_global[0];
  hid_t filespace = H5Screate_simple(3, dims_3d.data(), NULL);

  dims_3d[0] = sz[2] + var.nhalo[4] + var.nhalo[5];
  dims_3d[1] = sz[1] + var.nhalo[2] + var.nhalo[3];
  dims_3d[2] = sz[0] + var.nhalo[0] + var.nhalo[1];
  hid_t memspace = H5Screate_simple(3, dims_3d.data(), NULL);

  int64_t xfer_size = sz[0] * sz[1] * sz[2] * sizeof(fp);
  int nbatch = 1;
  hsize_t batch_size = sz[2];
  if (xfer_size > xfer_size_limit) {
    nbatch = div_ceiling(xfer_size, xfer_size_batch);
    batch_size = div_ceiling(sz[2], nbatch);
  }

  std::array<hsize_t, 3> blocksize;
  blocksize[0] = batch_size;
  blocksize[1] = sz[1];
  blocksize[2] = sz[0];

  // Create datasets with default properties
  hid_t dsetid = H5Dcreate(fileid, tag.data(), h5_real, filespace, H5P_DEFAULT,
                           H5P_DEFAULT, H5P_DEFAULT);
  // Create property list for independent/collective dataset write
  hid_t plistid = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plistid, H5FD_MPIO_COLLECTIVE);

  std::array<hsize_t, 3> offset;
  offset[0] = var.nhalo[4];
  offset[1] = var.nhalo[2];
  offset[2] = var.nhalo[0];
  std::array<hsize_t, 3> offset_f;
  offset_f[0] = st[2] - 1;
  offset_f[1] = st[1] - 1;
  offset_f[2] = st[0] - 1;

  if (nbatch == 1) {
    // Select hyperslab in the memory
    H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset.data(), NULL,
                        blocksize.data(), NULL);

    // Select hyperslab in the file
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_f.data(), NULL,
                        blocksize.data(), NULL);

    // Write the dataset
    if (is_involved) {
      H5Dwrite(dsetid, h5_real, memspace, filespace, plistid, var.data());
    }
  } else {
    for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
      // Reset the memspace & filespace size of the last batch
      if (ibatch == nbatch - 1) {
        blocksize[0] = sz[2] - ibatch * batch_size;
      }

      H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset.data(), NULL,
                          blocksize.data(), NULL);
      offset[0] += batch_size;

      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_f.data(), NULL,
                          blocksize.data(), NULL);
      offset_f[0] += batch_size;

      if (is_involved) {
        H5Dwrite(dsetid, h5_real, memspace, filespace, plistid, var.data());
      }
    }
  }

  // Close the property list
  H5Pclose(plistid);
  // Close the dataset
  H5Dclose(dsetid);
  // Close dataspaces
  H5Sclose(filespace);
  H5Sclose(memspace);
}

template void read3d(hid_t fileid, std::string_view tag, std::array<int, 3> st,
                     Array3DH3 &var);
template void read3d(hid_t fileid, std::string_view tag, std::array<int, 3> st,
                     Array3DH0 &var);
// template void write3d(hid_t fileid, std::string_view tag,
//                       std::array<int, 3> sz_global, std::array<int, 3> st,
//                       const Array3DH3 &var);
template void write3d(hid_t fileid, std::string_view tag,
                      std::array<int, 3> sz_global, std::array<int, 3> st,
                      const Array3DH1 &var);
template void write3d(hid_t fileid, std::string_view tag,
                      std::array<int, 3> sz_global, std::array<int, 3> st,
                      const Array3DH0 &var);
// template void write3d(hid_t fileid, std::string_view tag,
//                       std::array<int, 3> sz_global, std::array<int, 3> st,
//                       const Array3DH3 &var, bool is_involved);
template void write3d(hid_t fileid, std::string_view tag,
                      std::array<int, 3> sz_global, std::array<int, 3> st,
                      const Array3DH1 &var, bool is_involved);
template void write3d(hid_t fileid, std::string_view tag,
                      std::array<int, 3> sz_global, std::array<int, 3> st,
                      const Array3DH0 &var, bool is_involved);
} // namespace mod_hdf5