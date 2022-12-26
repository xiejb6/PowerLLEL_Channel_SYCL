#pragma once

#include "mod_mpi.hh"
#include "mod_type.hh"

#include <hdf5.h>

namespace mod_hdf5 {
void initIO(MPI_Comm comm_in);
void freeIO();
hid_t createFile(std::string_view filename);
hid_t openFile(std::string_view filename);
void closeFile(hid_t fileid);
void readAttribute(hid_t fileid, std::string_view tag, int &var);
void readAttribute(hid_t fileid, std::string_view tag, fp &var);
void writeAttribute(hid_t fileid, std::string_view tag, int var);
void writeAttribute(hid_t fileid, std::string_view tag, fp var);
void write1d(hid_t fileid, std::string_view tag, const Array1DH1 &var);
void write1d(hid_t fileid, std::string_view tag, const Array1DH1 &var,
             bool is_involved);
void write1d(hid_t fileid, std::string_view tag, const Array1DH1 &var,
             bool is_involved, int sz_global, int st);
template <typename T>
void read3d(hid_t fileid, std::string_view tag, std::array<int, 3> st, T &var);
template <typename T>
void write3d(hid_t fileid, std::string_view tag, std::array<int, 3> sz_global,
             std::array<int, 3> st, const T &);
template <typename T>
void write3d(hid_t fileid, std::string_view tag, std::array<int, 3> sz_global,
             std::array<int, 3> st, const T &var, bool is_involved);
} // namespace mod_hdf5
