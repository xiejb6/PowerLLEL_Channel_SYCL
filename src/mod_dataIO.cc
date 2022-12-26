#include "mod_dataIO.hh"
#include "mod_hdf5.hh"
#include "mod_monitor.hh"
#include "mod_mpi.hh"
#include "mod_parameters.hh"
#include "mod_statistics.hh"
#include "mpi.h"

#include <cstdio>
#include <fmt/core.h>
#include <string_view>

using mod_mpi::myrank;
using namespace mod_statistics;

namespace mod_dataIO {
namespace {
void checkTimestamp(int base, int actual, std::string_view vartag) {
  if (base != actual) {
    if (myrank == 0) {
      fmt::print("PowerLLEL.ERROR.inputData: The timestamp of <{}> does not "
                 "match with that of <u>!\n",
                 vartag);
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}
void checkStatTimestamp(int base, int actual, std::string_view vartag) {
  if (base != actual) {
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.ERROR.inputStatData: The timestamp of <{}> does not "
          "match with that of inst. field!\n",
          vartag);
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
}
constexpr auto get_stat_type(int i) {
  switch (i) {
  case 0:
    return "u";
  case 1:
    return "v";
  case 2:
    return "w";
  case 3:
    return "u2";
  case 4:
    return "v2";
  case 5:
    return "w2";
  case 6:
    return "uv";
  case 7:
    return "uw";
  case 8:
    return "vw";
  case 9:
    return "p";
  case 10:
    return "p2";
  default:
    __builtin_unreachable();
  }
};

template <typename T>
int inputField(std::string_view fn_field, std::string_view vartag, T &var,
               StatInfo *stat_info = nullptr) {
  int nt;
  auto fh = mod_hdf5::openFile(fn_field);
  if (stat_info == nullptr) {
    mod_hdf5::readAttribute(fh, "nt", nt);
  } else {
    mod_hdf5::readAttribute(fh, "nts", stat_info->nts);
    mod_hdf5::readAttribute(fh, "nte", stat_info->nte);
    mod_hdf5::readAttribute(fh, "nspl", stat_info->nspl);
    nt = stat_info->nte;
  }
  mod_hdf5::read3d(fh, vartag, mod_mpi::st, var);
  mod_hdf5::closeFile(fh);
  return nt;
}

template <typename T>
int outputField(std::string_view fn_field, std::string_view vartag, int nt,
                const T &var, const StatInfo *stat_info = nullptr) {
  auto fh = mod_hdf5::createFile(fn_field);
  if (stat_info == nullptr) {
    mod_hdf5::writeAttribute(fh, "nt", nt);
  } else {
    mod_hdf5::writeAttribute(fh, "nts", stat_info->nts);
    mod_hdf5::writeAttribute(fh, "nte", stat_info->nte);
    mod_hdf5::writeAttribute(fh, "nspl", stat_info->nspl);
    nt = stat_info->nte;
  }
  mod_hdf5::write3d(fh, vartag, {nx, ny, nz}, mod_mpi::st, var);
  mod_hdf5::closeFile(fh);
  return nt;
}
} // namespace

int inputData(Array3DH3 &u, Array3DH3 &v, Array3DH3 &w) {
  if (myrank == 0) {
    fmt::print("PowerLLEL.NOTE.inputData: Reading checkpoint fields ...\n");
  }
  int nt_last =
      inputField(fmt::format("{}_u.h5", fn_prefix_input_inst), "u", u);
  if (myrank == 0) {
    fmt::print(
        "PowerLLEL.NOTE.inputData: Finish reading checkpoint field <u>!\n");
  }
  int nt = nt_last;
  nt_last = inputField(fmt::format("{}_v.h5", fn_prefix_input_inst), "v", v);
  if (myrank == 0) {
    fmt::print(
        "PowerLLEL.NOTE.inputData: Finish reading checkpoint field <v>!\n");
  }
  checkTimestamp(nt, nt_last, "v");
  nt_last = inputField(fmt::format("{}_w.h5", fn_prefix_input_inst), "w", w);
  if (myrank == 0) {
    fmt::print(
        "PowerLLEL.NOTE.inputData: Finish reading checkpoint field <w>!\n");
  }
  checkTimestamp(nt, nt_last, "w");
  return nt_last;
}

void outputData(int nt, fp u_crf, Array3DH3 &u, const Array3DH3 &v,
                const Array3DH3 &w, const Array3DH1 &p) {
  if (nt % nt_out_moni == 0) {
    mod_monitor::outputMonitor(nt, u_crf, u, v, w, p);
  }

  if (nt % nt_out_inst == 0) {
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE.outputData: Writing instantaneous fields ...\n");
    }
    auto path_prefix = fmt::format("{}_{:08}_", fn_prefix_inst, nt);
    if (u_crf == 0) {
      outputField(fmt::format("{}u.h5", path_prefix), "u", nt, u);
    } else {
      u += u_crf;
      outputField(fmt::format("{}u.h5", path_prefix), "u", nt, u);
      u -= u_crf;
    }
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE.outputData: Finish writing inst. field <u>!\n");
    }
    outputField(fmt::format("{}v.h5", path_prefix), "v", nt, v);
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE.outputData: Finish writing inst. field <v>!\n");
    }
    outputField(fmt::format("{}w.h5", path_prefix), "w", nt, w);
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE.outputData: Finish writing inst. field <w>!\n");
    }
    outputField(fmt::format("{}p.h5", path_prefix), "p", nt, p);
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE.outputData: Finish writing inst. field <p>!\n");
    }
  }

  if (nt % nt_out_save == 0) {
    if (myrank == 0) {
      fmt::print("PowerLLEL.NOTE.outputData: Writing checkpoint fields ...\n");
    }
    std::string string_dump = "_";
    if (!overwrite_save)
      string_dump = fmt::format("_{:08}_", nt);
    auto path_prefix = fmt::format("{}{}", fn_prefix_save, string_dump);
    if (u_crf == 0) {
      outputField(fmt::format("{}u.h5", path_prefix), "u", nt, u);
    } else {
      u += u_crf;
      outputField(fmt::format("{}u.h5", path_prefix), "u", nt, u);
      u -= u_crf;
    }
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE.outputData: Finish writing checkpoint field <u>!\n");
    }
    outputField(fmt::format("{}v.h5", path_prefix), "v", nt, v);
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE.outputData: Finish writing checkpoint field <v>!\n");
    }
    outputField(fmt::format("{}w.h5", path_prefix), "w", nt, w);
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE.outputData: Finish writing checkpoint field <w>!\n");
    }
  }

  if (nt > nt_init_stat) {
    outputStatData(nt);
  }

  if (myrank == 0) {
    if (!overwrite_save && auto_cleanup && nt % nt_out_save == 0) {
      auto nt_cleanup = nt - (num_retained + 1) * nt_out_save;
      auto path_prefix = fmt::format("{}_{:08}_", fn_prefix_save, nt_cleanup);
      if (nt_cleanup > 0) {
        fmt::print("PowerLLEL.NOTE.outputData: Automatically cleanup of "
                   "inst. checkpoint files ...\n");
        std::remove(fmt::format("{}u.h5", path_prefix).c_str());
        std::remove(fmt::format("{}v.h5", path_prefix).c_str());
        std::remove(fmt::format("{}w.h5", path_prefix).c_str());
      }
      if (nt_cleanup > nt_init_stat) {
        fmt::print("PowerLLEL.NOTE.outputData: Automatically cleanup of "
                   "stat. checkpoint files ...\n");
        auto stat_template = fmt::format("{}stat_{{}}.h5", path_prefix);
        for (int i = 0; i < 11; ++i) {
          if (stat_which_var[i]) {
            std::remove(fmt::format(stat_template, get_stat_type(i)).c_str());
          }
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void inputStatData(int nt_in_inst) {
  if (myrank == 0) {
    fmt::print("PowerLLEL.NOTE.inputStatData: Reading checkpoint fields of "
               "statistics ...\n");
  }
  auto name_template = fmt::format("{}_{{}}.h5", fn_prefix_input_stat);
  for (int i = 0; i < 11; ++i) {
    if (stat_which_var[i]) {
      auto type = get_stat_type(i);
      auto filename = fmt::format(name_template, type);
      auto stat_name = fmt::format("{}_stat", type);
      int nt_last = inputField(filename, stat_name, stat_vec[i], &stat_info);
      if (myrank == 0) {
        fmt::print("PowerLLEL.NOTE.inputStatData: Finish reading checkpoint "
                   "field <{}>!\n",
                   stat_name);
      }
      checkStatTimestamp(nt_in_inst, nt_last, stat_name);
    }
  }
}

void outputStatData(int nt) {
  if (nt % nt_out_save == 0) {
    if (myrank == 0) {
      fmt::print("PowerLLEL.NOTE.outputStatData: Writing checkpoint fields of "
                 "statistics ...\n");
    }
    std::string string_dump = "_";
    if (overwrite_save) {
      string_dump = fmt::format("_{:08}_", nt);
    }
    auto name_template =
        fmt::format("{}{}stat_{{}}.h5", fn_prefix_save, string_dump);
    for (int i = 0; i < 11; ++i) {
      if (stat_which_var[i]) {
        auto type = get_stat_type(i);
        auto filename = fmt::format(name_template, type);
        auto stat_name = fmt::format("{}_stat", type);
        outputField(filename, stat_name, nt, stat_vec[i], &stat_info);
        if (myrank == 0) {
          fmt::print("PowerLLEL.NOTE.outputStatData: Finish writing checkpoint "
                     "field <{}>!\n",
                     stat_name);
        }
      }
    }
  }

  if ((nt - nt_init_stat) % nt_out_stat == 0) {
    if (myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE.outputStatData: Writing statistics fields ...\n");
    }
    auto name_template = fmt::format("{}_{:08}-{:08}_{{}}.h5", fn_prefix_stat,
                                     stat_info.nts, stat_info.nte);
    for (int i = 0; i < 11; ++i) {
      if (stat_which_var[i]) {
        auto type = get_stat_type(i);
        auto filename = fmt::format(name_template, type);
        auto stat_name = fmt::format("{}_stat", type);
        outputField(filename, stat_name, nt, stat_vec[i], &stat_info);
        if (myrank == 0) {
          fmt::print("PowerLLEL.NOTE.outputStatData: Finish writing stat. "
                     "field <{}>!\n",
                     stat_name);
        }
      }
    }
  }
}

} // namespace mod_dataIO