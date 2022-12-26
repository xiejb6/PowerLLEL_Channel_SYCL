#include "mod_parameters.hh"
#include "mod_mpi.hh"

#include <exception>
#include <fstream>

#include <fmt/core.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

namespace {

template <typename T, typename T2>
constexpr void get(const T2 &node, T &value) {
  if constexpr (std::is_same_v<T, int>) {
    value = node.GetInt();
  } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
    value = node.GetDouble();
  } else if constexpr (std::is_same_v<T, bool>) {
    value = node.GetBool();
  } else if constexpr (std::is_same_v<T, std::string>) {
    value = std::string(node.GetString(), node.GetStringLength());
  }
}

template <typename T, typename T2>
void _read(const T2 &node, const char *key, T &val) {
  auto it = node.FindMember(key);
  if (it != node.MemberEnd()) {
    auto &node = it->value;
    get(node, val);
  }
}

template <typename T, size_t N, typename T2>
void _read(const T2 &node, const char *key, std::array<T, N> &val) {
  auto it = node.FindMember(key);
  if (it != node.MemberEnd()) {
    auto &array = it->value;
    for (rapidjson::SizeType i = 0; i < N; ++i) {
      get(array[i], val[i]);
    }
  }
}

#define read(node, val) _read(node, #val, val)

void readPara(const rapidjson::Document &d) {
  auto it = d.FindMember("PARA");
  if (it == d.MemberEnd())
    throw "PowerLLEL.ERROR.readInputParam: Problem with PARA in the input "
          "file!";
  auto &node = it->value;
  read(node, p_row);
  read(node, p_col);
}

void readMesh(const rapidjson::Document &d) {
  auto it = d.FindMember("MESH");
  if (it == d.MemberEnd())
    throw "PowerLLEL.ERROR.readInputParam: Problem with MESH in the input "
          "file!";
  auto &node = it->value;
  read(node, read_mesh);
  read(node, mesh_type);
  read(node, stretch_func);
  read(node, stretch_ratio);
  read(node, nx);
  read(node, ny);
  read(node, nz);
  read(node, lx);
  read(node, ly);
  read(node, lz);
}

void readTime(const rapidjson::Document &d) {
  auto it = d.FindMember("TIME");
  if (it == d.MemberEnd())
    throw "PowerLLEL.ERROR.readInputParam: Problem with TIME in the input "
          "file!";
  auto &node = it->value;
  read(node, nt_end);
  read(node, dt);
  read(node, nt_check);
  read(node, cfl_limit);
  read(node, div_limit);
}

void readRestart(const rapidjson::Document &d) {
  auto it = d.FindMember("RESTART");
  if (it == d.MemberEnd())
    throw "PowerLLEL.ERROR.readInputParam: Problem with RESTART in the input "
          "file!";
  auto &node = it->value;
  read(node, is_restart);
  read(node, fn_prefix_input_inst);
  read(node, fn_prefix_input_stat);
}

void readPhysical(const rapidjson::Document &d) {
  auto it = d.FindMember("PHYSICAL");
  if (it == d.MemberEnd())
    throw "PowerLLEL.ERROR.readInputParam: Problem with PHYSICAL in the input "
          "file!";
  auto &node = it->value;
  read(node, re);
  read(node, u_ref);
  read(node, l_ref);
  read(node, initial_field);
  read(node, u0);
  read(node, init_with_noise);
  read(node, noise_intensity);
  read(node, smooth_wall_visc);
  read(node, u_crf);
}

void readForce(const rapidjson::Document &d) {
  auto it = d.FindMember("FORCE");
  if (it == d.MemberEnd())
    throw "PowerLLEL.ERROR.readInputParam: Problem with FORCE in the input "
          "file!";
  auto &node = it->value;
  read(node, is_forced);
  read(node, vel_force);
}

void readStatistics(const rapidjson::Document &d) {
  auto it = d.FindMember("STATISTICS");
  if (it == d.MemberEnd())
    throw "PowerLLEL.ERROR.readInputParam: Problem with STATISTICS in the "
          "input "
          "file!";
  auto &node = it->value;
  read(node, nt_init_stat);
  read(node, sample_interval);
  read(node, stat_which_var);
}

void readOutput(const rapidjson::Document &d) {
  auto it = d.FindMember("OUTPUT");
  if (it == d.MemberEnd())
    throw "PowerLLEL.ERROR.readInputParam: Problem with OUTPUT in the input "
          "file!";
  auto &node = it->value;
  read(node, nt_out_scrn);
  read(node, nt_out_inst);
  read(node, nt_out_stat);
  read(node, nt_out_save);
  read(node, overwrite_save);
  read(node, auto_cleanup);
  read(node, num_retained);
  read(node, nt_out_moni);
}

void readMonitor(const rapidjson::Document &d) {
  auto it = d.FindMember("MONITOR");
  if (it == d.MemberEnd())
    throw "PowerLLEL.ERROR.readInputParam: Problem with MONITOR in the input "
          "file!";
  auto &node = it->value;
  read(node, out_forcing);
  read(node, out_probe_point);
  read(node, probe_ijk);
  read(node, out_skf_z);
  read(node, out_region);
  read(node, nt_out_region);
  read(node, region_ijk);
}

bool checkMeshAndPara() {
  bool subcheck_passed = true;
  bool passed = true;

  passed = (nx % 2 == 0 && ny % 2 == 0 && nz % 2 == 0);
  if (!passed && mod_mpi::myrank == 0)
    fmt::print("PowerLLEL.ERROR.checkInputParam: <nx>, <ny>, <nz> should be "
               "even integers!\n");
  subcheck_passed &= passed;

  passed = (nx % p_row == 0);
  if (!passed && mod_mpi::myrank == 0)
    fmt::print("PowerLLEL.ERROR.checkInputParam: <nx> should be exactly "
               "divisible by <p_row>!\n");
  subcheck_passed &= passed;

#ifdef _PDD
  passed = (ny % p_row == 0);
  if (!passed && mod_mpi::myrank == 0)
    fmt::print("PowerLLEL.ERROR.checkInputParam: <ny> should be exactly "
               "divisible by <p_row>!\n");
#else
  passed = (ny % p_row == 0 && ny % p_col == 0);
  if (!passed && mod_mpi::myrank == 0)
    fmt::print("PowerLLEL.ERROR.checkInputParam: <ny> should be exactly "
               "divisible by both <p_row> and <p_col>!\n");
#endif
  subcheck_passed &= passed;

  passed = (nz % p_col == 0);
  if (!passed && mod_mpi::myrank == 0)
    fmt::print("PowerLLEL.ERROR.checkInputParam: <nz> should be exactly "
               "divisible by <p_col>!\n");
  subcheck_passed &= passed;

  passed = (nz % (p_col * nthreads) == 0);
  if (!passed && mod_mpi::myrank == 0)
    fmt::print("PowerLLEL.ERROR.checkInputParam: <nz> should be exactly "
               "divisible by <p_col*nthreads>!\n");
  subcheck_passed &= passed;
  return subcheck_passed;
}

void checkInputParam() {
  bool check_passed = checkMeshAndPara();
  if (!check_passed)
    MPI_Abort(MPI_COMM_WORLD, -1);
}
} // namespace

void readInputParam(std::string_view fn_input) {
  bool alive;
  int ios;
  int id_input = 10;

  rapidjson::Document d;
  do {
    std::ifstream ifs(fn_input.data());
    if (!ifs) {
      if (mod_mpi::myrank == 0) {
        fmt::print("PowerLLEL.ERROR.readInputParam: Fail to open file {}!\n",
                   fn_input);
      }
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    rapidjson::IStreamWrapper isw(ifs);
    d.ParseStream(isw);
  } while (0);

  try {
    readPara(d);
    readMesh(d);
    readTime(d);
    readRestart(d);
    readPhysical(d);
    readForce(d);
    readStatistics(d);
    readOutput(d);
    readMonitor(d);
  } catch (const std::exception &e) {
    if (mod_mpi::myrank == 0) {
      fmt::print("{}\n", e.what());
    }
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  // do a sanity check on input parameters
  checkInputParam();

  re_inv = u_ref * l_ref / re;
}