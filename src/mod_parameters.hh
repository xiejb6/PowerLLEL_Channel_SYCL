#pragma once

#include "mod_type.hh"

#include <cmath>
#include <string>
#include <string_view>

void readInputParam(std::string_view fn_input);

// Often used constants
constexpr fp pi = M_PI;

// parallel computing parameters
inline int p_row, p_col, nthreads = 1;

// mesh parameters
inline bool read_mesh;
// mesh_type, valid only when read_mesh == false,
// 0: uniform,
// 1: nonuniform, clustered at the lower end,
// 2: nonuniform, clustered at both ends,
// 3: nonuniform, clustered at the middle
inline int mesh_type;
// stretch_ratio, valid only when mesh_type != 0, should not be zero, or the
// mesh becomes uniform.
inline fp stretch_ratio;
inline int stretch_func;

inline int nx, ny, nz;
inline fp lx, ly, lz;

// time step parameters
inline int nt_end;
inline int nt_check;
inline fp dt;
inline fp cfl_limit;
inline fp div_limit;

// restart computing parameters
inline bool is_restart;
inline std::string fn_prefix_input_inst = "save";
inline std::string fn_prefix_input_stat = "save_stat";

// physical property parameters
inline fp re, u_ref, l_ref, re_inv;
inline std::string initial_field;
inline fp u0 = 0.0;
inline bool init_with_noise = false;
inline fp noise_intensity = 0.0;
inline bool smooth_wall_visc = false;
inline fp u_crf = 0.0;

// forced flow parameters
inline std::array<bool, 3> is_forced;
inline std::array<fp, 3> vel_force;

// statistics parameters
inline int nt_init_stat;
inline int sample_interval = 0;
inline std::array<bool, 11> stat_which_var;

// data output parameters
inline int nt_out_scrn;
inline int nt_out_inst;
inline int nt_out_stat;
inline int nt_out_save;
inline bool overwrite_save = true;
inline bool auto_cleanup = true;
inline int num_retained = 0;
inline int nt_out_moni = 0;
inline std::string fn_prefix_inst = "inst";
inline std::string fn_prefix_save = "save";
inline std::string fn_prefix_stat = "stat";

// monitor parameters
inline bool out_forcing;
inline bool out_probe_point;
inline std::array<int, 3> probe_ijk;
inline std::array<bool, 2> out_skf_z;
inline bool out_region = false;
inline std::array<int, 6> region_ijk;
inline int nt_out_region;
inline std::string fn_forcing = "monitor_forcing.out";
inline std::string fn_probe = "monitor_probe.out";
inline std::array<std::string, 2> fn_skf_z = {"monitor_skf_z1.out",
                                              "monitor_skf_z2.out"};
inline std::string fn_prefix_region = "monitor_region";
