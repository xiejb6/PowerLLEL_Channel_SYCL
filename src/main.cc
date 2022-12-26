#include "gptl.hh"
#include "mod_calcRHS.hh"
#include "mod_calcVel.hh"
#include "mod_dataIO.hh"
#include "mod_hdf5.hh"
#include "mod_initFlow.hh"
#include "mod_mesh.hh"
#include "mod_monitor.hh"
#include "mod_mpi.hh"
#include "mod_parameters.hh"
#include "mod_poisson_solver.hh"
#include "mod_statistics.hh"
#include "mod_type.hh"
#include "mod_updateBound.hh"
#include "mod_utils.hh"
#include "mod_variables.hh"
#include "mod_oneapi.hh"

#include <fmt/core.h>
#include <fmt/os.h>

using namespace mod_variables;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mod_mpi::myrank);
  if (mod_mpi::myrank == 0) {
    fmt::print("================================================================================\n"
               "                      PowerLLEL_channel (DNS, 2ndFD + RK2)                      \n"
               "================================================================================\n"
               "PowerLLEL.NOTE: Initialization starts ...\n");
  }

  GPTLinitialize();

  // read input parameters from file
  readInputParam("param.json");

  // initialize MPI
  mod_mpi::initMPI();

  // initialize parallel IO
  mod_hdf5::initIO(MPI_COMM_WORLD);

  // initialize the monitoring point
  mod_monitor::initMonitor();

  // allocate variables
  mod_variables::allocVariables(mod_mpi::sz);

  // initialize mesh
  mod_mesh::initMesh();

  // initialize Poisson solver
  init_poisson_solver(nx, ny, nz, mod_mesh::dx, mod_mesh::dy,
                      mod_mesh::dzf_global.data(), "PP", "PP", "NN",
                      mod_mpi::neighbor_xyz);

  // initialize the flow field
  int nt_in, nt_start;
  if (is_restart) {
    if (mod_mpi::myrank == 0) {
      fmt::print(
          "PowerLLEL.NOTE: Initializing flow from checkpoint fields ...\n");
    }
    nt_in = mod_dataIO::inputData(u, v, w);
    nt_start = nt_in + 1;
  } else {
    if (mod_mpi::myrank == 0) {
      fmt::print("PowerLLEL.NOTE: Initializing flow according to input "
                 "parameters ...\n");
    }
    mod_initFlow::initFlow(u, v, w);
    nt_start = 1;
  }

  // important! subtract the convecting reference frame velocity u_crf from u
  mod_calcVel::transform2CRF(u_crf, u, vel_force[0]);

  // update velocity & pressure boundary conditions
  mod_updateBound::updateBoundVel(u_crf, u, v, w);
  mod_updateBound::updateBoundP(p);

  // load the statistics data if necessary
  if (is_restart && nt_start > nt_init_stat + 1) {
    mod_statistics::allocStat(mod_mpi::sz);
    mod_dataIO::inputStatData(nt_start - 1);
  }

  // initialize MPI variables related to the calculation of CFL and Divergence
  mod_utils::value_index_pair_t cfl_max, div_max;
  mod_utils::initCheckCFLAndDiv(cfl_max, div_max);

  if (mod_mpi::myrank == 0) {
    fmt::print("PowerLLEL.NOTE: Initialization ends successfully!\n");
    fmt::print(
        "PowerLLEL.NOTE: Selected device is: {}, {} \n",
        mod_oneapi::queue.get_device().get_platform().get_info<sycl::info::platform::name>(),
        mod_oneapi::queue.get_device().get_info<sycl::info::device::name>());
    fmt::print("PowerLLEL.NOTE: Simulation starts at nt = {:9}!\n", nt_start);
    fmt::print("***************************************************************"
               "*****************\n");
    fmt::print("{:>9}  {:>17}  {:>14}  {:>10}  {:>10}\n", "nt",
               "speed(wSteps/Day)", "remaining time", "cfl_max", "div_max");
  }

  GPTLstart("Main loop");

  fp wtime = MPI_Wtime();

  // ===========================
  //   Main time marching loop
  // ===========================
  for (int nt = nt_start; nt <= nt_end; ++nt) {

    GPTLstart("uvw1");
    mod_calcVel::timeIntVelRK1(u_crf, u, v, w, u1, v1, w1);
    GPTLstop("uvw1");

    GPTLstart("Update boundary vel");
    mod_updateBound::updateBoundVel(u_crf, u1, v1, w1);
    GPTLstop("Update boundary vel");

    GPTLstart("uvw2");
    mod_calcVel::timeIntVelRK2(u_crf, u, v, w, u1, v1, w1);
    GPTLstop("uvw2");

    GPTLstart("Update boundary vel");
    mod_updateBound::updateBoundVel(u_crf, u, v, w);
    GPTLstop("Update boundary vel");

    GPTLstart("Calculate RHS");
    mod_calcRHS::calcRHS(u, v, w, p);
    GPTLstop("Calculate RHS");

    GPTLstart("Poisson solver");
    execute_poisson_solver(p);
    GPTLstop("Poisson solver");

    GPTLstart("Update boundary pres");
    mod_updateBound::updateBoundP(p);
    GPTLstop("Update boundary pres");

    GPTLstart("Correct vel");
    mod_calcVel::correctVel(p, u, v, w);
    GPTLstop("Correct vel");

    GPTLstart("Force vel");
    mod_calcVel::forceVel(u, v, w);
    GPTLstop("Force vel");

    GPTLstart("Update boundary vel");
    mod_updateBound::updateBoundVel(u_crf, u, v, w);
    GPTLstop("Update boundary vel");

    if (nt % nt_check == 0) {
      bool check_passed = true;
      check_passed &= mod_utils::checkNaN(u, "u");
      mod_utils::calcMaxCFL(dt, mod_mesh::dx_inv, mod_mesh::dy_inv,
                            mod_mesh::dzf_inv, u, v, w, cfl_max);
      check_passed &= mod_utils::checkCFL(cfl_limit, cfl_max);
      mod_utils::calcMaxDiv(mod_mesh::dx_inv, mod_mesh::dy_inv,
                            mod_mesh::dzf_inv, u, v, w, div_max);
      check_passed &= mod_utils::checkDiv(div_limit, div_max);
      if (!check_passed) {
        if (mod_mpi::myrank == 0) {
          fmt::print("PowerLLEL.ERROR: nt = {:9}, cfl_max = {:10.3e} at "
                     "({:5},{:5},{:5}) from rank{:5}\n",
                     nt, cfl_max.value, cfl_max.ig, cfl_max.jg, cfl_max.kg,
                     cfl_max.rank);
          fmt::print("PowerLLEL.ERROR: nt = {:9}, div_max = {:10.3e} at "
                     "({:5},{:5},{:5}) from rank{:5}\n",
                     nt, div_max.value, div_max.ig, div_max.jg, div_max.kg,
                     div_max.rank);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }

    if (nt > nt_init_stat) {
      if (nt == nt_init_stat + 1) {
        mod_statistics::allocStat(mod_mpi::sz);
        mod_statistics::initStat();
        if (mod_mpi::myrank == 0) {
          fmt::print(
              "PowerLLEL.NOTE: Statistical process starts at nt = {}!\n");
        }
        mod_statistics::calcStat(nt, u_crf, u, v, w, p);
      }
    }

    if (nt % nt_out_scrn == 0) {
      wtime = MPI_Wtime() - wtime;
      fp wtime_avg;
      MPI_Allreduce(&wtime, &wtime_avg, 1, MPI_REAL_FP, MPI_SUM,
                    MPI_COMM_WORLD);
      wtime_avg = wtime_avg / p_row / p_col;
      if (mod_mpi::myrank == 0) {
        int t = static_cast<int>((nt_end - nt) * wtime_avg / nt_out_scrn);
        int t_d = t / 86400;
        int t_res = t % 86400;
        int t_h = t_res / 3600;
        t_res %= 3600;
        int t_m = t_res / 60;
        int t_s = t_res % 60;
        auto remaining_time =
            fmt::format("{:3}d{:2}h{:2}m{:2}s", t_d, t_h, t_m, t_s);
        auto speed = nt_out_scrn * 3600.0 * 24.0 / wtime_avg / 10000;
        fmt::print("{:9}  {:17.3f}  {:>14}  {:10.3E}  {:10.3E}\n", nt, speed,
                   remaining_time, cfl_max.value, div_max.value);
      }
    }

    // output routines below
    mod_dataIO::outputData(nt, u_crf, u, v, w, p);

    if (nt % nt_out_scrn == 0) {
      wtime = MPI_Wtime();
    }
  }

  GPTLstop("Main loop");
  GPTLpr_summary(MPI_COMM_WORLD);
  GPTLfinalize();

  mod_hdf5::freeIO();
  free_poisson_solver();
  mod_mesh::freeMesh();
  mod_monitor::freeMonitor();
  freeVariables();
  mod_statistics::freeStat();
  mod_utils::freeCheckCFLAndDiv();
  mod_mpi::freeMPI();

  if (mod_mpi::myrank == 0) {
    fmt::print("***************************************************************"
               "*****************\n");
    fmt::print("PowerLLEL.NOTE: Simulation ends successfully!\n");
  }
  MPI_Finalize();
}