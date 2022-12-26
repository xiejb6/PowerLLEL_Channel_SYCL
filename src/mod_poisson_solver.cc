#include "mod_poisson_solver.hh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "gptl.hh"
#include "decomp_2d.hh"
#include "mod_fft.hh"
#include "mod_oneapi.hh"
#include "memory.hh"

double fft_normfactor;
int *xsize, *ysize, *zsize;
double *var_xpen, *var_ypen, *var_zpen;
double *a, *b, *c;
int *sz_trid, *st_trid, *neighbor_trid;

#ifdef _PDD
MPI_Comm COMM_CART_PDD;
double *w_pdd, *v_pdd, *tmp_v_pdd;
double *y1_pdd, *y2_pdd, *y3_pdd;
double *tmp_var_pdd;
size_t tmp_var_pdd_len;
#endif

size_t var_offset1;
size_t var_offset2;
size_t var_xpen_offset1;
size_t var_xpen_offset2;
size_t var_ypen_offset1;
size_t var_ypen_offset2;

static void set_trid_coeff(int n, double *dzf, char bctype[], char c_or_f, int neighbor[2], 
                           double a[], double b[], double c[]) {
    int dzf_st = -1;
    if (c_or_f == 'c') {
        for (int k = 0; k < n; k++) {
            a[k] = 2.0 / (dzf[k - dzf_st] * (dzf[k-1 - dzf_st] + dzf[k - dzf_st]));
            c[k] = 2.0 / (dzf[k - dzf_st] * (dzf[k+1 - dzf_st] + dzf[k - dzf_st]));
        }
    } else if (c_or_f == 'f') {
        for (int k = 0; k < n; k++) {
            a[k] = 2.0 / (dzf[k   - dzf_st] * (dzf[k+1 - dzf_st] + dzf[k - dzf_st]));
            c[k] = 2.0 / (dzf[k+1 - dzf_st] * (dzf[k+1 - dzf_st] + dzf[k - dzf_st]));
        }
    }

    for (int i = 0; i < n; i++) {
        b[i] = - a[i] - c[i];
    }

    // coefficients correction according to BC types
    double factor;
    char bc;
    if (neighbor[0] == MPI_PROC_NULL) {
        bc = bctype[0];
        if (bc == 'P') {
            factor = 0.0;
        } else if (bc == 'D') {
            factor = -1.0;
        } else if (bc == 'N') {
            factor = 1.0;
        }

        if (c_or_f == 'c') {
            b[0] += factor * a[0];
            a[0] *= fabs(factor) - 1.0;
        } else if (c_or_f == 'f') {
            if (bc == 'N') {
                b[0] += factor * a[0];
                a[0] *= fabs(factor) - 1.0;
            }
        }
    }
    
    if (neighbor[1] == MPI_PROC_NULL) {
        bc = bctype[1];
        if (bc == 'P') {
            factor = 0.0;
        } else if (bc == 'D') {
            factor = -1.0;
        } else if (bc == 'N') {
            factor = 1.0;
        }

        if (c_or_f == 'c') {
            b[n-1] += factor * c[n-1];
            c[n-1] *= fabs(factor) - 1.0;
        } else if (c_or_f == 'f') {
            if (bc == 'N') {
                b[n-1] += factor * c[n-1];
                c[n-1] *= fabs(factor) - 1.0;
            }
        }
    }
}

#ifdef _PDD
static void init_pdd_array(int sz[3]) {
    v_pdd = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                               sz[0] * sz[1] * sz[2], 
                                               mod_oneapi::queue);
    w_pdd = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                               sz[0] * sz[1] * sz[2], 
                                               mod_oneapi::queue);
    tmp_var_pdd_len = sz[0]*sz[1];
    y1_pdd = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                                tmp_var_pdd_len, 
                                                mod_oneapi::queue);
    y2_pdd = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                                tmp_var_pdd_len, 
                                                mod_oneapi::queue);
    y3_pdd = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                                tmp_var_pdd_len, 
                                                mod_oneapi::queue);
    tmp_v_pdd = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                                tmp_var_pdd_len, 
                                                mod_oneapi::queue);
    tmp_var_pdd = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                                tmp_var_pdd_len, 
                                                mod_oneapi::queue);

    memset(v_pdd, 0, sizeof(double) * sz[0] * sz[1] * sz[2]);
    memset(w_pdd, 0, sizeof(double) * sz[0] * sz[1] * sz[2]);
    memset(y1_pdd, 0, sizeof(double) * tmp_var_pdd_len);
    memset(y2_pdd, 0, sizeof(double) * tmp_var_pdd_len);
    memset(y3_pdd, 0, sizeof(double) * tmp_var_pdd_len);

    int ie = sz[0];
    int je = sz[1];
    int ke = sz[2];
    for (int j = 0; j < je; j++) {
        for (int i = 0; i < ie; i++) {
            v_pdd[i + j * ie +      0 * ie * je] = a[0];
            w_pdd[i + j * ie + (ke-1) * ie * je] = c[ke-1];
        }
    }
    a[0] = 0.0;
    c[ke-1] = 0.0;

    for (int k = 1; k < ke; k++) {
        for (int j = 0; j < je; j++) {
            for (int i = 0; i < ie; i++) {
                double a_tmp = a[k] / b[i + j * ie + (k-1) * ie * je];
                v_pdd[i + j * ie + k * ie * je] -= a_tmp * v_pdd[i + j * ie + (k-1) * ie * je];
                w_pdd[i + j * ie + k * ie * je] -= a_tmp * w_pdd[i + j * ie + (k-1) * ie * je];
            }
        }
    }

    for (int j = 0; j < je; j++) {
        for (int i = 0; i < ie; i++) {
            if (b[i + j * ie + (ke-1) * ie * je] != 0.0) {
                v_pdd[i + j * ie + (ke-1) * ie * je] /= b[i + j * ie + (ke-1) * ie * je];
                w_pdd[i + j * ie + (ke-1) * ie * je] /= b[i + j * ie + (ke-1) * ie * je];
            } else {
                v_pdd[i + j * ie + (ke-1) * ie * je] = 0.0;
                w_pdd[i + j * ie + (ke-1) * ie * je] = 0.0;
            }
        }
    }

    for (int k = ke-2; k >= 0; k--) {
        for (int j = 0; j < je; j++) {
            for (int i = 0; i < ie; i++) {
                v_pdd[i + j * ie + k * ie * je] = (v_pdd[i + j * ie + k * ie * je] - c[k] * v_pdd[i + j * ie + (k+1) * ie * je]) / b[i + j * ie + k * ie * je];
                w_pdd[i + j * ie + k * ie * je] = (w_pdd[i + j * ie + k * ie * je] - c[k] * w_pdd[i + j * ie + (k+1) * ie * je]) / b[i + j * ie + k * ie * je];
            }
        }
    }

    MPI_Sendrecv(v_pdd, tmp_var_pdd_len, MPI_DOUBLE, neighbor_trid[4], 100,
                 tmp_v_pdd, tmp_var_pdd_len, MPI_DOUBLE, neighbor_trid[5], 100,
                 COMM_CART_PDD, MPI_STATUS_IGNORE);
}
#endif

void init_poisson_solver(int nx_global, int ny_global, int nz_global, 
                         double dx, double dy, double *dzf_global,
                         char bctype_x[], char bctype_y[], char bctype_z[], 
                         std::array<std::array<int, 6>, 3> &neighbor_xyz) {

    xsize = decomp_main.xsz.data();
    ysize = decomp_main.ysz.data();
    zsize = decomp_main.zsz.data();

    var_offset1 = (xsize[0]+2) * (xsize[1]+2);
    var_offset2 = (xsize[0]+2);
    var_xpen_offset1 = xsize[0] * xsize[1];
    var_xpen_offset2 = xsize[0];
    var_ypen_offset1 = ysize[0] * ysize[1];
    var_ypen_offset2 = ysize[0];

    var_xpen = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                                  xsize[0]*xsize[1]*xsize[2] + MEM_ALIGN_SIZE, 
                                                  mod_oneapi::queue);
    if (xsize[0] == ysize[0] && xsize[1] == ysize[1] && xsize[2] == ysize[2]) {
        var_ypen = var_xpen;
    }
    else {
        var_ypen = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                                      ysize[0]*ysize[1]*ysize[2] + MEM_ALIGN_SIZE, 
                                                      mod_oneapi::queue);
    }
    if (ysize[0] == zsize[0] && ysize[1] == zsize[1] && ysize[2] == zsize[2]) {
        var_zpen = var_ypen;
    }
    else {
        var_zpen = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                                      zsize[0]*zsize[1]*zsize[2] + MEM_ALIGN_SIZE, 
                                                      mod_oneapi::queue);
    }

    // determine a decomposition mode used for solving the tridiagonal system
#ifdef _PDD
    sz_trid = decomp_main.ysz.data();
    st_trid = decomp_main.yst.data();
    neighbor_trid = &neighbor_xyz[1][0];
#else
    sz_trid = decomp_main.zsz.data();
    st_trid = decomp_main.zst.data();
    neighbor_trid = &neighbor_xyz[2][0];
#endif

    // initialize FFT
    init_fft(xsize, ysize, bctype_x, bctype_y);

    // calculate eigenvalues corresponding to BC types
    double *lambdax  = sycl::malloc_shared<double>(sz_trid[0], mod_oneapi::queue);
    double *lambday  = sycl::malloc_shared<double>(sz_trid[1], mod_oneapi::queue);
    double *lambdaxy = sycl::malloc_shared<double>(sz_trid[0] * sz_trid[1], mod_oneapi::queue);
    get_eigen_values(st_trid[0], sz_trid[0], nx_global, bctype_x, lambdax);
    for (int i = 0; i < sz_trid[0]; i++) {
        lambdax[i] /= (dx*dx);
    }
    get_eigen_values(st_trid[1], sz_trid[1], ny_global, bctype_y, lambday);
    for (int i = 0; i < sz_trid[1]; i++) {
        lambday[i] /= (dy*dy);
    }
    for (int j = 0; j < sz_trid[1]; j++) {
        for (int i = 0; i < sz_trid[0]; i++) {
            lambdaxy[i + j * sz_trid[0]] = lambdax[i] + lambday[j];
        }
    }
    
    // calculate coefficients of tridiagonal systems
    a = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                           sz_trid[2] + MEM_ALIGN_SIZE, 
                                           mod_oneapi::queue);
    c = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                           sz_trid[2] + MEM_ALIGN_SIZE, 
                                           mod_oneapi::queue);
    b = sycl::aligned_alloc_shared<double>(MEM_ALIGN_SIZE,
                                           sz_trid[0]*sz_trid[1]*sz_trid[2] + MEM_ALIGN_SIZE, 
                                           mod_oneapi::queue);
    double *b_tmp = sycl::malloc_shared<double>(sz_trid[2], mod_oneapi::queue);
    double *dzf = sycl::malloc_shared<double>((sz_trid[2] + 2), mod_oneapi::queue);
    
    for (int i = 0; i < sz_trid[2] + 2; i++) {
        dzf[i] = dzf_global[st_trid[2]-1+i];
    }
    set_trid_coeff(sz_trid[2], dzf, bctype_z, 'c', &neighbor_trid[4], a, b_tmp, c);
    for (int k = 0; k < sz_trid[2]; k++) {
        for (int j = 0; j < sz_trid[1]; j++) {
            for (int i = 0; i < sz_trid[0]; i++) {
                b[i + j * sz_trid[0] + k * sz_trid[0] * sz_trid[1]] = 
                    b_tmp[k] + lambdaxy[i + j * sz_trid[0]];
            }
        }
    }

    // decompose coefficient b
    for (int k = 1; k < sz_trid[2]; k++) {
        for (int j = 0; j < sz_trid[1]; j++) {
            for (int i = 0; i < sz_trid[0]; i++) {
                double a_tmp = a[k] / b[i + j * sz_trid[0] + (k-1) * sz_trid[0] * sz_trid[1]];
                b[i + j * sz_trid[0] + k * sz_trid[0] * sz_trid[1]] -= (a_tmp * c[k-1]);
            }
        }
    }
    // double *b_k_ptr = b + sz_trid[0] * sz_trid[1];
    // double *b_km1_ptr = b;
    // for (int k = 1; k < sz_trid[2]; k++) {
    //     for (int j = 0; j < sz_trid[1]; j++) {
    //         for (int i = 0; i < sz_trid[0]; i++) {
    //             double a_tmp = a[k] / *b_km1_ptr;
    //             *b_k_ptr -= a_tmp * c[k-1];
    //             b_k_ptr++;
    //             b_km1_ptr++;
    //         }
    //     }
    // }

    // determine whether the tridiagonal systems are periodic or not
    // NOTE: not yet implemented

    // calculate the correction of the right-hand side according to BC types in x, y, z direction
    // NOTE: not yet implemented

#ifdef _PDD
    COMM_CART_PDD = decomp_2d_comm_cart_y;
    
    // initialize work arrays for PDD algorithm
    init_pdd_array(sz_trid);
#endif

    sycl::free(lambdax, mod_oneapi::queue);
    sycl::free(lambday, mod_oneapi::queue);
    sycl::free(lambdaxy, mod_oneapi::queue);
    sycl::free(b_tmp, mod_oneapi::queue);
    sycl::free(dzf, mod_oneapi::queue);

}

void ssolve_trid(int *sz, double *var) {

    int ie = sz[0];
    int je = sz[1];
    int ke = sz[2];

    auto _a = a;
    auto _b = b;
    auto _c = c;
    auto sz_trid0 = sz_trid[0];
    auto sz_trid1 = sz_trid[1];
    auto r = sycl::range(je, ie);
    auto ev = mod_oneapi::queue.parallel_for(r, [=](sycl::item<2> idx) {
      int i = idx[1];
      int j = idx[0];
      auto rg = idx.get_range();

      for (int k = 1; k < ke; ++k) {
        double var_up = var[(k - 1) * rg[0] * rg[1] + j * rg[1] + i];
        double bv = _b[(k - 1) * sz_trid0 * sz_trid1 + j * sz_trid0 + i];
        double a_tmp = _a[k] / bv;
        var[k * rg[0] * rg[1] + j * rg[1] + i] -= a_tmp * var_up;
      }

      double b_val = _b[(ke - 1) * sz_trid0 * sz_trid1 + j * sz_trid0 + i];
      double &var_p = var[(ke - 1) * rg[0] * rg[1] + j * rg[1] + i];
      if (b_val != 0)
        var_p /= b_val;
      else
        var_p = 0;

      for (int k = ke - 2; k >= 0; k--) {
        double &var_down = var[k * rg[0] * rg[1] + j * rg[1] + i];
        double var_up = var[(k + 1) * rg[0] * rg[1] + j * rg[1] + i];
        double bv = _b[k * sz_trid0 * sz_trid1 + j * sz_trid0 + i];
        double cv = _c[k];
        var_down = (var_down - cv * var_up) / bv;
      }
    });
    ev.wait();
}

#ifdef _PDD
void psolve_trid(int *sz, double *var) {

    int ie = sz[0];
    int je = sz[1];
    int ke = sz[2];

    auto _a = a;
    auto _b = b;
    auto _c = c;
    auto _v_pdd = v_pdd;
    auto _w_pdd = w_pdd;
    auto _tmp_v_pdd = tmp_v_pdd;
    auto _tmp_var_pdd = tmp_var_pdd;
    auto _y1_pdd = y1_pdd;
    auto _y2_pdd = y2_pdd;
    auto _y3_pdd = y3_pdd;
    auto sz_trid0 = sz_trid[0];
    auto sz_trid1 = sz_trid[1];
    auto r = sycl::range(je, ie);
    auto ev = mod_oneapi::queue.parallel_for(r, [=](sycl::item<2> idx) {
      auto [i, j] = mod_oneapi::unpack<0>(idx);
      auto rg = idx.get_range();

      for (int k = 1; k < ke; ++k) {
        double var_up = var[(k - 1) * rg[0] * rg[1] + j * rg[1] + i];
        double bv = _b[(k - 1) * sz_trid0 * sz_trid1 + j * sz_trid0 + i];
        double a_tmp = _a[k] / bv;
        var[k * rg[0] * rg[1] + j * rg[1] + i] -= a_tmp * var_up;
      }

      double b_val = _b[(ke - 1) * sz_trid0 * sz_trid1 + j * sz_trid0 + i];
      double &var_p = var[(ke - 1) * rg[0] * rg[1] + j * rg[1] + i];
      if (b_val != 0)
        var_p /= b_val;
      else
        var_p = 0;

      for (int k = ke - 2; k >= 0; k--) {
        double &var_down = var[k * rg[0] * rg[1] + j * rg[1] + i];
        double var_up = var[(k + 1) * rg[0] * rg[1] + j * rg[1] + i];
        double bv = _b[k * sz_trid0 * sz_trid1 + j * sz_trid0 + i];
        double cv = _c[k];
        var_down = (var_down - cv * var_up) / bv;
      }
    });
    ev.wait();

    GPTLstart("----Comm in PDD");
    MPI_Sendrecv(var, tmp_var_pdd_len, MPI_DOUBLE, neighbor_trid[4], 100,
         tmp_var_pdd, tmp_var_pdd_len, MPI_DOUBLE, neighbor_trid[5], 100,
         COMM_CART_PDD, MPI_STATUS_IGNORE);
    GPTLstop("----Comm in PDD");

    if (neighbor_trid[5] != MPI_PROC_NULL) {
      ev = mod_oneapi::queue.parallel_for(r, [=](sycl::item<2> idx) {
        auto [i, j] = mod_oneapi::unpack<0>(idx);
        auto rg = idx.get_range();

        double w_pdd_val =
            _w_pdd[(ke - 1) * sz_trid0 * sz_trid1 + j * sz_trid0 + i];
        double tmp_v_pdd_val = _tmp_v_pdd[j * sz_trid0 + i];
        double var_val = var[(ke - 1) * rg[0] * rg[1] + j * rg[1] + i];
        double tmp_var_pdd_val = _tmp_var_pdd[j * sz_trid0 + i];
        double det_pdd_inv = 1.0 / (w_pdd_val * tmp_v_pdd_val - 1.0);
        _y2_pdd[j * sz_trid0 + i] =
            (var_val * tmp_v_pdd_val - tmp_var_pdd_val) * det_pdd_inv;
        _y3_pdd[j * sz_trid0 + i] =
            (tmp_var_pdd_val * w_pdd_val - var_val) * det_pdd_inv;
      });
      ev.wait();
    }

    GPTLstart("----Comm in PDD");
    MPI_Sendrecv(y3_pdd, tmp_var_pdd_len, MPI_DOUBLE, neighbor_trid[5], 100,
                 y1_pdd, tmp_var_pdd_len, MPI_DOUBLE, neighbor_trid[4], 100,
                 COMM_CART_PDD, MPI_STATUS_IGNORE);
    GPTLstop("----Comm in PDD");

    ev = mod_oneapi::queue.parallel_for(
        sycl::range(ke, je, ie), [=](sycl::item<3> idx) {
          auto [i, j, k] = mod_oneapi::unpack<0>(idx);
          auto rg = idx.get_range();

          double v_pdd_val = _v_pdd[k * sz_trid0 * sz_trid1 + j * sz_trid0 + i];
          double w_pdd_val = _w_pdd[k * sz_trid0 * sz_trid1 + j * sz_trid0 + i];
          double y1_pdd_val = _y1_pdd[j * sz_trid0 + i];
          double y2_pdd_val = _y2_pdd[j * sz_trid0 + i];
          var[k * rg[1] * rg[2] + j * rg[2] + i] -=
              (v_pdd_val * y1_pdd_val + w_pdd_val * y2_pdd_val);
        });
    ev.wait();
}
#endif

void execute_poisson_solver(Array3DH1& p) {
    double* var = p.data();

    GPTLstart("--Copy & Fwd X-FFT");
    execute_fft_fwd_xpen(p, var_xpen);
    GPTLstop("--Copy & Fwd X-FFT");

    GPTLstart("--Transpose x to y");
    if (var_xpen != var_ypen) {
        transpose_x_to_y_real(var_xpen, var_ypen, xsize, ysize, &decomp_main);
    }
    GPTLstop("--Transpose x to y");

    GPTLstart("--Forward Y-FFT");
    execute_fft_fwd_ypen(var_ypen);
    GPTLstop("--Forward Y-FFT");

    #ifdef _PDD
    {
        GPTLstart("--Solve trid");
        psolve_trid(ysize, var_ypen);
        GPTLstop("--Solve trid");
    }
    #else
    {
        GPTLstart("--Transpose y to z");
        if (var_ypen != var_zpen) {
            transpose_y_to_z_real(var_ypen, var_zpen, ysize, zsize, &decomp_main);
        }
        GPTLstop("--Transpose y to z");

        GPTLstart("--Solve trid");
        ssolve_trid(zsize, var_zpen);
        GPTLstop("--Solve trid");

        GPTLstart("--Transpose z to y");
        if (var_ypen != var_zpen) {
            transpose_z_to_y_real(var_zpen, var_ypen, zsize, ysize, &decomp_main);
        }
        GPTLstop("--Transpose z to y");
    }
    #endif

    GPTLstart("--Backward Y-FFT");
    execute_fft_bwd_ypen(var_ypen);
    GPTLstop("--Backward Y-FFT");

    GPTLstart("--Transpose y to x");
    if (var_xpen != var_ypen) {
        transpose_y_to_x_real(var_ypen, var_xpen, ysize, xsize, &decomp_main);
    }
    GPTLstop("--Transpose y to x");

    GPTLstart("--Bwd X-FFT & Copy");
    execute_fft_bwd_xpen(var_xpen, p);
    GPTLstop("--Bwd X-FFT & Copy");
}

void free_poisson_solver() {
    
    // release work arrays
    if (var_xpen != NULL) sycl::free(var_xpen, mod_oneapi::queue);
    if (var_xpen != var_ypen && var_ypen != NULL) sycl::free(var_ypen, mod_oneapi::queue);
    if (var_ypen != var_zpen && var_zpen != NULL) sycl::free(var_zpen, mod_oneapi::queue);

    // release tridiagonal coefficients arrays
    if (a != NULL) sycl::free(a, mod_oneapi::queue);
    if (b != NULL) sycl::free(b, mod_oneapi::queue);
    if (c != NULL) sycl::free(c, mod_oneapi::queue);

#ifdef _PDD
    // release PDD related arrays
    if (v_pdd != NULL) sycl::free(v_pdd, mod_oneapi::queue);
    if (w_pdd != NULL) sycl::free(w_pdd, mod_oneapi::queue);
    if (y1_pdd != NULL) sycl::free(y1_pdd, mod_oneapi::queue);
    if (y2_pdd != NULL) sycl::free(y2_pdd, mod_oneapi::queue);
    if (y3_pdd != NULL) sycl::free(y3_pdd, mod_oneapi::queue);
    if (tmp_v_pdd != NULL) sycl::free(tmp_v_pdd, mod_oneapi::queue);
    if (tmp_var_pdd != NULL) sycl::free(tmp_var_pdd, mod_oneapi::queue);
#endif

    // release FFT
    free_fft();
}
