#include "mod_type.hh"

// Only support 'PP' bctype with oneMKL yet
void init_fft(int xsz[3], int ysz[3], char bctype_x[], char bctype_y[]);
void execute_fft_fwd_xpen(const Array3DH1 &p, double *var_xpen);
void execute_fft_bwd_xpen(const double *var_xpen, Array3DH1 &p);
void execute_fft_fwd_ypen(double *var_ypen);
void execute_fft_bwd_ypen(double *var_ypen);
void free_fft();
void get_eigen_values(int ist, int isz, int isz_global, char bctype[],
                      double *lambda);
