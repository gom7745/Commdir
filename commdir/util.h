#ifndef __UTIL_H__
#define __UTIL_H__
#ifdef __cplusplus
extern "C" 
{
#endif
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <limits.h>

#define NO_LABEL INT_MAX
#define FNAME_MAX 255
#define EPS 1e-15
#define OO 1e41

typedef struct Fnode Fnode;
typedef struct Loss Loss;
typedef struct Data Data;
typedef struct Param Param;
typedef struct Model Model;
typedef struct Vector Vector;

struct Fnode 
{
        int index;
        double value;
};

struct Loss
{
        // z = yi w'xi
        void (*function)(double z, double *value);
        //first and second order derivatives to z
        void (*derivatives)(double z, double *first, double *second);
};

struct Data
{
        int nr_fnode;
        int l, n;
        int ns, nd;
        double bias;
        Fnode **sparse_x;
        double **dense_x;
        int *y;
};

struct Param // Should be Prob
{
        int loss_type;
        Loss loss;
	double eps;	        /* stopping criteria */
	double C;
        int m;
        int max_inner;
};

struct Model
{
        Param param;
	int nr_class;		/* number of classes */
	int nr_feature;
        int nr_sparse_feature, nr_dense_feature;
        double bias;
        double *w;
        int *label;
};

// time
int64_t wall_clock_ns();
double wall_time_diff(int64_t ed, int64_t st);

// file utility
const char* readline(FILE *input);
const char *basename(const char *path);
FILE *fopen_or_abort(const char *fname, const char *mode);
int for_each_line_show_progress(FILE *fin, int (*filter_func)(void *ctx, const char *line), void *ctx);
// filter_func:
// Read one new line at each iteration.
// Then for each iteration, the user defined contex *ctx
// and the new line *line will be passed to filter_func.
//
// Return negative if you want to terminate the filter

// svm data format
int read_instance(const char *s, Fnode **x, int *label_len);
int last_feature_index(const char *s);
void do_data_statistics(FILE *fp, int *n, int *l, int *nnz);
int read_and_filter(FILE *fin, Data *data, int nd, double bias,
        int (*filter_func)(void *ctx, const Fnode *xin,
                const Fnode **sparse_xout, const double **dense_xout),
        void* filter_ctx);

int read_data(FILE *fin, double bias, Data *data);
void transpose(Data *data);
void save_data(FILE *fout, Data data);
Data load_data(FILE *fin);
void free_data(Data *data);

// logging
void start_logging(const char *prog_name);
void lprintf(const char *format, ...);
void end_logging();

// vector for buffer management
struct Vector {
        char *data;
        size_t len;
        size_t cap;
        size_t size;
};
struct Vector* make_vector(size_t size);
void* vector_push_elements(struct Vector *v, const void *elements, size_t n);
void vector_shrink_to_fit(struct Vector *v);

// math
double max(double x, double y);
double min(double x, double y);
void swap_double(double *a, int i, int j);
void swap_int(int *a, int i, int j);
// math: dense
double ddot(const double *x, const double *y, int l);
double ddot3(const double *x, const double *y, const double *z, int l);
double dnrm2(const double *x, int l);
void dcopy(double *x, double *y, int l);
void dscal(double *x, double a, int l);
void daxpy(double *restrict y, double a, const double *restrict x, int l);
void dzero(double *v, int l);
// math: sparse
double sdot(const Fnode *xi, const double *v);
double sdot_boundcheck(const Fnode *xi, const double *v, int max_index);
double snrm2(const Fnode *xi);
void saxpy(double *y, double a, const Fnode *x);
void solve_nxn(int n, double *H, double *g, double *d);
// solve Ax=b , in which A[i*n+j] = Aij and we only solve submatrix A[m:m]
double LDL_with_pivoting(const double *A, const double *b, double *x, int n, int m);

// random
void permutation(int *a, int l);

// model
int number_of_w(int nr_class);
void init_model(const Data data, const Param param, Model *model);
void free_model(Model *model);
void save_model(FILE *f, const Model model);
void load_model(FILE *f, Model *model);
#ifdef __cplusplus
}
#endif
#endif
