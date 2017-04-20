#include "solver.h"
#ifndef TIMER
#define TIMER
#define NS_PER_SEC 1000000000
double wall_time_diff(int64_t ed, int64_t st)
{
	return (double)(ed-st)/(double)NS_PER_SEC;
}
int64_t wall_clock_ns()
{
#ifdef __unix__
	struct timespec tspec;
	int r = clock_gettime(CLOCK_MONOTONIC, &tspec);
	return tspec.tv_sec*NS_PER_SEC + tspec.tv_nsec;
#else
	struct timeval tv;
	//int r = gettimeofday( &tv, NULL );
	return 0;
	//return tv.tv_sec*NS_PER_SEC + tv.tv_usec*1000;
#endif
}
#endif

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void LBFGS::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*lbfgs_print_string)(buf);
}

void LBFGS::set_print_string(void (*print_string) (const char *buf))
{
	lbfgs_print_string = print_string;
}

LBFGS::LBFGS(const function *fun_obj, double eps, int m, double eta, int max_iter)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
	this->eta = eta;
	lbfgs_print_string = default_print;
	this->M = m;
}

LBFGS::~LBFGS()
{
}

void LBFGS::lbfgs(double *w)
{
	int n = fun_obj->get_nr_variable();
	int i;
	int k = 0;
	double f;
	int iter = 1, inc = 1;
	double *g;
	double *old_g;
	double init_size = 1;
	double *step = new double[n];
	memset(w, 0, sizeof(double) * size_t(n));
	int64_t time_st = wall_clock_ns();
	double gnorm1;
	double gnorm;

	iter = 0;

	double **s = new double*[M];
	double **y = new double*[M];
	double *rho = new double[M];
	for (int i = 0; i < M; i++)
	{
		s[i] = new double[n];
		y[i] = new double[n];
	}

	f = fun_obj->fun(w);

	double origDirDeriv = 0.0;
	double step_size = 0.0;
	while (iter < max_iter)
	{
		int DynamicM = min(iter, M);
		int64_t iter_ed = wall_clock_ns();
		info("iter= %d m=%d |g|=%5.3e f=%.14e acc=0.00%% time=%.4es\n", iter,DynamicM, gnorm, f,
				wall_time_diff(iter_ed, time_st));

		g = new double[n];
		fun_obj->grad(w, g);
		gnorm = dnrm2_(&n, g, &inc);
		if (iter == 0)
			gnorm1 = gnorm;

		if (gnorm <= eps*gnorm1)
			break;
		if (iter > 0)
		{
			double s0y0 = 0;
			for (i=0;i<n;i++)
			{
				y[k][i] = g[i] - old_g[i];
				s[k][i] = step[i] * step_size;
				s0y0 += y[k][i] * s[k][i];
			}

			if (s0y0 == 0)
				break;
			rho[k] = 1.0 / s0y0;
			k = (k+1)%M;

			TwoLoop(g, s, y, rho, DynamicM, step, k, n);
			delete[] old_g;
			init_size = 1;
		}
		else
		{
			for (i=0;i<n;i++)
				step[i] = -g[i];
			init_size = 1.0 / gnorm;
		}

		/*
		origDirDeriv = ddot_(&n, step, &inc, g, &inc);
		if (origDirDeriv >= 0)
		{
			info("WARNING: not a descent direction\n");
			break;
		}
		*/
		double ws = ddot_(&n, w, &inc, step, &inc);

		step_size = fun_obj->line_search(step, ws, &f, eta, init_size);
		if (step_size == 0)
		{
			info("WARNING: step size too small\n");
			break;
		}
		daxpy_(&n, &step_size, step, &inc, w, &inc);
		old_g = g;
		iter++;
	}
}

void LBFGS::TwoLoop(double* g, double** s, double** y, double* rho, int DynamicM, double *step, int k, int n)
{
	int i, j;
	int inc = 1;
	double *alpha = new double[DynamicM];
	for (i=0;i<n;i++)
		step[i] = -g[i];
	int start = k-1;
	if (k < DynamicM)
		start += DynamicM;

	int lastrho = -1;
	for (i = 0; i < DynamicM; i++)
	{
		j = start % DynamicM;

		start--;
		if (s[j] == NULL || y[j] == NULL)
			continue;
		if (rho[j] > 0)
		{
			alpha[j] = rho[j] * ddot_(&n, step, &inc, s[j], &inc);
			double a = -alpha[j];
			daxpy_(&n, &a, y[j], &inc, step, &inc);
			if (lastrho == -1)
				lastrho = j;
		}
		else
		{
			fprintf(stderr,"ERROR: rho[%d] <= 0\n",i);
			return;
		}
	}
	if (lastrho != -1)
	{
		double hk0 = 1.0/(rho[lastrho] * ddot_(&n, y[lastrho], &inc, y[lastrho], &inc));
		dscal_(&n, &hk0, step, &inc);
		for (i = 0; i < DynamicM; i++)
		{
			start++;
			j = start % DynamicM;
			if (rho[j] <= 0 || s[j] == NULL)
				continue;
			double beta = rho[j] *ddot_(&n, step, &inc, y[j], &inc);
			beta = alpha[j] - beta;
			daxpy_(&n, &beta, s[j], &inc, step, &inc);
		}
	}
}

void NEWTON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*newton_print_string)(buf);
}

NEWTON::NEWTON(const function *fun_obj, double eps, double eta, int max_iter)
{
	this->fun_obj = const_cast<function *>(fun_obj);
	this->eps = eps;
	this->eta = eta;
	this->max_iter=max_iter;
	newton_print_string = default_print;
}

NEWTON::~NEWTON()
{
}

void NEWTON::newton(double *w)
{
	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	double cg_eps = 0.1;
	double snorm;
	double f, fnew, actred, gs;
	int search = 1, iter = 1, inc = 1;
	double *s = new double[n];
	double *r = new double[n];
	double *g = new double[n];

	memset(w, 0, sizeof(double) * size_t(n));

	int64_t time_st = wall_clock_ns();
	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	double gnorm1 = dnrm2_(&n, g, &inc);
	double gnorm = gnorm1;
	double init_size = 1;
	double step;
	int64_t iter_ed;

	if (gnorm <= eps*gnorm1)
		search = 0;

	iter = 1;

	while (iter <= max_iter && search)
	{
		cg_iter = cg(g, s, r, cg_eps);

		gs = ddot_(&n, g, &inc, s, &inc);
		double ws = ddot_(&n, w, &inc, s, &inc);
		step = fun_obj->line_search(s, ws, &fnew, eta, init_size);
		daxpy_(&n, &step, s, &inc, w, &inc);
		//prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));

		// Compute the actual reduction.
		actred = f - fnew;

		iter_ed = wall_clock_ns();
		info("iter %2d act %5.3e f %.14e |g| %5.3e CG %3d step_size %5.3e time %5.3es\n", iter, actred, f, gnorm, cg_iter, step, wall_time_diff(iter_ed, time_st));
		time_st = wall_clock_ns();
		iter++;
		f = fnew;
		fun_obj->grad(w, g);

		gnorm = dnrm2_(&n, g, &inc);
		if (gnorm <= eps*gnorm1)
			break;
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
		if (fabs(actred) <= 0)
		{
			info("WARNING: actred <= 0\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f))
		{
			info("WARNING: actred too small\n");
			break;
		}
	}
	iter_ed = wall_clock_ns();
	info("iter %2d act %5.3e f %.14e |g| %5.3e CG %3d step_size %5.3e time %5.3es\n", iter, 0.0, f, gnorm, 0, 1.0, wall_time_diff(iter_ed, time_st));

	delete[] g;
	delete[] r;
	delete[] s;
}

int NEWTON::cg(double *g, double *s, double *r, double cg_eps)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	double *d = new double[n];
	double *Hd = new double[n + 1];
	double rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = cg_eps * dnrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (dnrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
	delete[] Hd;

	return(cg_iter);
}

void NEWTON::set_print_string(void (*print_string) (const char *buf))
{
	newton_print_string = print_string;
}

void AFG::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*afg_print_string)(buf);
}

AFG::AFG(const function *fun_obj, double eps, double t_pre, double mu,  int max_iter)
{
	this->fun_obj = const_cast<function *>(fun_obj);
	this->eps = eps;
	this->mu= mu;
	this->max_iter=max_iter;
	this->t_pre = t_pre;
	afg_print_string = default_print;
}

AFG::~AFG()
{
}

void AFG::afg(double *w)
{
	int n = fun_obj->get_nr_variable();
	//parameters follow from Xiao Lin's code
	int inc = 1;
	double beta = 0.5;
	double beta_square = 0.25;
	double alpha = 1;
	double alpha_pre = 1;
	double t = 1;
	int i = 1;
	double *x = new double[n];
	double *x_pre = new double[n];
	double *v = new double[n];
	double *y = new double[n];
	double *gy = new double[n];
	double f;
	memset(x_pre, 0, sizeof(double) * n);
	memcpy(v, x_pre, sizeof(double) * n);
	double *g = new double[n];
	int search = 1;
	f = fun_obj->fun(x_pre);
	fun_obj->grad(x_pre, g);

	double gnorm1 = dnrm2_(&n, g, &inc);
	double gnorm = gnorm1;

	if (gnorm <= eps*gnorm1)
		search = 0;

	info("iter= %d m=%d |g|=%5.3e f=%.14e acc=0.00%% passes=%d 1/t=%5.3e time=%.4es\n", 0, 2, gnorm1, f, 0,
				1/t_pre, 0.0);
	int64_t time_st = wall_clock_ns();
	while (i <= max_iter && search)
	{
		t = t_pre / beta_square;
		int counter = 0;
		while (true)
		{
			t *= beta;
			double a = 1;
			double tmp = alpha_pre * alpha_pre * t / t_pre;
			double b = - mu * t + tmp;
			double c = - tmp;
			alpha = (- b + sqrt((b * b) - 4 * a * c)) / (2 * a);
			double theta = alpha / (1 + (alpha / alpha_pre / alpha_pre) * t_pre * mu);
			double coeff = 1 - theta;

			for (int j=0;j<n;j++)
				y[j] = coeff * x_pre[j] + theta * v[j];

			double fy = fun_obj->fun(y);
			fun_obj->grad(y, gy);
			counter++;

			// compute proximal mapping
			double inner_product = 0;
			double norm2 = 0;
			for (int j=0;j<n;j++)
				x[j] = y[j] - t * gy[j];
			inner_product = - ddot_(&n, gy, &inc, gy, &inc) * t;
			norm2 = ddot_(&n, gy, &inc, gy, &inc) * t * 0.5;
			f = fun_obj->fun(x);

			if (f <= fy + inner_product + norm2)
				break;
		}
		for (int j = 0; j < n; j++)
			v[j] = x_pre[j] + (x[j] - x_pre[j]) / alpha;

		// update x, alpha_pre and t_pre for next iteration
		delete[] x_pre;
		x_pre = x;
		alpha_pre = alpha;
		t_pre = t;
		int64_t iter_ed = wall_clock_ns();
		fun_obj->grad(x_pre, g);
		gnorm = dnrm2_(&n, g, &inc);
		info("iter= %d m=%d |g|=%5.3e f=%.14e acc=0.00%% passes=%d 1/t=%5.3e time=%.4es\n", i, 2, gnorm, f, counter,
				1/t, wall_time_diff(iter_ed, time_st));
		if (gnorm <= eps*gnorm1)
			break;
		time_st = wall_clock_ns();
		i++;
		x = new double[n];
	}
	delete[] y;
	delete[] v;
	memcpy(w,x,sizeof(double) * n);
	delete[] x;
}

void AFG::set_print_string(void (*print_string) (const char *buf))
{
	afg_print_string = print_string;
}

