#ifndef _SOLVER_H
#define _SOLVER_H
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>

class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double * Hs) = 0 ;
	virtual double line_search(double *s, double ws, double *fnew, double eta, double init_size = 1.0) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){} ;
};

class LBFGS
{
public:
	LBFGS(const function *fun_obj, double eps = 0.1, int m=10, double eta = 0.25, int max_iter = 1000);
	~LBFGS();

	void lbfgs(double *w);
	void set_print_string(void (*i_print) (const char *buf));

private:
	void TwoLoop(double* g, double** s, double** y, double* rho, int DynamicM, double *step, int k, int n);
	double eps;
	double eta;
	int max_iter;
	int M;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*lbfgs_print_string)(const char *buf);
};

class NEWTON
{
public:
	NEWTON(const function *fun_obj, double eps = 0.1, double eta = 0.25, int max_iter = 1000);
	~NEWTON();

	void newton(double *w);
	void set_print_string(void (*i_print) (const char *buf));

private:
	int cg(double *g, double *s, double *r, double cg_eps = 0.1);
	double eps;
	double eta;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*newton_print_string)(const char *buf);
};

class AFG
{
public:
	AFG(const function *fun_obj, double eps = 0.1, double t_pre = 100, double mu= 1.0, int max_iter = 1000);
	~AFG();

	void afg(double *w);
	void set_print_string(void (*i_print) (const char *buf));
private:
	double t_pre;
	double eps;
	double mu;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*afg_print_string)(const char *buf);
};
#endif
