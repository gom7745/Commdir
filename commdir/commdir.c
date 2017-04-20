#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "util.h"

#define NORMALIZE

/* Consider the problem
 *
 * min_{w} 1/2 w'w + \sum_{i=1...l} loss(w'x_i, y_i)
 *
 */
// loss(double, void)

typedef struct Task Task;
struct Task{
        int n, l;
        int *y;
        double C;
        Loss loss;

        double *wx;
        double *D1, *D2;
};

// g = w + X' D^1, D^1_i = C y_i first_der(i)
// H = I + X' D^2 X, D^2_{ii} = C second_der(i)
void task_update(Task *t, double theta, double *dx)
{
        for(int i=0; i<t->l; i++){
                if(dx != NULL)
                        t->wx[i] = t->wx[i] + theta * dx[i];
                double zi = t->wx[i] * t->y[i];
                double first_der, second_der;
                t->loss.derivatives(zi, &first_der, &second_der);

                t->D1[i] = t->C * first_der * t->y[i];
                t->D2[i] = t->C * second_der;
        }
}

void task_init(Task *t, int n, int l, int *y, double C, Loss loss)
{
        t->n = n;
        t->l = l;
        t->y = y;
        t->C = C;
        t->loss = loss;

        t->wx = calloc((size_t)l, sizeof(*t->wx));
        t->D1 = calloc((size_t)l, sizeof(*t->D1));
        t->D2 = calloc((size_t)l, sizeof(*t->D2));

        task_update(t, 0, NULL);
}

void task_free(Task *t)
{
        free(t->wx);
        free(t->D1);
        free(t->D2);
}

typedef struct Direction Direction;
struct Direction {
        int n, l;
        int m, cap;
        double **p;
        double **px;
        double **pdot;
};

void direction_grow(Direction *dir, int cap)
{
        if(cap <= dir->cap)
                return;

        dir->p =    (double**) realloc(dir->p, (size_t)cap * sizeof(*dir->p));
        dir->px =   (double**) realloc(dir->px, (size_t)cap * sizeof(*dir->px));
        dir->pdot=  (double**) realloc(dir->pdot, (size_t)cap * sizeof(*dir->pdot));

        for(int i=dir->cap; i<cap; i++){
                dir->p[i] =  (double*) calloc((size_t)dir->n, sizeof(**dir->p));
                dir->px[i] = (double*) calloc((size_t)dir->l, sizeof(**dir->px));
                dir->pdot[i] = (double*) calloc((size_t)(i+1), sizeof(**dir->pdot));
        }

        dir->cap = cap;
}

void direction_free(Direction *dir)
{
        for(int i=0; i<dir->cap; i++){
                free(dir->p[i]);
                free(dir->px[i]);
                free(dir->pdot[i]);
        }
        free(dir->p);
        free(dir->px);
        free(dir->pdot);
        dir->cap = 0;
}

void direction_init(Direction *dir, int n, int l, int cap)
{
        dir->n = n, dir->l = l;
        dir->m = dir->cap = 0;
        dir->p = dir->px = dir->pdot = NULL;

        direction_grow(dir, cap);
}

void direction_push(Direction *dir, const double *p, const double *px)
{
        int m = dir->m, n = dir->n, l = dir->l;
        if(dir->cap <= m)
                direction_grow(dir, (int)(dir->cap * 1.5));

        memcpy(dir->p[m], p, (size_t)n * sizeof(*p));
        memcpy(dir->px[m], px, (size_t)l * sizeof(*px));

#ifdef NORMALIZE
        for(int i=0; i<m; i++){
                double t = ddot(dir->p[m], dir->p[i], n) / dir->pdot[i][i];
                daxpy(dir->p[m], -t, dir->p[i], n);
                daxpy(dir->px[m], -t, dir->px[i], l);
        }
        double pnorm = dnrm2(dir->p[m], n);
        dscal(dir->p[m], 1/pnorm, n);
        dscal(dir->px[m], 1/pnorm, l);

        dir->pdot[m][m] = ddot(dir->p[m], dir->p[m], n);
        for(int i=0; i<m; i++)
                dir->pdot[m][i] = 0;
        printf("pdot[m][m] = %.3e\n", dir->pdot[m][m]);
        if(pnorm < EPS){
                //dir->m = 0;
                return;
        }
#else
        for(int i=0; i<=m; i++){
                dir->pdot[m][i] = ddot(dir->p[i], dir->p[m], n);
        }
#endif
        dir->m++;
}

void direction_clear(Direction *dir)
{
        dir->m = 0;
}

int64_t do_gx(Task t, Data data, double *g, double *gx)
{
        int64_t time_st = wall_clock_ns();
        for(int i=0; i<data.l; i++){
                double v = 0;

                if(data.sparse_x != NULL && data.sparse_x[i] != NULL){
                        v += sdot(data.sparse_x[i], g);
                }
                if(data.dense_x != NULL && data.dense_x[i] != NULL){
                        v += ddot(g+data.ns, data.dense_x[i], data.nd);
                }

                gx[i] = v;
        }

        return wall_clock_ns() - time_st;
}

// g = w + X' D^1
int64_t do_g(Task t, Data data, double *w, double *g)
{
        int64_t time_st = wall_clock_ns();

        dzero(g, data.n);
        for(int i=0; i<data.l; i++){
                double a = t.D1[i];
                if(fabs(a) < EPS)
                        continue;

                if(data.sparse_x != NULL && data.sparse_x[i] != NULL){
                        saxpy(g, a, data.sparse_x[i]);
                }
                if(data.dense_x != NULL && data.dense_x[i] != NULL){
                        daxpy(g+data.ns, a, data.dense_x[i], data.nd);
                }
        }
        daxpy(g, 1, w, data.n);

        return wall_clock_ns() - time_st;
}

// H = I + X' D^2 X
// Hv = v + X'D^2(Xv)
void do_Hv(Task t, Data data, const double *v, double *Hv, double *vx)
{
        dzero(Hv, data.n);
        for(int i=0; i<data.l; i++){
                vx[i] = 0;
                if(data.sparse_x != NULL && data.sparse_x[i] != NULL)
                        vx[i] += sdot(data.sparse_x[i], v);
                if(data.dense_x != NULL && data.dense_x[i] != NULL)
                        vx[i] += ddot(data.dense_x[i], v, data.nd);

                double a = t.D2[i] * vx[i];
                if(fabs(a) < EPS)
                        continue;

                if(data.sparse_x != NULL && data.sparse_x[i] != NULL)
                        saxpy(Hv, a, data.sparse_x[i]);
                if(data.dense_x != NULL && data.dense_x[i] != NULL)
                        daxpy(Hv+data.ns, a, data.dense_x[i], data.nd);
        }
        daxpy(Hv, 1, v, data.n);
}

double eval_Pg(double *D1, double *w, double **p, double **px, double **pdot, int l, int n, int m, double *Pg)
{
        // construct Pg
        // p'g = p'(w + XD^1)
        for(int i=0; i<m; i++)
                Pg[i] = ddot(p[i], w, n) + ddot(D1, px[i], l);

        // eval normalized Pg norm
        double Pg_norm = 0;
        for(int i=0; i<m; i++){
                Pg_norm += Pg[i] * Pg[i] / pdot[i][i];
        }
        Pg_norm = sqrt(Pg_norm);
        return Pg_norm;
}

void eval_PHP(double *D2, double **px, double **pdot, int l, int n, int m, double *PHP)
{
        // construct PHP
        // H = I + X'D^2X
        // P'HP = P'P + (PX)'D^2(PX)
        for(int i=0; i<m; i++)
                for(int j=0; j<=i; j++)
                        PHP[i*m+j] = pdot[i][j];

        // perform pair-wise tripple dot
#if 1
        #define CSIZE 128
        int k;
        for(k=0; k<l/CSIZE; k++){
        for(int i=0; i<m; i++)
                for(int j=0; j<=i; j++)
                        PHP[i*m+j] += ddot3(D2+k*CSIZE, px[i]+k*CSIZE, px[j]+k*CSIZE, CSIZE);
        }

        for(int i=0; i<m; i++)
                for(int j=0; j<=i; j++)
                        PHP[i*m+j] += ddot3(D2+k*CSIZE, px[i]+k*CSIZE, px[j]+k*CSIZE, l-k*CSIZE);
#else
        for(int i=0; i<m; i++)
                for(int j=0; j<=i; j++)
                        PHP[i*m+j] += ddot3(D2, px[i], px[j], l);
#endif

        for(int i=0; i<m; i++)
                for(int j=i+1; j<m; j++)
                        PHP[i*m+j] = PHP[j*m+i];
}

double eval_pre(double *Pg, double *PHP, double *t, int m)
{

        // predicted decrease of 1/2 t'P'HPt + t'P'g
        double pre = 0;
        for(int i=0; i<m; i++)
                for(int j=0; j<m; j++)
                        pre += t[i]*PHP[i*m+j]*t[j];
        pre = pre/2;
        for(int i=0; i<m; i++)
                pre += t[i]*Pg[i];
        pre = -pre;

        return pre;
}

double function_value(Task t, double theta, double *dx, double wTw, double dTw, double dTd)
{
        double fval = wTw/2 + theta*dTw + theta*theta*dTd/2;
        for(int i=0; i<t.l; i++){
                double zi = t.wx[i] + theta * dx[i];
                zi = zi * t.y[i];
                double loss_value;
                t.loss.function(zi, &loss_value);
                fval += t.C * loss_value;
        }
        return fval;
}

// Goal: find theta such that
// f(x) - f(x+\theta d) >= \sigma/2 * \lambda \norm{\theta d}^2
void back_tracking_line_search(Task *t, double *w, double *d, double *dx, double *_theta, double *_fnew, double *act)
{

        int n = t->n;
        double wTw = ddot(w, w, n);
        double dTw = ddot(d, w, n);
        double dTd = ddot(d, d, n);
        double f0 = function_value(*t, 0, dx, wTw, dTw, dTd);

        double theta = 1;
        double beta = 0.4;
        double sigma = 1;
        double lambda = 0.5;
        double fnew;
        while(1){
                fnew = function_value(*t, theta, dx, wTw, dTw, dTd);

                if(f0 - fnew >= sigma/2 * lambda * theta*theta*dTd){
                        break;
                }else{
                        theta = theta * beta;
                }

                if(theta < EPS){
                        theta = 0;
                        fnew = f0;
                        break;
                }
        }
        *_theta = theta;
        *_fnew = fnew;
        *act = f0 - fnew;
}

#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"
void train_one(const Data data, const Param param, double *w)
{
        int max_iter = 1000;

        int64_t iter_st, iter_ed, one_st, one_time;
        int64_t g_time=0, co_time=0, cg_time=0;

        one_st = wall_clock_ns();

        int n = data.n, l = data.l;

        Direction dir;
#define DEFAULT_DIR_CAP 15
        direction_init(&dir, data.n, data.l, DEFAULT_DIR_CAP);
        Task task;
        task_init(&task, data.n, data.l, data.y, param.C, param.loss);

        double *d    = calloc((size_t)n, sizeof(*d));
        double *dx   = calloc((size_t)l, sizeof(*dx));
        double *g    = calloc((size_t)n, sizeof(*g));
        double *gx   = calloc((size_t)l, sizeof(*gx));

        // find threshold
        int pos=0, neg = 0;
        for(int i=0; i<l; i++){
                if(data.y[i]>0) pos++;
                else if(data.y[i]<0) neg++;
        }
        printf("pos=%d neg=%d\n", pos, neg);
        double eps = param.eps * max(min(pos, neg), 1)/l;

        double thres = -1;
        int iter;
        for(iter = 0; iter<max_iter; iter++){
                iter_st = wall_clock_ns();

                g_time += do_g(task, data, w, g);

                // stopping condition
                double gnorm = dnrm2(g, n);
                if(thres < 0){
                        thres = max(eps  * gnorm, EPS);
                        printf("\n\nthres = %.3e\n", thres);
                }
                if(gnorm < thres){
                        printf("ending gnorm=%.3e\n", gnorm);
                        break;
                }
                do_gx(task, data, g, gx);
                direction_push(&dir, g, gx);

                // line search & update
                double fval=0, acc=0;

                double thres_Pg_norm = min(gnorm*gnorm, gnorm*0.1);
                int nt = 0;
                int m = dir.m;
                double *PHP = (double*) calloc((size_t)(m*m), sizeof(*PHP));
                double *Pg = (double*) calloc((size_t)m, sizeof(*Pg));
                double *t = (double*) calloc((size_t)m, sizeof(*t));
                double **p = dir.p, **px = dir.px, **pdot = dir.pdot;
                printf("thresPg=%.3e gnorm=%.3e\n", thres_Pg_norm, gnorm);
                for(; param.max_inner==0 || nt < param.max_inner; nt++){
                        double Pg_norm;
                        Pg_norm = eval_Pg(task.D1, w, p, px, pdot, l, n, m, Pg);
                        if(Pg_norm <= thres_Pg_norm)
                                break;

                        int64_t time_st = wall_clock_ns();
                        eval_PHP(task.D2, px, pdot, l, n, m, PHP);
                        co_time += wall_clock_ns() - time_st;

                        // solve PHPt+Pg = 0
                        double err = LDL_with_pivoting(PHP, Pg, t, m, m);
                        dscal(t, -1, m);

                        double pre = eval_pre(Pg, PHP, t, m);
                        if(pre < 0){
                                direction_clear(&dir);
                                if(m == 1){
                                        printf("reach machine Epsilon\n");
                                        // exit outer loop
                                }
                                break;
                        }

                        dzero(d, n);
                        dzero(dx, l);
                        for(int i=0; i<m; i++){
                                daxpy(d, t[i], p[i], n);
                                daxpy(dx, t[i], px[i], l);
                        }
                        double act, theta;
                        back_tracking_line_search(&task, w, d, dx, &theta, &fval, &act);
                        if(theta < EPS){
                                break;
                        }

                        // update
                        daxpy(w, theta, d, n);
                        task_update(&task, theta, dx);

                        printf("|Pg|=%.3e, act=%.3e pre=%.3e err=%.3e theta=%.3e\n",
                                Pg_norm, act, pre, err, theta);
                        fflush(stdout);
                }
                free(PHP);
                free(Pg);
                free(t);

                // print iteration info
                iter_ed = wall_clock_ns();
                printf("iter=%3d m=%d |g|=%.3e f=%.14e acc=%.2f%% "
                        "time=%.4es\n",
                                iter+1, dir.m,
                                gnorm, fval, acc*100.,
                                wall_time_diff(iter_ed, iter_st));
                fflush(stdout);
        }
        one_time = wall_clock_ns() - one_st;
        printf("training=%.2gs: g=%.2lf%% co=%.2lf%% cg=%.2lf%%\n",
                        wall_time_diff(one_time, 0),
                        (double)g_time*100./(double)one_time,
                        (double)co_time*100./(double)one_time,
                        (double)cg_time*100./(double)one_time);

        direction_free(&dir);
        task_free(&task);
        free(d);
        free(dx);
        free(g);
}

void train(const Data data, const Param param, Model *model)
{
        init_model(data, param, model);

        Data two_class = data;
        two_class.y = (int*) calloc((size_t)data.l, sizeof(*data.y));

        int nr_w = number_of_w(model->nr_class);
        for(int i=0; i<nr_w; i++){
                printf("\nlabel = %d\n", model->label[i]);
                for(int j=0; j<data.l; j++){
                        if(model->label[i] == data.y[j])
                                two_class.y[j] = 1;
                        else
                                two_class.y[j] = -1;
                }
                double *w = model->w + i * data.n;
                train_one(two_class, param, w);
        }
        free(two_class.y);
}

void predict_value(const Model model, const Fnode *sparse_x, const double *dense_x, int *label, double *value)
{
        double max_z = -INFINITY;
        int predict_class = 0;
        int nr_w = number_of_w(model.nr_class);
        for(int i=0; i<nr_w; i++){
                double z = 0;
                if(sparse_x != NULL)
                        z += sdot_boundcheck(sparse_x, model.w, model.nr_sparse_feature);
                if(dense_x != NULL)
                        z += ddot(dense_x, model.w+model.nr_sparse_feature, model.nr_dense_feature);
                if(max_z < z){
                        max_z = z;
                        predict_class = i;
                }
        }
        *value = max_z;
        if(model.nr_class == 2){
                if(max_z >= 0)
                        *label = model.label[0];
                else
                        *label = model.label[1];
        }else{
                *label = model.label[predict_class];
        }
}

int predict(const Model model, const Fnode *sparse_x, const double *dense_x)
{
        int label;
        double value;
        predict_value(model, sparse_x, dense_x, &label, &value);

        return label;
}

int predict_accurate(const Model model, const Data data)
{
        int acc = 0;
        for(int i=0; i<data.l; i++){
                int predict_label, target_label;
                const Fnode *sparse_x = NULL;
                double *dense_x = NULL;
                if(data.sparse_x != NULL)
                        sparse_x = data.sparse_x[i];
                if(data.dense_x != NULL)
                        dense_x = data.dense_x[i];
                predict_label = predict(model, sparse_x, dense_x);
                target_label = data.y[i];
                if(predict_label == target_label)
                        acc++;
        }
        return acc;
}
