#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>
#include <unistd.h>
#include <assert.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#ifndef __unix__
#include <sys/time.h>
#endif

#include "util.h"

#define DEFAULT_N_FNODE 1024
#define MIN(x,y) (((x)<(y))?(x):(y))

const char *space = " \t";
const char *ending = "#\r\n";

// NOTE Should be consistent with ending.
int is_ending(char c)
{
        return c=='#' || c=='\r' || c=='\n';
}

//#define USE_MY_STRTOL

int my_strtol(const char *restrict p, char **restrict endptr)
{
        const char *p0 = p;
        int64_t s = 0;

        int sign = 1;
        if(*p == '-')
                sign = -1, p++;
        else if(*p == '+')
                sign = 1, p++;

        for(int i=0; '0' <= *p && *p <= '9' && i<10; p++, i++){
                s = s*10 + (*p - '0');
        }
        s = sign * s;

        if(INT_MIN <= s && s <= INT_MAX){
                *endptr = (char *)p;
                return (int)s;
        }else{
                *endptr = (char *)p0;
                return 0;
        }
}

#define NS_PER_SEC 1000000000
int64_t wall_clock_ns()
{
#ifdef __unix__
	struct timespec tspec;
	int r = clock_gettime(CLOCK_MONOTONIC, &tspec);
	assert(r==0);
	return tspec.tv_sec*NS_PER_SEC + tspec.tv_nsec;
#else
	struct timeval tv;
	int r = gettimeofday( &tv, NULL );
	assert(r==0);
	return tv.tv_sec*NS_PER_SEC + tv.tv_usec*1000;
#endif
}

double wall_time_diff(int64_t ed, int64_t st)
{
	return (double)(ed-st)/(double)NS_PER_SEC;
}

#define DEBUG
void INFO(const char *tmpl, ...)
{
#ifdef DEBUG
	va_list ap;
	va_start(ap, tmpl);
	vfprintf(stderr, tmpl, ap);
	fprintf(stderr, "\n");
	va_end(ap);
#endif
}

// read an instance till end of line
// return bytes read
int read_instance(const char *s, Fnode **ins, int *label_len)
{
        const char *s0 = s;

        // init feature buffer
        size_t len = 0;
	size_t cap = DEFAULT_N_FNODE;
        Fnode *p = (Fnode*) malloc(cap * sizeof(Fnode));

        // skip label
	int before_space = (int)strcspn(s, space);
        int before_ending = (int)strcspn(s, ending);
        int before_feature = (int)strcspn(s, ":");
        int possible_label_end = MIN(before_space, before_ending);
        if(possible_label_end > before_feature){
                *label_len = 0;
        }else{
                *label_len = possible_label_end+1;
        }
        s += *label_len;
        s += strspn(s, space);

	while(1){

                // increase cap if out of space
		if(len >= cap){
			cap *= 2;
			p = (Fnode*) realloc(p,
                                        cap * sizeof(Fnode) );
		}
		Fnode *node = &p[len];

                if( *s == '\0' || is_ending(*s)){
                        node->index = -1;
                        node->value = 0;
                        len++;
                        break;
                }

                // get the index
		char *endptr = NULL;
		errno = 0;
#ifdef USE_MY_STRTOL
                node->index = (int)my_strtol(s, &endptr);
#else
		node->index = (int)strtol(s, &endptr, 10);
#endif
		if(errno != 0 || s == endptr){
                        free(p);
			return 0;
		}else if(*endptr != ':'){
                        free(p);
                        return 0;
                }else if(node->index <= 0){
                        free(p);
                        return 0;
                }else{
                        s = endptr+1;
                }

                // get the value
                errno = 0;
                node->value = strtod(s, &endptr);
                if(errno != 0 || s == endptr){
                        free(p);
                        return 0;
                }else{
                        len++;
                        s = endptr;
                }

                s += strspn(s, space);
	}
        
        // skip comments and '\r'
        if(*s == '\n'){
                s += 1;
        }else if(*s){
                s += strcspn(s, "\n");
                if(*s == '\n'){
                        s += 1;
                }
        }

        *ins = (Fnode *) realloc(p, len * sizeof(Fnode));

	return (int)(s-s0);
}

int last_feature_index(const char *s)
{
        int before_ending = (int)strcspn(s, ending);
        const char *p = s + before_ending;
        for(; p > s && *p != ':'; p--)
                ;
        if(p==s)
                return 0;
        p--;
        for(; p > s && isdigit(*p); p--)
                ;
        char *endptr = NULL;
        errno = 0;
#ifdef USE_MY_STRTOL
        int index = (int)my_strtol(p, &endptr);
#else
        int index = (int)strtol(p, &endptr, 10);
#endif
        if(errno != 0 || s == endptr){
                return 0;
        }
        return index;
}

void do_data_statistics(FILE *fp, int *l, int *n, int *nr_fnode)
{
        *l = 0;
        *n = 0;
        *nr_fnode = 0;
        const char* line;
        size_t fp_set = (size_t)ftell(fp);
        while((line = readline(fp))){
                int last_index = last_feature_index(line);
                if(*n < last_index)
                        *n = last_index;

                const char *p = line;
                for(; *p != '\0' && !is_ending(*p); p++){
                        if(*p == ':')
                                (*nr_fnode) ++;
                }
                (*nr_fnode)++; // for guardian -1
                (*l)++;
        }
        (*n) ++;
        fseek(fp, (int)fp_set, SEEK_SET);
}

typedef 
int (*DATA_FILTER)(void *ctx, const Fnode *xin,
                const Fnode **sparse_xout, const double **dense_xout);

struct read_and_filter_ctx
{
        struct Vector *sparse_pool;
        struct Vector *dense_pool;
        struct Vector *y;

        int ns, nd;
        int sparse_has_bias;
        int dense_has_bias;
        double bias;

        DATA_FILTER filter_func;
        void *filter_func_ctx;
};

int data_saver(void *_ctx, const char *line)
{
        struct read_and_filter_ctx* ctx = (struct read_and_filter_ctx*)_ctx;

        // read instance
        Fnode *sparse_x;
        int label_len;
        int bytes_read = read_instance(line, &sparse_x, &label_len);
        if(bytes_read == 0){
                fprintf(stderr, "Error in line %d\n", (int)ctx->y->len+1);
                return -1;
        }
        
        // read label
        int label = NO_LABEL;
        if(label_len != 0){
                sscanf(line, "%d", &label);
        }
        vector_push_elements(ctx->y, &label, 1);

        // filter sparse_x
        const Fnode *sparse_xout = NULL;
        const double *dense_xout = NULL;
        ctx->filter_func(ctx->filter_func_ctx, sparse_x, &sparse_xout, &dense_xout);

        // if bias > 0:
        //      if sparse & dense: save bias in dense
        //      else: save bias in sparse
        
        // save sparse_xout
        if(sparse_xout != NULL){
                size_t len = 0;
                const Fnode *x = sparse_xout;
                for(; x->index != -1; x++, len++){
                        if(ctx->ns <= x->index)
                                ctx->ns = x->index+1;
                }
                len++; // for guadian -1

                if(ctx->sparse_has_bias){
                        Fnode p = {.index=0, .value=ctx->bias};
                        vector_push_elements(ctx->sparse_pool, &p, 1);
                }
                vector_push_elements(ctx->sparse_pool, sparse_xout, len);
        }

        // save dense_xout
        if(dense_xout != NULL){
                size_t len = (size_t)ctx->nd;

                if(ctx->dense_has_bias){
                        double p = 0;
                        vector_push_elements(ctx->dense_pool, &p, 1);
                }
                vector_push_elements(ctx->dense_pool, dense_xout, len);
        }

        free(sparse_x);

        return 0;
}

int read_and_filter(FILE *fin, Data *data, int nd, double bias,
                DATA_FILTER filter_func, void *filter_func_ctx)
{
        struct read_and_filter_ctx ctx = {
                .y = make_vector(sizeof(*data->y)),
                .sparse_pool = make_vector(sizeof(**data->sparse_x)),
                .dense_pool = make_vector(sizeof(**data->dense_x)),
                .ns = 0, 
                .nd = nd,
                .sparse_has_bias = (nd==0 && bias > 0),
                .dense_has_bias =  (nd!=0 && bias > 0),
                .bias = bias,
                .filter_func = filter_func,
                .filter_func_ctx = filter_func_ctx
        };

        int ret = for_each_line_show_progress(fin, data_saver, &ctx);
        if(ret != 0)
                return ret;
        
        // assign data
        data->l = (int)ctx.y->len;

        if(ctx.dense_pool->len != 0){
                vector_shrink_to_fit(ctx.dense_pool);

                data->dense_x = (double**)malloc((size_t)data->l * sizeof(*data->dense_x));
                size_t size = ctx.dense_pool->size;
                size_t instance_size;
                if(ctx.dense_has_bias)
                        instance_size = (size_t)(1+nd) * size;
                else
                        instance_size = (size_t)nd * size;

                assert(ctx.dense_pool->len * size == (size_t)data->l * instance_size);

                for(int i=0; i<data->l; i++)
                        data->dense_x[i] = (double*)&ctx.dense_pool->data[(size_t)i * instance_size];
        }else{
                data->dense_x = NULL;
                free(ctx.dense_pool->data);
        }

        if(ctx.sparse_pool->len != 0){
                vector_shrink_to_fit(ctx.sparse_pool);

                data->sparse_x = (Fnode**)malloc((size_t)data->l * sizeof(*data->sparse_x));

                size_t size = ctx.sparse_pool->size;
                int begin = 1;
                int i=0, j=0;
                for(; i<ctx.sparse_pool->len; i++){
                        Fnode *p = (Fnode*)&ctx.sparse_pool->data[(size_t)i * size]; 
                        if(begin){
                                assert(j < data->l);
                                data->sparse_x[j] = p;
                                j++;
                                begin = 0;
                        }
                        if(p->index == -1)
                                begin = 1;
                }

                assert(begin == 1 && i == ctx.sparse_pool->len && j == data->l);
        }else{
                data->sparse_x = NULL;
                free(ctx.sparse_pool->data);
        }

        data->ns = ctx.ns;
        data->nd = ctx.nd;
        data->nr_fnode = (int)ctx.sparse_pool->len;
        data->bias = bias;
        data->n = data->ns + data->nd;
        data->y = (int*)ctx.y->data;

        free(ctx.sparse_pool);
        free(ctx.dense_pool);
        free(ctx.y);
        return 0;
}

int read_data_filter(void *ctx, const Fnode *xin, const Fnode **sparse_xout, const double **dense_xout)
{
        *sparse_xout = xin;
        *dense_xout = NULL;

        return 0;
}

void transpose(Data *data)
{
        int *nfeat = (int*)calloc((size_t)data->n, sizeof(*nfeat));
        for(int i=0; i<data->l; i++){
                const Fnode *xi = data->sparse_x[i];
                for(; xi->index != -1; xi++)
                        nfeat[xi->index]++;
        }
        for(int i=0; i<data->n; i++) // for ending
                nfeat[i] ++;
        for(int i=1; i<data->n; i++)
                nfeat[i] += nfeat[i-1];

        fprintf(stderr, "n %d l %d orig %d transport %d\n", data->n, data->l, data->nr_fnode, nfeat[data->n-1]);
        Fnode *tdata = (Fnode *)malloc((size_t)nfeat[data->n-1] * sizeof(*tdata));
        Fnode end = {.index = -1, .value= 0};
        for(int i=0; i<data->n; i++){
                tdata[--nfeat[i]] = end;
        }
        for(int i=0; i<data->l; i++){
                const Fnode *xi = data->sparse_x[i];
                for(; xi->index != -1; xi++){
                        tdata[--nfeat[xi->index]] = (Fnode){.index = i, .value = xi->value};
                }
        }
        free(data->sparse_x[0]);
        free(data->sparse_x);
        data->sparse_x = malloc((size_t)data->n * sizeof(*data->sparse_x));
        for(int i=0; i<data->n; i++){
                data->sparse_x[i] = tdata + nfeat[i];
        }
        free(nfeat);
}

int read_data(FILE *fin, double bias, Data *data)
{
        return read_and_filter(fin, data, 0, bias, read_data_filter, NULL);
}

int max_line_len = 1024;
char *line;
const char* readline(FILE *input)
{
        int len;
    
        if (line == NULL)
                line = (char *) malloc((size_t)max_line_len);
        if (fgets(line, max_line_len, input) == NULL)
                return NULL;

        while (strrchr(line,'\n') == NULL) {
                max_line_len *= 2;
                line = (char *) realloc(line, (size_t)max_line_len);
                len = (int) strlen(line);
                if (fgets(line+len, max_line_len-len, input) == NULL)
                        break;
        }
        return line;
}

void show_progress(int64_t time_st, size_t fin_len, size_t fin_set, size_t fin_now)
{
        int64_t time_now = wall_clock_ns();
        size_t fin_diff = fin_now - fin_set;
        double fperc = (double)(fin_diff)*1./(double)fin_len;
        double ETC = (1-fperc)*wall_time_diff(time_now, time_st)
                /(fperc+1e-3);
        fprintf(stderr, "\r%.0lf%% finished\t"
                        "%.0lf secs remains (ETC)"
                        "          ",
                        fperc*100., ETC);
}

FILE *fopen_or_abort(const char *fname, const char *mode)
{
        FILE* fp = fopen(fname, mode);
        if(!fp){
                perror(fname);
                exit(1);
        }
        return fp;
}

int for_each_line_show_progress(FILE *fin, int (*filter_func)(void *ctx, const char *line), void *ctx)
{
        const char *line;
        size_t fin_set = (size_t)ftell(fin);
        fseek(fin, 0, SEEK_END);
        size_t fin_size = (size_t)ftell(fin) - fin_set;
        fseek(fin, (long)fin_set, SEEK_SET);
        int64_t time_st = wall_clock_ns();
        int64_t report_st = time_st;

        int l = 0;
        for(; (line = readline(fin)); l++){
                if(l%1000 == 0){
                        int64_t time_now = wall_clock_ns();
                        if(wall_time_diff(time_now, report_st) > 5){
                                show_progress(time_st, fin_size, fin_set, (size_t)ftell(fin));
                                report_st = time_now;
                        }
                }
                int ret = filter_func(ctx, line);
                if(ret < 0)
                        return ret;
        }

        for(int i=0; i<100; i++)
                fputc(' ', stderr);
        fputc('\r', stderr);
        return 0;
}

void free_data(Data *data)
{
        if(data == NULL)
                return;
        if(data->sparse_x != NULL){
                free(data->sparse_x[0]);
                free(data->sparse_x);
        }
        data->sparse_x = NULL;
        if(data->dense_x != NULL){
                free(data->dense_x[0]);
                free(data->dense_x);
        }
        data->dense_x = NULL;
        free(data->y);
        data->y = NULL;
}

void save_data(FILE *fout, Data data)
{
        fwrite(&data.nr_fnode , sizeof(data.nr_fnode), 1, fout);
        fwrite(&data.l, sizeof(data.l), 1, fout);
        fwrite(&data.n, sizeof(data.n), 1, fout);
        fwrite(&data.ns, sizeof(data.ns), 1, fout);
        fwrite(&data.nd, sizeof(data.nd), 1, fout);

        if(data.ns != 0){
                int len = data.nr_fnode;
                fwrite(data.sparse_x[0], sizeof(**data.sparse_x), (size_t)len, fout);
        }
        if(data.nd != 0){
                int len = data.l * data.nd;
                fwrite(data.dense_x[0], sizeof(**data.dense_x), (size_t)len, fout);
        }
        fwrite(data.y, sizeof(*data.y), (size_t)data.l, fout);
}

Data load_data(FILE *fin)
{
        Data data;
        fread(&data.nr_fnode, sizeof(data.nr_fnode), 1, fin);
        fread(&data.l, sizeof(data.l), 1, fin);
        fread(&data.n, sizeof(data.n), 1, fin);
        fread(&data.ns, sizeof(data.ns), 1, fin);
        fread(&data.nd, sizeof(data.nd), 1, fin);

        if(data.ns == 0){
                data.sparse_x = NULL;
        }else{
                int len = data.nr_fnode;
                Fnode *x = (Fnode *)calloc((size_t)len, sizeof(*x));
                fread(x, sizeof(*x), (size_t)len, fin);

                data.sparse_x = (Fnode**)calloc((size_t)data.l, sizeof(*data.sparse_x));
                for(int i=0, j=0; i<data.l && j<len; i++){
                        data.sparse_x[i] = x;
                        for(; j<len && x->index != -1; j++, x++)
                                ;
                        j++, x++;
                }
        }
        if(data.nd == 0){
                data.dense_x = NULL;
        }else{
                int len = data.l * data.nd;
                double *x = (double *)calloc((size_t)len, sizeof(*x));
                fread(x, sizeof(*x), (size_t)len, fin);

                data.dense_x = (double**)calloc((size_t)data.l, sizeof(*data.dense_x));
                for(int i=0; i<data.l; i++){
                        data.dense_x[i] = &x[i * data.nd];
                }
        }

        data.y = (int*) calloc((size_t)data.l, sizeof(*data.y));
        fread(data.y, sizeof(*data.y), (size_t)data.l, fin);

        return data;
}

void mmap_data(const char *data_fname, Data *data)
{
        int fd = open(data_fname, O_RDONLY);
        if(fd == -1){
                perror(data_fname);
                exit(1);
        }
        struct stat sb;
        if(fstat(fd, &sb) == -1){
                perror("fstat");
                exit(1);
        }
        if(!S_ISREG(sb.st_mode)){
                fprintf(stderr, "%s not a regular file\n", data_fname);
                exit(1);
        }
        void *p = mmap(0, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if(p == MAP_FAILED){
                perror("mmap");
                exit(1);
        }
        if(close(fd) == -1){
                perror("close");
                exit(1);
        }
}

const char *basename(const char *path)
{
        int i = (int)strlen(path);
        i--;
        for(; i>=0 && (path[i] != '/' && path[i] != '\\'); i--)
                ;
        return path+i+1;
}

FILE *LOG_FILE = NULL;
int TO_LOG = 0;
void start_logging(const char *log_name)
{
        log_name = basename(log_name);
        char date_str[100], fname[FNAME_MAX];
        time_t date = time(NULL);
        mkdir("log", 0755);
        strftime(date_str, 100, "%d-%m-%Y-%H-%M-%S", gmtime(&date));
        sprintf(fname, "log/%s-%s.log", log_name, date_str);
        if(LOG_FILE)
                fclose(LOG_FILE);
        LOG_FILE = fopen(fname, "w");
        if(LOG_FILE == NULL){
                perror(fname);
                exit(1);
        }
        TO_LOG = 1;
}

void lprintf(const char *format, ...)
{
        va_list args;

        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end(args);
        if(TO_LOG){
                va_start(args, format);
                vfprintf(LOG_FILE, format, args);
                va_end(args);
        }
}

void end_logging()
{
        fclose(LOG_FILE);
        TO_LOG = 0;
}

#define DEFAULT_VECTOR_CAP 1024

struct Vector* make_vector(size_t size)
{
        struct Vector *v = calloc(1, sizeof(*v));
        v->size = size;
        v->cap = DEFAULT_VECTOR_CAP;
        v->data = calloc(v->cap, size);
        return v;
}
void* vector_push_elements(struct Vector *v, const void *elements, size_t n)
{

        while(v->len + n > v->cap){
                v->cap += v->cap/2;
                v->data = (char*) realloc(v->data, v->cap * v->size);
        }
        char *p = v->data + v->len * v->size;
        memcpy(p, elements, n * v->size);
        v->len += n;

        return p;
}
void vector_shrink_to_fit(struct Vector *v)
{
        v->cap = v->len;
        v->data = realloc(v->data, v->cap * v->size);
}

double max(double x, double y)
{
        return (x>y)?x:y;
}

double min(double x, double y)
{
        return (x<y)?x:y;
}

double ddot(const double *x, const double *y, int l)
{
#if 0
        double s = 0;
        for(int i=0; i<l; i++)
                s += x[i] * y[i];
        return s;
#else
        double s = 0;
        int m = l-4;
        int i;
        for (i = 0; i < m; i += 5)
                s += x[i] * y[i] + x[i+1] * y[i+1] + x[i+2] * y[i+2] +
                        x[i+3] * y[i+3] + x[i+4] * y[i+4];

        for ( ; i < l; i++)        /* clean-up loop */
                s += x[i] * y[i];

        return s;
#endif
}

double ddot3(const double *x, const double *y, const double *z, int l)
{
#if 0
        double s = 0;
        for(int i=0; i<l; i++)
                s += (x[i] * y[i] * z[i]);
        return s;
#else
        double s = 0;
        int m = l-4;
        int i;
        for (i = 0; i < m; i += 5)
                s +=    x[i+0] * y[i+0] * z[i+0] +
                        x[i+1] * y[i+1] * z[i+1] +
                        x[i+2] * y[i+2] * z[i+2] +
                        x[i+3] * y[i+3] * z[i+3] +
                        x[i+4] * y[i+4] * z[i+4] ;

        for ( ; i < l; i++)        /* clean-up loop */
                s += x[i] * y[i] * z[i];

        return s;
#endif
}

double dnrm2(const double *x, int l)
{
        double xx = ddot(x, x, l);
        return sqrt(xx);
}

void dcopy(double *x, double *y, int l)
{
        memcpy(y, x, sizeof(*x)*(size_t)l);
}

void dscal(double *x, double a, int l)
{
        int m = l-4;
        int i;
        for (i = 0; i < m; i += 5){
                x[i] *= a;
                x[i+1] *= a;
                x[i+2] *= a;
                x[i+3] *= a;
                x[i+4] *= a;
        }

        for ( ; i < l; i++)        /* clean-up loop */
                x[i] *= a;
}

void daxpy(double *restrict y, double a, const double *restrict x, int l)
{
#if 0
        for(int i=0; i<l; i++){
                y[i] += a * x[i];
        }
#else
        int m = l-3;
        int i;
        for (i = 0; i < m; i += 4)
        {
                y[i] += a * x[i];
                y[i+1] += a * x[i+1];
                y[i+2] += a * x[i+2];
                y[i+3] += a * x[i+3];
        }
        for ( ; i < l; ++i) /* clean-up loop */
                y[i] += a * x[i];
#endif
}

void dzero(double *v, int l)
{
        for(int i=0; i<l; i++){
                v[i] = 0;
        }
}

double sdot(const Fnode *xi, const double *v)
{
        double s = 0;
        for(; xi->index != -1; xi++)
                s += v[xi->index] * xi->value;
        return s;
}

double sdot_boundcheck(const Fnode *xi, const double *v, int max_index)
{
        double s = 0;
        for(; xi->index != -1; xi++){
                if(xi->index >= max_index)
                        continue;
                s += v[xi->index] * xi->value;
        }
        return s;
}

double snrm2(const Fnode *xi)
{
        double s = 0;
        for(; xi->index != -1; xi++){
                s += xi->value * xi->value;
        }
        return sqrt(s);
}

void saxpy(double *y, double a, const Fnode *x)
{
        for(; x->index != -1; x++)
                y[x->index] += a * x->value;
}

void swap_double(double *a, int i, int j)
{
        double t = a[i];
        a[i] = a[j];
        a[j] = t;
}

void swap_long_double(long double *a, int i, int j)
{
        long double t = a[i];
        a[i] = a[j];
        a[j] = t;
}

void swap_int(int *a, int i, int j)
{
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
}

#define MEPS 1e-16L
#define PIVOTING
#define DEBUG_LDL

// solve Ax=b , in which A[i*n+j] = Aij and we only solve submatrix A[m:m]
double LDL_with_pivoting(const double *_A, const double *_b, double *x, int n, int m)
{
        size_t bytes = ((size_t)(n*n) * sizeof(long double));
        long double *A = malloc(bytes);
        for(int i=0; i<n*n; i++)
                A[i] = _A[i];
        bytes = (size_t)n * sizeof(long double);
        long double *b = malloc(bytes);
        for(int i=0; i<n; i++)
                b[i] = _b[i];

#ifdef PIVOTING
        int *p = malloc((size_t)m * sizeof(*p));
        for(int i=0; i<m; i++)
                p[i] = i;
#endif

        // decomp
        // A -> D0
        //      L00 D1
        //      L10 L11 D2
        for(int j=0; j<m; j++){
#ifdef PIVOTING
                int pj = j;
                long double pj_val = fabsl(A[j*n+j]);
                // find piv
                for(int i=j+1; i<m; i++){
                        long double pi_val = fabsl(A[i*n+i]);
                        if(pj_val < pi_val){
                                pj_val = pi_val;
                                pj = i;
                        }
                }
                // swap rows and columns of j and pj
                for(int i=0; i<m; i++)
                        swap_long_double(A, j*n+i, pj*n+i);
                for(int i=0; i<m; i++)
                        swap_long_double(A, i*n+j, i*n+pj);
                swap_long_double(b, j, pj);
                swap_int(p, j, pj);
#endif
                
                // loop invariant: Aik = Lik, Akk = Dk with k<j
                long double Dj = A[j*n+j];
                for(int k=0; k<j; k++){
                        Dj -= A[j*n+k]*A[j*n+k] * A[k*n+k];
                }
                A[j*n+j] = Dj;
                if(fabsl(Dj) < MEPS){
                        for(int i=j+1; i<m; i++)
                                A[i*n+j] = 0;
                        continue; // xj should be 0
                }

                for(int i=j+1; i<m; i++){
                        long double Lij = A[i*n+j];
                        for(int k=0; k<j; k++){
                                Lij -= A[i*n+k] * A[j*n+k] * A[k*n+k];
                        }
                        Lij /= Dj;
                        A[i*n+j] = Lij;
                }
        }

        // invariant: A[i*n+j] = Lij, forall i>j
        //            A[i*n+i] = Dii, forall i
        
        // LD y = b
        // di yi =  bi - \sum_{j<i} yj Lij
        // place y in b
        for(int i=0; i<m; i++){
                // loop invariant: bk = yk forall k<i
                long double yi = b[i];
                for(int k=0; k<i; k++)
                        yi -= A[i*n+k] * b[k];
                b[i] = yi;
        }
        for(int i=0; i<m; i++){
                long double Di = A[i*n+i];
                if(fabsl(Di) < MEPS)
                        b[i] = 0;
                else
                        b[i] /= Di;
        }
        // L'x = y
        // xi = yi - \sum{j>i} xj Lij
        for(int i=m-1; i>=0; i--){
                // loop invariant: xk ready forall k>i
                long double xi = b[i];
#ifdef PIVOTING
                for(int k=m-1; k>i; k--)
                        xi -= A[k*n+i] * x[p[k]];
                x[p[i]] = (double)xi;
#else
                for(int k=m-1; k>i; k--)
                        xi -= A[k*n+i] * x[k];
                x[i] = (double)xi;
#endif
        }

#if 1
        double err = 0;
        for(int i=0; i<m; i++){
                double Ax_i = 0;
                for(int j=0; j<m; j++)
                        Ax_i += _A[i*n+j] * x[j];
                err += fabs(Ax_i-_b[i]);
        }
        double max_pv = 0;
        for(int i=0; i<m; i++)
                if(max_pv < _A[i*n+i])
                        max_pv = (double)_A[i*n+i];
        if(max_pv != 0)
                err /= max_pv;
#endif
        free(A);
        free(b);
#ifdef PIVOTING
        free(p);
#endif
        return err;
}

void permutation(int *a, int l)
{
        for(int i=0; i<l; i++)
                a[i] = i;

        for(int i=0; i<l; i++){
                int j = (int)random() % (l-i);
                swap_int(a, i, j);
        }
}

int number_of_w(int nr_class)
{
        if(nr_class == 2)
                return 1;
        else
                return nr_class;
}

void init_model(const Data data, const Param param, Model *model)
{
        int nr_class = 2;
        model->label = (int*) calloc((size_t)nr_class, sizeof(*model->label));

        model->label[0] = data.y[0];
        for(int i=1; i<data.l; i++){
                if(model->label[0] != data.y[i]){
                        model->label[1] = data.y[i];
                        break;
                }
        }

        for(int i=0; i<data.l; i++){
                int j;
                for(j=0; j<nr_class; j++)
                        if(data.y[i] == model->label[j])
                                break;
                if(j == nr_class){
                        nr_class++;
                        model->label = (int*)realloc(model->label, (size_t)nr_class*sizeof(*model->label));
                        model->label[nr_class-1] = data.y[i];
                }
        }
        fprintf(stderr, "nr_class=%d\n", nr_class);
        model->param = param;
        model->nr_class = nr_class;
        model->nr_feature = data.n;
        model->nr_sparse_feature = data.ns;
        model->nr_dense_feature = data.nd;
        model->bias = data.bias;
        int nr_w = number_of_w(nr_class);
        model->w = (double*) calloc((size_t)(data.n * nr_w), sizeof(*model->w));
}

void free_model(Model *model)
{
        if(model == NULL)
                return;
        free(model->label);
        model->label = NULL;
        free(model->w);
        model->w = NULL;
}
void save_model(FILE *f, const Model model)
{
        fprintf(f, "param.loss_type %d\n", model.param.loss_type);

        fprintf(f, "nr_class %d\n", model.nr_class);
        fprintf(f, "nr_feature %d\n", model.nr_feature);
        fprintf(f, "nr_sparse_feature %d\n", model.nr_sparse_feature);
        fprintf(f, "nr_dense_feature %d\n", model.nr_dense_feature);
        fprintf(f, "bias %.16g\n", model.bias);
        fprintf(f, "label");
        for(int i=0; i<model.nr_class; i++)
                fprintf(f, " %d", model.label[i]);
        fprintf(f, "\nw\n");
        int nr_w = number_of_w(model.nr_class);
        int total = nr_w * model.nr_feature;
        for(int i=0; i<total; i++){
                fprintf(f, "%.16g\n", model.w[i]);
        }
        fprintf(f, "\n");
}

void load_model(FILE *f, Model *model)
{
        char cmd[81];
        while(1){
                fscanf(f, "%80s", cmd);
                if(strcmp("param.loss_type", cmd) == 0){
                        fscanf(f, "%d", &model->param.loss_type);
                }else if(strcmp("nr_class", cmd) == 0){
                        fscanf(f, "%d", &model->nr_class);
                }else if(strcmp("nr_feature", cmd) == 0){
                        fscanf(f, "%d", &model->nr_feature);
                }else if(strcmp("nr_sparse_feature", cmd) == 0){
                        fscanf(f, "%d", &model->nr_sparse_feature);
                }else if(strcmp("nr_dense_feature", cmd) == 0){
                        fscanf(f, "%d", &model->nr_dense_feature);
                }else if(strcmp("bias", cmd) == 0){
                        fscanf(f, "%lf", &model->bias);
                }else if(strcmp("label", cmd) == 0){
                        model->label = (int*)calloc((size_t)model->nr_class, sizeof(*model->label));
                        for(int i=0; i<model->nr_class; i++){
                                fscanf(f, "%d", &model->label[i]);
                        } 
                }else if(strcmp("w", cmd) == 0){
                        int nr_w = number_of_w(model->nr_class);
                        int total = nr_w * model->nr_feature;
                        model->w = (double *)calloc((size_t)total, sizeof(*model->w));
                        for(int i=0; i<total; i++){
                                fscanf(f, "%lf", &(model->w[i]));
                        }
                        break;
                }else{
                        fprintf(stderr, "Model file corrupted\n");
                        fprintf(stderr, "%s\n", cmd);
                        exit(1);
                }
        }
}
