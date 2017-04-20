#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "util.h"
#include "commdir.h"

void logistic_loss_function(double z, double *value)
{
        double v;
        if(z>=0)
                v = log(1+exp(-z));
        else
                v = -z+log(1+exp(z));
        *value = v;
}

void logistic_loss_derivatives(double z, double *first, double *second)
{
        double sigma = 1/(1+exp(-z));
        *first = sigma - 1;
        *second = (1-sigma)*sigma;
}

struct Loss logistic_loss = {
        .function = logistic_loss_function,
        .derivatives = logistic_loss_derivatives
};

void l2_loss_function(double z, double *value)
{
        if(1-z <= 0)
                *value = 0;
        else
                *value = (1-z)*(1-z);
}

void l2_loss_derivatives(double z, double *first, double *second)
{
        if(1-z <= 0){
                *first = *second = 0;
        }else{
                *first = -2*(1-z);
                *second = 2;
        }
}

struct Loss l2_loss = {
        .function = l2_loss_function,
        .derivatives = l2_loss_derivatives
};

enum {L2_HINGE_LOSS, LOGISTIC_LOSS, NR_LOSS};
char loss_name[][100] = {
        "L2 Hinge Loss : max(1-yi w'xi)^2",
        "Logistic Loss : log(1+exp(-yi w'xi))"
};

Loss loss_from_type(int type)
{
        if(type == LOGISTIC_LOSS){
                return logistic_loss;
        }else if(type == L2_HINGE_LOSS){
                return l2_loss;
        }else{
                fprintf(stderr, "No such loss\n");
                exit(1);
        }
}

void main_predict_exit_with_help(const char *prog_name)
{
        printf("%s test_file model_file output_file\n", prog_name);
        exit(1);
}

int main_predict(int argc, const char **argv)
{
        const char *prog_name = argv[0];
        if(argc < 4)
                main_predict_exit_with_help(prog_name);
        FILE *test_file = fopen_or_abort(argv[1], "r");
        FILE *model_file = fopen_or_abort(argv[2], "r");
        FILE *output_file = fopen_or_abort(argv[3], "w");

        Model model;
        load_model(model_file, &model);

        Data data;
        read_data(test_file, model.bias, &data);

        int l_with_label = 0, acc = 0;
        for(int i=0; i<data.l; i++){
                const Fnode *sparse_x = NULL;
                double *dense_x = NULL;
                if(data.sparse_x != NULL)
                        sparse_x = data.sparse_x[i];
                if(data.dense_x != NULL)
                        dense_x = data.dense_x[i];

                int predict_label, target_label = data.y[i];
                double value;
                predict_value(model, sparse_x, dense_x, &predict_label, &value);
                fprintf(output_file, "%d\n", predict_label);

                if(target_label != NO_LABEL)
                        l_with_label ++;
                if(predict_label == target_label)
                        acc++;
        }
        if(l_with_label != 0)
                printf("Accuracy = %.3f%% (%d/%d)\n", acc*100./l_with_label, acc, l_with_label);
        free_data(&data);
        free_model(&model);
        fclose(test_file);
        fclose(model_file);
        fclose(output_file);

        return 0;
}

struct nheavy_param {
        const char *prog_name;

        const char *data_fname;
        FILE *data_file;
        const char *test_fname;
        FILE *test_file;
        char model_fname[FNAME_MAX];
        FILE *model_file;
        int to_log;

        int loss_type;
        double C;
        double eps;
        double bias;
        int bias_setted;
        int m;
        int max_inner;
};

void exit_with_help(struct nheavy_param *p)
{
        printf("%s [options] training_set_file [model_file]\n", p->prog_name);
        printf("\n");

        printf("-l loss_type : set type of loss function (default %d)\n", p->loss_type);
        for(int i=0; i<NR_LOSS; i++){
                printf("\t%d -- %s\n", i, loss_name[i]);
        }
        printf("\n");

        printf("-c cost   : set cost parameter (default %.2f)\n", p->C);
        printf("-e eps    : set eps of stopping condition (default %.2f)\n", p->eps);
        printf("-m m   : set number of descending direction (default auto tune)\n");
        printf("-B bias   : if bias > 0, instance becomes [x; bias];"
               "            (default -1)\n");
        printf("-i        : max inner iteration > 0 (default ulimited)\n");
        printf("\n");
        printf("-x test_set_file : set test set file (default none)\n");
        //printf("-z binary_data : use binary data; write one if not exists\n");
        printf("\n");
        printf("default stopping threshold : |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2\n");
        exit(1);
}

void parse_arguments(int argc, const char **argv, struct nheavy_param *p)
{
        struct nheavy_param def = {
                .prog_name = p->prog_name,
                .to_log = 0,
                .loss_type = L2_HINGE_LOSS,
                .C = 1,
                .eps = 0.01,
                .bias = -1,
                .bias_setted = 0,
                .m = 0,
                .max_inner = 0,
                .test_file = NULL,
                .test_fname = NULL
        };
        *p = def;

        int i;
        for(i=1; i<argc; i++){
                if(argv[i][0] != '-')
                        break;
                const char *cmd = argv[i]+1;
                if(++i >= argc)
                        exit_with_help(&def);
                const char *value = argv[i];
                if(strcmp("", cmd) == 0){
                }else if(strcmp("l", cmd) == 0){
                        int d = atoi(value);
                        if(!(0 <= d && d < NR_LOSS))
                                exit_with_help(&def);
                        p->loss_type = d;
                }else if(strcmp("c", cmd) == 0){
                        p->C = atof(value);
                }else if(strcmp("e", cmd) == 0){
                        p->eps = atof(value);
                }else if(strcmp("B", cmd) == 0){
                        p->bias = atof(value);
                        p->bias_setted = 1;
                }else if(strcmp("i", cmd) == 0){
                        p->max_inner = atoi(value);
                        if(p->max_inner <= 0)
                                exit_with_help(&def);
                }else if(strcmp("n", cmd) == 0){
                        p->m = atoi(value);
                        if(p->m < 0)
                                exit_with_help(&def);
                }else if(strcmp("x", cmd) == 0){
                        p->test_fname = value;
                        p->test_file = fopen_or_abort(value, "r");
                }else{
                        exit_with_help(&def);
                }
        }

        // open data file
        if(i >= argc)
                exit_with_help(&def);
        p->data_fname = argv[i];
        p->data_file = fopen_or_abort(p->data_fname, "r");
        i++;
        // open model file
        if(i < argc){
                strncpy(p->model_fname, argv[i], FNAME_MAX);
        }else{
                sprintf(p->model_fname, "%s.hmodel", p->data_fname);
        }
        p->model_file = fopen_or_abort(p->model_fname, "w");
}

int64_t read_data_and_summarize(FILE *fin, double bias, Data *data)
{
        int64_t time_st, time_ed;
        time_st = wall_clock_ns();

        read_data(fin, bias, data);

        int nnz = data->nd * data->l;
        if(data->sparse_x != NULL)
                nnz += data->nr_fnode - data->l;
        lprintf("l = %d n = %d nnz = %d (sparsity=%.2e, nnz/l=%.2e, nnz/n=%.2e)\n",
                        data->l, data->n, nnz,
                        nnz*1./data->l/data->n, nnz*1./data->l, nnz*1./data->n);

        fclose(fin);
        time_ed = wall_clock_ns();

        return time_ed - time_st;
}

int main_train(int argc, const char** argv)
{
        struct nheavy_param p;
        p.prog_name = argv[0];
        parse_arguments(argc, argv, &p);

        if(p.to_log)
                start_logging(p.data_fname);

        int64_t prog_st, prog_ed;
        int64_t loading_time, prog_time;
        prog_st = wall_clock_ns();

        Data data;
        loading_time = read_data_and_summarize(p.data_file, p.bias, &data);

        Loss loss = loss_from_type(p.loss_type);
        lprintf("Solving %s\n", loss_name[p.loss_type]);

        Param param = {
                .loss_type = p.loss_type,
                .loss = loss,
                .C = p.C,
                .eps = p.eps,
                .m = p.m,
                .max_inner = p.max_inner
        };
        Model model;
        train(data, param, &model);

        prog_ed = wall_clock_ns();
        prog_time = prog_ed - prog_st;
        lprintf("\nrunning time: %fs loading=%f%%\n", wall_time_diff(prog_ed, prog_st), 
                        (double)loading_time*100./(double)prog_time);
        return 0;

        int acc = predict_accurate(model, data);
        lprintf("accuracy = %.2f%%(%d/%d)\n", (double)acc/data.l*100, acc, data.l);

        if(p.test_file != NULL){
                Data test_data;
                read_data(p.test_file, p.bias, &test_data);

                acc = predict_accurate(model, test_data);
                lprintf("test accuracy = %.2f%%(%d/%d)\n", (double)acc/test_data.l*100, acc, test_data.l);

                free_data(&test_data);
        }

        save_model(p.model_file, model);
        fclose(p.model_file);

        free_data(&data);
        free_model(&model);
        end_logging();

        return 0;
}

int main(int argc, const char **argv)
{
        srandom(0);
        if(strcmp("npredict", basename(argv[0])) == 0)
                return main_predict(argc, argv);
        else
                return main_train(argc, argv);
}
