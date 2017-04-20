void train(const Data data, const Param param, Model *model);
void predict_value(const Model model, const Fnode *sparse_x, const double *dense_x, int *label, double *value);
int predict(const Model model, const Fnode *sparse_x, const double *dense_x);
int predict_accurate(const Model model, const Data data);
