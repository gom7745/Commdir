make -C other_solvers
make -C commdir
cd data
./get_data.sh
cd -
mkdir -f table log model figures/tikz 2> /dev/null
python single_multiple.py
python compare_methods.py
python single_multiple_bias.py
python compare_methods_bias.py
