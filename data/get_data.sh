#!/usr/bin/env bash

if [ ! -f a9a ]; then
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
fi

if [ ! -f covtype.libsvm.binary ]; then
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2
fi

if [ ! -f epsilon_normalized ]; then
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
fi

if [ ! -f kddb ]; then
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kddb.bz2
fi

if [ ! -f news20.binary ]; then
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2
fi

if [ ! -f rcv1_test.binary ]; then
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
fi

if [ ! -f url_combined ]; then
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/url_combined.bz2
fi

if [ ! -f webspam_wc_normalized_trigram.svm ]; then
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2
fi

bunzip2 *.bz2
