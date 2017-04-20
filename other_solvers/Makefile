CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC -lrt
LIBS = blas/blas.a
SHVER = 2
OS = $(shell uname)
#LIBS = -lblas

all: train predict

lib: linear.o solver.o blas/blas.a
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,liblinear.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,liblinear.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} linear.o solver.o blas/blas.a -o liblinear.so.$(SHVER)

train: solver.o linear.o train.c blas/blas.a
	$(CXX) $(CFLAGS) -o train train.c solver.o linear.o $(LIBS)

predict: solver.o linear.o predict.c blas/blas.a
	$(CXX) $(CFLAGS) -o predict predict.c solver.o linear.o $(LIBS)

solver.o: solver.cpp solver.h
	$(CXX) $(CFLAGS) -c -o solver.o solver.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ solver.o linear.o train predict liblinear.so.$(SHVER)
