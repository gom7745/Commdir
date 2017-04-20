CC := gcc
CFLAGS := -O3 -pedantic -std=gnu99 -Wall -Wconversion -g -Werror -Wno-unused-result -lrt
LDFLAGS := -lm -g
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	LDFLAGS += -lrt
endif

all: ntrain npredict

util.o: util.c util.h
commdir.o: commdir.c commdir.h
ntrain: util.o commdir.o main.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
npredict: ntrain
	ln -sf ntrain npredict

.PHONY: clean
clean:
	rm -f -r *.dSYM *.o ntrain npredict
