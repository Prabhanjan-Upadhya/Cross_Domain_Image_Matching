CC=gcc
CFLAGS=-std=c99
LDLIBS=-L -lrt -lm -pg

CC2=nvcc
CFLAGS2=-arch sm_20
LDLIBS2=-L /opt/cuda-toolkit/5.5.22/lib64 -lcudart

all: project1.o serial.o
	$(CC2) $(CFLAGS2) -o matching.exe $(LDLIBS2) project1.o
	$(CC) $(CFLAGS) -o serial.out $(LDLIBS) serial.o

project1.o: project1.cu
	$(CC2) $(CFLAGS2) -c project1.cu

serial.o: serial.c
	$(CC) $(CFLAGS) -pg -c serial.c -lm -o serial.o
clean: 
	rm -rf *.o

