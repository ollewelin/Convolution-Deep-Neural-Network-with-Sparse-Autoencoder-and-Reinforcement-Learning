# I find a good basic tutorial I like regarding understand MAKE here
# https://www.cs.bu.edu/teaching/cpp/writing-makefiles/
#-D WITH_CUDA=ON
CC = g++ -std=c++11 -O3
CFLAGS = -g -Wall
SRCS = main.cpp sparse_autoenc.hpp CIFAR_test_data.cpp CIFAR_test_data.h

PROG = OW_CNN_R

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

