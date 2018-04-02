CC = g++
CFLAGS = -g -Wall
SRCS = main.cpp
PROG = OW_CNN_R

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
