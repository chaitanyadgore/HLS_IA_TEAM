CC=gcc

IDIR = /usr/include/hdf5/serial/
CFLAGS = -I$(IDIR) -O3
LIBS = -lhdf5_serial -lm

lenet_cnn_float: lenet_cnn_float.o fc.o pool.o conv.o utils.o
	$(CC) -o lenet_cnn_float lenet_cnn_float.o fc.o pool.o conv.o utils.o $(LIBS)

lenet_cnn_float.o: lenet_cnn_float.c 
	$(CC) -c lenet_cnn_float.c $(CFLAGS)

fc.o: fc.c 
	$(CC) -c fc.c $(CFLAGS)

pool.o: pool.c 
	$(CC) -c pool.c $(CFLAGS)

conv.o: conv.c 
	$(CC) -c conv.c $(CFLAGS)

utils.o: utils.c 
	$(CC) -c utils.c $(CFLAGS)
	
clean: 
	rm -r lenet_cnn_float.o utils.o conv.o fc.o pool.o lenet_cnn_float
