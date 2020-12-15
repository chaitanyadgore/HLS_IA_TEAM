/**
  ******************************************************************************
  * @file    conv.c
  * @author  Chaitanya Devidas Gore, Bogdan Mihai Nistor, Nelli Nyisztor, Université Côte d'Azur, France
  * @version V4.0
  * @date    07 september 2020
  * @brief   Convolutional layers for modified LeNet5
  */

#include <stdio.h>
#include <stdlib.h>

#include "lenet_cnn_float.h"

void Conv1_28x28x1_5x5x20_1_0(  unsigned char input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],          // IN [1][28][28]
                                short kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],  // IN [20][1][5][5]
                                short bias[CONV1_NBOUTPUT],                                     // IN [20]
                                short output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH])        // OUT [20][24][24]
{
    #pragma HLS RESOURCE variable=bias core=RAM_1P_LUTRAM
    #pragma HLS RESOURCE variable=output core=RAM_1P_LUTRAM

    unsigned short o,h,w,x,y;
    int conv_px_sum;

    // input array could not be partitioned with SDSoC
    // thus, introducing identical array input_to_partition
    unsigned short i,j,k;
    unsigned char input_to_partition[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=input_to_partition complete dim=3

    // fill temp array for partitioning
    for (i = 0; i < IMG_DEPTH; i++)
        for (j = 0; j < IMG_HEIGHT; j++)
            for (k = 0; k < IMG_WIDTH; k++)
                #pragma HLS pipeline
                input_to_partition[i][j][k] = input[i][j][k];

    // loop on output array dimensions
    for(o = 0; o < CONV1_NBOUTPUT; o++) { // 20
        for(h = 0; h < CONV1_HEIGHT; h++) { // 20*24 > 480
            for(w = 0; w < CONV1_WIDTH; w++) { // 20*24*24 > 11520 iteration

                // initialize sum for each pixel
                conv_px_sum = 0;

                #pragma HLS pipeline
                // convolution of 5x5 image part and kernel
                for(y = 0; y < CONV1_DIM; y++) {
                    for(x = 0; x < CONV1_DIM; x++) {
                        #pragma HLS RESOURCE variable=conv_px_sum core=MulnS latency=2
                        conv_px_sum = conv_px_sum + input_to_partition[0][h+y][w+x]*kernel[o][0][y][x];
                    }
                }

                // neuron activation
                if(conv_px_sum+bias[o]<=0) {
                    output[o][h][w]=0;
                } else {
                    // shifting back after matrix*kernel multiplication
                    conv_px_sum = conv_px_sum >> FIXED_POINT;
                    output[o][h][w]=conv_px_sum + bias[o];
                }
            }
        }
    }

}

void Conv2_12x12x20_5x5x40_1_0( short input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],             // IN [20][12][12]
                                short kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM], // IN [40][20][5][5]
                                short bias[CONV2_NBOUTPUT],                                         // IN [40]
                                short output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH])            // OUT [40][8][8]
{
    #pragma HLS RESOURCE variable=bias core=RAM_1P_LUTRAM
    #pragma HLS RESOURCE variable=output core=RAM_1P_LUTRAM

    #pragma HLS INLINE region recursive
    #pragma HLS UNROLL factor=4

    unsigned short f,d,o,h,w,x,y,oh,ow;
    int conv_px_sum;

    // input array could not be partitioned with SDSoC
    // thus, introducing identical array input_to_partition
    unsigned short i,j,k;
    short input_to_partition[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=input_to_partition complete dim=3

    // fill temp array for partitioning
    for (i = 0; i < POOL1_NBOUTPUT; i++)
        for (j = 0; j < POOL1_HEIGHT; j++)
            for (k = 0; k < POOL1_WIDTH; k++)
                input_to_partition[i][j][k] = input[i][j][k];

    // loop on output first dimension
    for(f=0; f<CONV2_NBOUTPUT; f++) { // 40
        // apply multiple kernels on image
        for(d=0; d<POOL1_NBOUTPUT; d++) { // 40*20 > 800

            for(h=0; h<CONV2_HEIGHT; h++) { // 40*20*8 > 6400
                for(w=0; w<CONV2_WIDTH; w++) { // 40*20*8*8 > 51200 iteration
                    // initialize sum for each pixel
                    conv_px_sum = 0;

                    #pragma HLS pipeline
                    // convolution of 5x5 image part and kernel
                    for(y = 0; y < CONV2_DIM; y++) {
                        for(x = 0; x < CONV2_DIM; x++) {
                            #pragma HLS RESOURCE variable=conv_px_sum core=MulnS latency=2
                            conv_px_sum = conv_px_sum + input_to_partition[d][h+y][w+x]*kernel[f][d][y][x];
                        }
                    }

                    // shifting back after matrix*kernel multiplication
                    conv_px_sum = conv_px_sum >> FIXED_POINT;

                    // to initialize first element
                    if(d==0) {
                        output[f][h][w] = conv_px_sum;
                    } else {
                        output[f][h][w]+= conv_px_sum;
                    }
                }
            }
        }

        // neuron activation
        for(oh=0; oh<CONV2_HEIGHT; oh++) {
            for(ow=0; ow<CONV2_WIDTH; ow++) {
                if(output[f][oh][ow]+bias[f]<=0) {
                    output[f][oh][ow]=0;
                } else {
                    output[f][oh][ow]=output[f][oh][ow] + bias[f];
                }
            }
        }
    }
}