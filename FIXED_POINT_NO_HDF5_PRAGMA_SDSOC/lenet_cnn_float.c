/**
  ******************************************************************************
  * @file    lenet_cnn_float.c
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @version V1.0
  * @date    04 february 2019
  * @brief   Plain C code for the implementation of Convolutional Neural Networks on FPGA
  * @brief   Designed to support Vivado HLS synthesis
  */

// LeNet
// Based on
// https://engmrk.com/lenet-5-a-classic-cnn-architecture/
// https://ml4a.github.io/ml4a/looking_inside_neural_nets/
// How will channels (RGB) effect convolutional neural network?
// https://www.researchgate.net/post/How_will_channels_RGB_effect_convolutional_neural_network

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// Xilinx time measurement
#include "sds_lib.h"

#include "lenet_cnn_float.h"
#include "weights.h"

// Top Level HLS function
void lenet_cnn(unsigned char input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH], // IN
               short output[FC2_NBOUTPUT])                            // OUT
{

  short conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
  short pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
  short conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
  short pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
  short fc1_output[FC1_NBOUTPUT];

  Conv1_28x28x1_5x5x20_1_0(input, CONV1_KERNEL, CONV1_BIAS, conv1_output);
  Pool1_24x24x20_2x2x20_2_0(conv1_output, pool1_output);
  Conv2_12x12x20_5x5x40_1_0(pool1_output, CONV2_KERNEL, CONV2_BIAS, conv2_output);
  Pool2_8x8x40_2x2x40_2_0(conv2_output, pool2_output);
  Fc1_40_400(pool2_output, FC1_KERNEL, FC1_BIAS, fc1_output);
  Fc2_400_10(fc1_output, FC2_KERNEL, FC2_BIAS, output);
}

// GLOBAL VARIABLES
unsigned char REF_IMG[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
unsigned char INPUT_NORM[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
short FC2_OUTPUT[FC2_NBOUTPUT];
float SOFTMAX_OUTPUT[FC2_NBOUTPUT];

/**
  ******************************************************************************
  * @brief   main code deploying a LeNet inference CNN on MNIST dataset
  */

int main()
{
  short k, m;
  char *test_labels_filename = "mnist/t10k-labels-idx1-ubyte";
  FILE *label_file;
  int ret;
  unsigned char label, number;
  unsigned int error;
  unsigned char labels_legend[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  char img_filename[120];
  char img_count[10];
  float max;
  struct timeval start, end;
  double tdiff, tmin, tmax, tavg;
  unsigned long long xilinx_start, xilinx_end, xilinx_time, xilinx_time_max, xilinx_time_min, xilinx_time_avg;

  printf("\e[1;1H\e[2J");

  printf("\nOpening labels file \n");
  label_file = fopen(test_labels_filename, "r");
  if (!label_file)
  {
    printf("Error: Unable to open file %s.\n", test_labels_filename);
    exit(1);
  }

  for (k = 0; k < 8; k++) // Skip 8 first header bytes
    ret = fscanf(label_file, "%c", &label);

  printf("\nProcessing \n");
  m = 0;                 // test image counter
  tavg = 0;              // average processing time (us)
  xilinx_time_avg = 0;   // Xilinx average processing time (cpu cycles)
  tmin = 1000000;        // minimum processing time (us)
  tmax = 0;              // maximum processing time (us)
  xilinx_time_min = 1e9; // Xilinx minimum processing time (cpu cycles)
  xilinx_time_max = 0;   // Xilinx maximum processing time (cpu cycles)
  error = 0;             // number of mispredictions

  // MAIN TEST LOOP
  gettimeofday(&start, NULL);
  while (1)
  {
    //  for (x = 0; x < 1; x++) {

    ret = fscanf(label_file, "%c", &label);
    if (feof(label_file))
      break;

    strcpy(img_filename, "mnist/t10k-images-idx3-ubyte[");
    //    strcpy(img_filename, "mnist/train-images-idx3-ubyte[");
    sprintf(img_count, "%d", m);
    if (m < 10)
      strcat(img_filename, "0000");
    else if (m < 100)
      strcat(img_filename, "000");
    else if (m < 1000)
      strcat(img_filename, "00");
    else if (m < 10000)
      strcat(img_filename, "0");
    strcat(img_filename, img_count);
    strcat(img_filename, "].pgm");

    /*printf("\033[%d;%dH%s\n", 7, 0, img_filename); */

    ReadPgmFile(img_filename, (unsigned char *)REF_IMG);

    NormalizeImg((unsigned char *)REF_IMG, (unsigned char *)INPUT_NORM, IMG_WIDTH, IMG_WIDTH);

    xilinx_start = sds_clock_counter();

    // main cnn function with reduced parameters (result of hdf5 removal)
    lenet_cnn(INPUT_NORM, FC2_OUTPUT);

    xilinx_end = sds_clock_counter();

    Softmax(FC2_OUTPUT, SOFTMAX_OUTPUT);
    /* printf("\n\nSoftmax output: \n"); */
    max = 0;
    number = 0;
    for (k = 0; k < FC2_NBOUTPUT; k++)
    {
      /* printf("%.2f%% ", SOFTMAX_OUTPUT[k] * 100); */
      if (SOFTMAX_OUTPUT[k] > max)
      {
        max = SOFTMAX_OUTPUT[k];
        number = k;
      }
    }

    /* printf("\n\nPredicted: %d \t Actual: %d\n", labels_legend[number], label); */
    if (labels_legend[number] != label)
      error = error + 1;

    xilinx_time = xilinx_end - xilinx_start;

    if (xilinx_time < xilinx_time_min)
      xilinx_time_min = xilinx_time;
    if (xilinx_time > xilinx_time_max)
      xilinx_time_max = xilinx_time;

    xilinx_time_avg = xilinx_time_avg + xilinx_time;
    m++;

  } // END MAIN TEST LOOP
  gettimeofday(&end, NULL);

  tdiff = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000;
  printf("TOTAL PROCESSING TIME (gettimeofday): %f s\n", tdiff);

  printf("\n\nErrors : %d / %d", error, m);
  printf("\n\nSuccess rate = %f%%", (1 - ((float)error / m)) * 100);

  printf("\n\nThw_min = %lld cpu cycles \t Thw_max = %lld cpu cycles \t Thw_avg = %lld cpu cycles (Xilinx) ", xilinx_time_min, xilinx_time_max, xilinx_time_avg/m );

  printf("\n\n");

  fclose(label_file);

  return 0;
}
