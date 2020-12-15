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
//#include "hdf5.h"

// Xilinx time measurement
//#include "sds_lib.h"    

#include "lenet_cnn_float.h"

// Top Level HLS function
void lenet_cnn(	float 	input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH], 							// IN
				float 	conv1_kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],		// IN
				float 	conv1_bias[CONV1_NBOUTPUT], 						                // IN
				float 	conv2_kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM], // IN
				float 	conv2_bias[CONV2_NBOUTPUT], 						                // IN
				float 	fc1_kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],// IN
				float 	fc1_bias[FC1_NBOUTPUT],			 				                    // IN
				float 	fc2_kernel[FC2_NBOUTPUT][FC1_NBOUTPUT], 				            // IN
				float 	fc2_bias[FC2_NBOUTPUT], 						                    // IN
				float 	output[FC2_NBOUTPUT]) {							                    // OUT
  
  float	 	conv1_output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]; 
  float 	pool1_output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]; 
  float	 	conv2_output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]; 
  float 	pool2_output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]; 
  float 	fc1_output[FC1_NBOUTPUT]; 
  short 	k, y, x; 

  Conv1_28x28x1_5x5x20_1_0(input, conv1_kernel, conv1_bias, conv1_output); 
/*  printf("\nCONV1_WIDTH / CONV1_HEIGHT: %d / %d\n", CONV1_WIDTH, CONV1_HEIGHT); 
  WritePgmFile(output_filename, (float *)CONV1_OUTPUT[0], CONV1_WIDTH, CONV1_HEIGHT); 
  printf("\nConv1 output[0]: \n"); 
  for (y=0; y<CONV1_HEIGHT; y++) {
    for (x=0; x<CONV1_WIDTH; x++) 
      printf("%.2f ", conv1_output[0][y][x]); 
    printf("\n"); 
  }
*/

  Pool1_24x24x20_2x2x20_2_0(conv1_output, pool1_output); 
/*  printf("\nPOOL1_WIDTH / POOL1_HEIGHT: %d / %d\n", POOL1_WIDTH, POOL1_HEIGHT); 
  WritePgmFile(output_filename, (float *)POOL1_OUTPUT[0], POOL1_WIDTH, POOL1_HEIGHT); 
  printf("\nPool1 output[0]: \n"); 
  for (y=0; y<POOL1_HEIGHT; y++) {
    for (x=0; x<POOL1_WIDTH; x++) 
      printf("%.2f ", pool1_output[0][y][x]); 
    printf("\n"); 
  }
*/


  Conv2_12x12x20_5x5x40_1_0(pool1_output, conv2_kernel, conv2_bias, conv2_output); 
/*  printf("\nCONV2_WIDTH / CONV2_HEIGHT: %d / %d\n", CONV2_WIDTH, CONV2_HEIGHT); 
  WritePgmFile(output_filename, (float *)CONV2_OUTPUT[0], CONV2_WIDTH, CONV2_HEIGHT); 
  printf("\nConv2 output[0]: \n");
  for (y=0; y<CONV2_HEIGHT; y++) {
    for (x=0; x<CONV2_WIDTH; x++) 
      printf("%.2f ", conv2_output[0][y][x]); 
    printf("\n"); 
  }
*/

  Pool2_8x8x40_2x2x40_2_0(conv2_output, pool2_output); 
/*  printf("\nPOOL2_WIDTH / POOL2_HEIGHT: %d / %d\n", POOL2_WIDTH, POOL2_HEIGHT); 
  WritePgmFile(output_filename, (float *)POOL2_OUTPUT[15], POOL2_WIDTH, POOL2_HEIGHT); 
  printf("\nPool2 output[0]: \n"); 
  for (y=0; y<POOL2_HEIGHT; y++) {
    for (x=0; x<POOL2_WIDTH; x++) 
      printf("%.2f ", pool2_output[0][y][x]); 
    printf("\n"); 
  }
*/

  Fc1_40_400(pool2_output, fc1_kernel, fc1_bias, fc1_output); 
/*  printf("\n\nFc1 output[0..%d]: \n", FC1_NBOUTPUT-1);
  for (k = 0; k < FC1_NBOUTPUT; k++)
    printf("%f ", fc1_output[k]); 
*/

  Fc2_400_10(fc1_output, fc2_kernel, fc2_bias, output); 
/*  printf("\n\nFc2 output[0..%d]: \n", FC2_NBOUTPUT-1);
  for (k = 0; k < FC2_NBOUTPUT; k++)
    printf("%.2f ", output[k]); 
*/
}


// GLOBAL VARIABLES
unsigned char 	REF_IMG[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH]; 
float 			INPUT_NORM[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH]; 
float 			CONV1_KERNEL[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM]; 
float 			CONV1_BIAS[CONV1_NBOUTPUT]; 
float 			CONV2_KERNEL[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM]; 
float 			CONV2_BIAS[CONV2_NBOUTPUT]; 
float 			FC1_KERNEL[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]; 
float 			FC1_BIAS[FC1_NBOUTPUT]; 
float 			FC2_KERNEL[FC2_NBOUTPUT][FC1_NBOUTPUT]; 
float 			FC2_BIAS[FC2_NBOUTPUT]; 
float 			FC2_OUTPUT[FC2_NBOUTPUT]; 
float			SOFTMAX_OUTPUT[FC2_NBOUTPUT]; 

/**
  ******************************************************************************
  * @brief   main code deploying a LeNet inference CNN on MNIST dataset
  */

void main() {
  short 	x, y, z, k, m; 
  char 		*hdf5_filename = 		"lenet_weights.hdf5"; 
  char 		*conv1_weights = 		"conv2d_1/conv2d_1/kernel:0"; 
  char 		*conv1_bias = 			"conv2d_1/conv2d_1/bias:0"; 
  char 		*conv2_weights = 		"conv2d_2/conv2d_2/kernel:0"; 
  char 		*conv2_bias = 			"conv2d_2/conv2d_2/bias:0"; 
  char* 	fc1_weights = 			"dense_1/dense_1/kernel:0"; 
  char* 	fc1_bias = 				"dense_1/dense_1/bias:0"; 
  char* 	fc2_weights = 			"dense_2/dense_2/kernel:0"; 
  char* 	fc2_bias = 				"dense_2/dense_2/bias:0"; 
  char* 	test_labels_filename = 	"mnist/t10k-labels-idx1-ubyte"; 
//  char* 	test_labels_filename = 		"mnist/train-labels-idx1-ubyte"; 
//  char* 	output_filename = 		"output.pgm"; 
  FILE* 	label_file;
  int ret; 
  unsigned char label, number; 
  unsigned int 	error; 
  unsigned char labels_legend[10] = 		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; 
  char 		img_filename[120]; 
  char 		img_count[10]; 
  float 	max; 
  struct timeval start, end; 
  double 	tdiff, tmin, tmax, tavg; 
  unsigned long long xilinx_start, xilinx_end, xilinx_time, xilinx_time_max, xilinx_time_min, xilinx_time_avg; 

  printf("\e[1;1H\e[2J");

  printf("\nReading weights \n"); 
  ReadConv1Weights(hdf5_filename, conv1_weights, CONV1_KERNEL);
  ReadConv1Bias(hdf5_filename, conv1_bias, CONV1_BIAS); 
  ReadConv2Weights(hdf5_filename, conv2_weights, CONV2_KERNEL);
  ReadConv2Bias(hdf5_filename, conv2_bias, CONV2_BIAS); 
  ReadFc1Weights(hdf5_filename, fc1_weights, FC1_KERNEL);
  ReadFc1Bias(hdf5_filename, fc1_bias, FC1_BIAS);
  ReadFc2Weights(hdf5_filename, fc2_weights, FC2_KERNEL);
  ReadFc2Bias(hdf5_filename, fc2_bias, FC2_BIAS);
//WriteWeights("temp.txt", CONV1_KERNEL); 

  printf("\nOpening labels file \n"); 
  label_file = fopen( test_labels_filename, "r" );
  if (!label_file) {
    printf("Error: Unable to open file %s.\n", test_labels_filename);
    exit(1);
  }

  for (k = 0; k < 8; k++) // Skip 8 first header bytes
    ret = fscanf(label_file, "%c", &label); 
  
  printf("\nProcessing \n");
  m = 0; 		        // test image counter
  tavg = 0; 		    // average processing time (us)
  xilinx_time_avg = 0;  // Xilinx average processing time (cpu cycles)
  tmin = 1000000; 	    // minimum processing time (us)
  tmax = 0; 		    // maximum processing time (us)
  xilinx_time_min = 1e9;// Xilinx minimum processing time (cpu cycles)
  xilinx_time_max = 0;  // Xilinx maximum processing time (cpu cycles)
  error = 0; 		    // number of mispredictions

  // MAIN TEST LOOP
  gettimeofday(&start, NULL); 
  while (1) { 
//  for (x = 0; x < 1; x++) {

    ret = fscanf(label_file, "%c", &label); 
    if (feof(label_file)) break;

    strcpy(img_filename, "mnist/t10k-images-idx3-ubyte[");
//    strcpy(img_filename, "mnist/train-images-idx3-ubyte[");
    sprintf(img_count, "%d", m); 
    if 		    (m < 10) 	strcat(img_filename, "0000");
    else if 	(m < 100) 	strcat(img_filename, "000");
    else if 	(m < 1000) 	strcat(img_filename, "00");
    else if 	(m < 10000) 	strcat(img_filename, "0");
    strcat(img_filename, img_count);
    strcat(img_filename, "].pgm");

/**/    printf("\033[%d;%dH%s\n", 7, 0, img_filename);
//    printf("%s\n", img_filename);

    ReadPgmFile(img_filename, (unsigned char *)REF_IMG); 

    NormalizeImg((unsigned char *)REF_IMG, (float *)INPUT_NORM, IMG_WIDTH, IMG_WIDTH); 
/*  for (z = 0; z < IMG_DEPTH; z++)
    for (y=0; y<IMG_HEIGHT; y++) {
      for (x=0; x<IMG_WIDTH; x++)  
        printf("%.2f ", INPUT_NORM[z][y][x]);
      printf("\n");
    }
*/

////    xilinx_start = sds_clock_counter();

    lenet_cnn(	INPUT_NORM, 				
				CONV1_KERNEL, 		
				CONV1_BIAS, 		
				CONV2_KERNEL, 			
				CONV2_BIAS, 			
				FC1_KERNEL, 				
				FC1_BIAS, 				
				FC2_KERNEL,					
				FC2_BIAS,					
				FC2_OUTPUT); 

////    xilinx_end = sds_clock_counter(); 

    Softmax(FC2_OUTPUT, SOFTMAX_OUTPUT); 
/**/    printf("\n\nSoftmax output: \n");
    max = 0; 
    number = 0; 
    for (k = 0; k < FC2_NBOUTPUT; k++) {
/**/      printf("%.2f%% ", SOFTMAX_OUTPUT[k]*100); 
      if (SOFTMAX_OUTPUT[k] > max) {
        max = SOFTMAX_OUTPUT[k]; 
        number = k; 
      }
    }


/**/    printf("\n\nPredicted: %d \t Actual: %d\n", labels_legend[number], label); 
    if (labels_legend[number] != label) error = error + 1; 

    xilinx_time = xilinx_end - xilinx_start; 

    if (xilinx_time < xilinx_time_min) xilinx_time_min = xilinx_time; 
    if (xilinx_time > xilinx_time_max) xilinx_time_max = xilinx_time; 

    xilinx_time_avg = xilinx_time_avg + xilinx_time; 
    m++; 

  } // END MAIN TEST LOOP
  gettimeofday(&end, NULL); 

  tdiff = (double)(end.tv_sec-start.tv_sec); 
  printf("TOTAL PROCESSING TIME (gettimeofday): %f s\n", tdiff); 

  printf("\n\nErrors : %d / %d", error, m); 
  printf("\n\nSuccess rate = %f%%", (1-((float)error/m))*100); 

////  printf("\n\nThw_min = %lld cpu cycles \t Thw_max = %lld cpu cycles \t Thw_avg = %lld cpu cycles (Xilinx) ", xilinx_time_min, xilinx_time_max, xilinx_time_avg/m );

  printf("\n\n"); 

  fclose(label_file); 

}


