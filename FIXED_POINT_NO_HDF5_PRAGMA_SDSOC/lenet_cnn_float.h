/**
  ******************************************************************************
  * @file    lenet_cnn_float.h
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @version V1.0
  * @date    04 february 2019
  * @brief   Plain C code for the implementation of Convolutional Neural Networks on FPGA
  * @brief   Designed to support Vivado HLS synthesis
  */


#define IMG_WIDTH	28
#define IMG_HEIGHT	28
#define IMG_DEPTH	1

#define CONV1_DIM	    5
#define CONV1_NBOUTPUT	20
#define CONV1_STRIDE	1
#define CONV1_PAD	    0
#define CONV1_WIDTH	    ( ( (IMG_WIDTH - CONV1_DIM + (2*CONV1_PAD) ) / CONV1_STRIDE ) + 1 )
#define CONV1_HEIGHT	( ( (IMG_HEIGHT - CONV1_DIM + (2*CONV1_PAD) ) / CONV1_STRIDE ) + 1 )

#define POOL1_DIM	    2
#define POOL1_NBOUTPUT	CONV1_NBOUTPUT
#define POOL1_STRIDE	2
#define POOL1_PAD	    0
#define POOL1_WIDTH	    ( ( (CONV1_WIDTH - POOL1_DIM + (2*POOL1_PAD) ) / POOL1_STRIDE ) + 1 )
#define POOL1_HEIGHT	( ( (CONV1_HEIGHT - POOL1_DIM + (2*POOL1_PAD) ) / POOL1_STRIDE ) + 1 )

#define CONV2_DIM	    5
#define CONV2_NBOUTPUT	40
#define CONV2_STRIDE	1
#define CONV2_PAD	    0
#define CONV2_WIDTH	    ( ( (POOL1_WIDTH - CONV2_DIM + (2*CONV2_PAD) ) / CONV2_STRIDE ) + 1 )
#define CONV2_HEIGHT	( ( (POOL1_HEIGHT - CONV2_DIM + (2*CONV2_PAD) ) / CONV2_STRIDE ) + 1 )

#define POOL2_DIM	    2
#define POOL2_NBOUTPUT	CONV2_NBOUTPUT
#define POOL2_STRIDE	2
#define POOL2_PAD	    0
#define POOL2_WIDTH	    ( ( (CONV2_WIDTH - POOL2_DIM + (2*POOL2_PAD) ) / POOL2_STRIDE ) + 1 )
#define POOL2_HEIGHT	( ( (CONV2_HEIGHT - POOL2_DIM + (2*POOL2_PAD) ) / POOL2_STRIDE ) + 1 )

#define FC1_NBOUTPUT	400

#define FC2_NBOUTPUT	10
#define FIXED_POINT		8

void ReadPgmFile(char *filename, unsigned char *pix); 
void WritePgmFile(char *filename, float *pix, short width, short height); 
void ReadTestLabels(char *filename, short size); 
void RescaleImg(unsigned char *input, short width,short height, float *output, short new_width, short new_height); 
void NormalizeImg(unsigned char *input, unsigned char *output, short width, short height);  

void Conv1_28x28x1_5x5x20_1_0(	unsigned char	input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH], 	                // IN
				                short 		    kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM], 	// IN
				                short 		    bias[CONV1_NBOUTPUT],						                // IN
				                short 		    output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]); 		// OUT


void Pool1_24x24x20_2x2x20_2_0(	short 	input[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH], 	    // IN
				                short 	output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]);		// OUT

void Conv2_12x12x20_5x5x40_1_0(	short input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH], 	            // IN
				                short kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM], 	// IN
				                short bias[CONV2_NBOUTPUT], 						                    // IN
				                short output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]); 		        // OUT

void Pool2_8x8x40_2x2x40_2_0(	short 	input[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH], 	    // IN
				                short 	output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);		// OUT

void Fc1_40_400(	short 	input[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH], 			        // IN
			        short 	kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],	// IN
			        short 	bias[FC1_NBOUTPUT],							                        // IN
			        short 	output[FC1_NBOUTPUT]); 							                    // OUT

void Fc2_400_10(	short 	input[FC1_NBOUTPUT], 			        // IN
			        short 	kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],	    // IN
			        short 	bias[FC2_NBOUTPUT],			            // IN
			        short 	output[FC2_NBOUTPUT]); 			        // OUT

void Softmax(short vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT]); 

