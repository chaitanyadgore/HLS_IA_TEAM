/**
  ******************************************************************************
  * @file    utils.c
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @version V1.0
  * @date    04 february 2019
  * @brief   Plain C code for the implementation of Convolutional Neural Networks on FPGA
  * @brief   Designed to support Vivado HLS synthesis
  */


#include <stdio.h>
#include <stdlib.h>

#include "lenet_cnn_float.h"
#include "hdf5.h"

void ReadPgmFile(char *filename, unsigned char *pix) {
  FILE* pgm_file; 
  int i, width, height, max, ret; 
  char readChars[256]; 

  pgm_file = fopen( filename, "rb" );
  if (!pgm_file) {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }

  ret = fscanf (pgm_file, "%s", readChars); 
  ret = fscanf (pgm_file, "%d", &width);
  ret = fscanf (pgm_file, "%d", &height);
  ret = fscanf (pgm_file, "%d", &max);
//  printf("Reading PGM file %s \t -> Type %s, width %d, height %d, max %d\n", filename, readChars, width, height, max);
//  if (width != IMG_WIDTH) printf("Warning: Image width mismatch (%d, expecting %d) \t -> Consider rescaling\n", width, IMG_WIDTH); 
//  if (height != IMG_HEIGHT) printf("Warning: Image height mismatch(%d, expecting %d) \t -> Consider rescaling\n", height, IMG_HEIGHT); 

  for (i = 0; i < width*height; i++) // DEBUG IF IMG_DEPTH > 1 ??
    ret = fscanf(pgm_file, "%c", &pix[i]); 

  fclose(pgm_file); 
}


void WritePgmFile(char *filename, float *pix, short width, short height) {
  FILE* pgm_file; 
  short i; 

  pgm_file = fopen( filename, "w" );
  if (!pgm_file) {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }
  fprintf (pgm_file, "P2\n"); 
  fprintf (pgm_file, "%d %d\n", width, height);
  fprintf (pgm_file, "255\n");

  for (i = 0; i < width*height; i++) { // DEBUG IF IMG_DEPTH > 1 ??
    fprintf(pgm_file, "%d ", (unsigned char)(pix[i]*64)); // *64 because pix values are too small
    if ( i%width == width-1 ) fprintf(pgm_file, "\n");  
  }

  fclose(pgm_file); 
}


void ReadTestLabels(char *filename, short size) {
  FILE* label_file; 
  int ret; 
  short k; 
  unsigned char label; 

  label_file = fopen( filename, "r" );
  if (!label_file) {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }

  for (k = 0; k < size; k++) {
    ret = fscanf(label_file, "%c", &label); 
    if (k >= 8) printf("img%d -> 0x%x \n" , k - 8, label); 
  }
  printf("\n"); 

  fclose(label_file); 
}


// Nearest neighbor, linear interpolation
// Based on 
// http://courses.cs.vt.edu/~masc1044/L17-Rotation/ScalingNN.html
#define min(a,b) ( (a) < (b) ? (a) : (b) )
void RescaleImg(unsigned char *input, short width,short height, float *output, short new_width, short new_height) {
  short x, y; 
  short interpol_x, interpol_y; 

  for (y=0; y<new_height; y++) {
    for (x=0; x<new_width; x++) {
      interpol_x = (short)( ((float)x/(float)new_width)*(float)width + 0.5 ); 
      interpol_x = min( interpol_x, width-1); 
      interpol_y = (short)( ((float)y/(float)new_height)*(float)height + 0.5 ); // MOVE TO Y LOOP
      interpol_y = min( interpol_y, height-1); 
      output[(y*new_width)+x] = input[(interpol_y*width)+interpol_x]; 
    }
  }
}

void NormalizeImg(unsigned char *input, float *output, short width, short height) {
  short x, y; 

  for (y=0; y<height; y++) 
    for (x=0; x<width; x++) 
      output[(y*width)+x] = ( (float)input[(y*width)+x] / 255 ); 

}


/* Used to generate weights */
void WriteWeights(char *filename, short weight[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM]) {
  FILE* 	weight_file; 
  short 	i, j, k, l; 

  weight_file = fopen( filename, "w" );
  if (!weight_file) {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }

  fprintf (weight_file, "short CONV1_KERNEL[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM] = { \n");
  for (i = 0; i < CONV1_NBOUTPUT; i++) {
    fprintf (weight_file, "{ \n");
    for (j = 0; j < IMG_DEPTH; j++) {
      fprintf (weight_file, "{ \n");
      for (k = 0; k < CONV1_DIM; k++) {
		fprintf (weight_file, "{ "); 
        for (l = 0; l < CONV1_DIM; l++)
	  	  fprintf(weight_file, "%d, ", weight[i][j][k][l]); 
	    fprintf (weight_file, "}, ");
      }
   	  fprintf (weight_file, "}, \n");
 	}
    fprintf (weight_file, "}, \n");
  }
  fprintf (weight_file, "}; \n");

  fclose(weight_file); 
}



void ReadConv1Weights(char *filename, char *datasetname, float weight[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM]) {
  unsigned short 	x, y, z, k; 
  float 	 		buffer_float[CONV1_DIM][CONV1_DIM][IMG_DEPTH][CONV1_NBOUTPUT]; // y, x, z, k
  hid_t 	 		file, dataspace, dataset; 
  herr_t 	 		status; 

  file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen (file, datasetname, H5P_DEFAULT);
  status = H5Dread (dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_float); 
  for (k = 0; k < CONV1_NBOUTPUT; k++)
    for (y = 0; y < CONV1_DIM; y++)
      for (x = 0; x < CONV1_DIM; x++) 
		for (z = 0; z < IMG_DEPTH; z++) 
		  weight[k][z][y][x] = buffer_float[y][x][z][k]; // re-ordering [y][x][z][k] -> [k][z][y][x]

  status = H5Dclose (dataset);
  status = H5Fclose (file);

}


void ReadConv1Bias(char *filename, char *datasetname, float *bias) {
  unsigned short 	k; 
  hid_t 	 		file, dataspace, dataset; 
  herr_t 	 		status; 
  float 	 		buffer_float[CONV1_NBOUTPUT]; 

  file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen (file, datasetname, H5P_DEFAULT);
  status = H5Dread (dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_float); 
  for (k = 0; k < CONV1_NBOUTPUT; k++) 
    bias[k] = buffer_float[k]; 

  status = H5Dclose (dataset);
  status = H5Fclose (file);

}


void ReadConv2Weights(char *filename, char *datasetname, float weight[CONV2_NBOUTPUT][CONV1_NBOUTPUT][CONV2_DIM][CONV2_DIM]) {
  unsigned short 	x, y, z, k; 
  float 	 		buffer_float[CONV2_DIM][CONV2_DIM][CONV1_NBOUTPUT][CONV2_NBOUTPUT]; // y, x, z, k
  hid_t 	 		file, dataspace, dataset; 
  herr_t 	 		status; 

  file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen (file, datasetname, H5P_DEFAULT);
  status = H5Dread (dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_float); 
  for (k = 0; k < CONV2_NBOUTPUT; k++)
    for (y = 0; y < CONV2_DIM; y++)
      for (x = 0; x < CONV2_DIM; x++)
		for (z = 0; z < CONV1_NBOUTPUT; z++)
		  weight[k][z][y][x] = buffer_float[y][x][z][k]; // re-ordering [y][x][z][k] -> [k][z][y][x]

  status = H5Dclose (dataset);
  status = H5Fclose (file);

}


void ReadConv2Bias(char *filename, char *datasetname, float *bias) {
  unsigned short 	k; 
  hid_t 	 		file, dataspace, dataset; 
  herr_t 	 		status; 
  float 	 		buffer_float[CONV2_NBOUTPUT]; 

  file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen (file, datasetname, H5P_DEFAULT);
  status = H5Dread (dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_float); 
  for (k = 0; k < CONV2_NBOUTPUT; k++) 
    bias[k] = buffer_float[k]; 

  status = H5Dclose (dataset);
  status = H5Fclose (file);

}


// Flatten layer impacts reading order: 
// Keras / Tensorflow uses NHWC channels last
// so the 800 (50*4*4) flatten values are in order NHWC channels last
void ReadFc1Weights(char *filename, char *datasetname, float weight[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]) {
  unsigned short 	x, y, z, k; 
  //float 			buffer[POOL2_NBOUTPUT*POOL2_HEIGHT*POOL2_WIDTH][FC1_NBOUTPUT]; // zyx, k
  float 			buffer_float[POOL2_HEIGHT*POOL2_WIDTH*POOL2_NBOUTPUT][FC1_NBOUTPUT]; // yxz, k
  hid_t 			file, dataspace, dataset; 
  herr_t 			status; 

  file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen (file, datasetname, H5P_DEFAULT);
  status = H5Dread (dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_float); 
  for (k = 0; k < FC1_NBOUTPUT; k++)
    for (z = 0; z < POOL2_NBOUTPUT; z++)
      for (y = 0; y < POOL2_HEIGHT; y++)
        for (x = 0; x < POOL2_WIDTH; x++)
		  //weight[k][z][y][x] = buffer[(z*POOL2_WIDTH*POOL2_HEIGHT)+(y*POOL2_WIDTH)+x][k]; // re-ordering [zyx][k] -> [k][z][y][x]
		  weight[k][z][y][x] = buffer_float[(y*POOL2_WIDTH*POOL2_NBOUTPUT)+(x*POOL2_NBOUTPUT)+z][k]; // re-ordering [yxz][k] -> [k][z][y][x]

  status = H5Dclose (dataset);
  status = H5Fclose (file);

}


void ReadFc1Bias(char *filename, char *datasetname, float *bias) {
  unsigned short 	k; 
  hid_t 	 		file, dataspace, dataset; 
  herr_t 	 		status; 
  float 	 		buffer_float[FC1_NBOUTPUT]; 

  file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen (file, datasetname, H5P_DEFAULT);
  status = H5Dread (dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_float); 
  for (k = 0; k < FC1_NBOUTPUT; k++) 
    bias[k] = buffer_float[k]; 

  status = H5Dclose (dataset);
  status = H5Fclose (file);

}


void ReadFc2Weights(char *filename, char *datasetname, float weight[FC2_NBOUTPUT][FC1_NBOUTPUT]) {
  unsigned short 	z, k; 
  float 			buffer_float[FC1_NBOUTPUT][FC2_NBOUTPUT]; // z, k
  hid_t 			file, dataspace, dataset; 
  herr_t 			status; 

  file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen (file, datasetname, H5P_DEFAULT);
  status = H5Dread (dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_float); 
  for (k = 0; k < FC2_NBOUTPUT; k++)
    for (z = 0; z < FC1_NBOUTPUT; z++)
	  weight[k][z] = buffer_float[z][k]; // re-ordering [z][k] -> [k][z]

  status = H5Dclose (dataset);
  status = H5Fclose (file);

}


void ReadFc2Bias(char *filename, char *datasetname, float *bias) {
  unsigned short 	k; 
  hid_t 	 		file, dataspace, dataset; 
  herr_t 	 		status; 
  float 	 		buffer_float[FC2_NBOUTPUT]; 

  file = H5Fopen (filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  dataset = H5Dopen (file, datasetname, H5P_DEFAULT);
  status = H5Dread (dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer_float); 
  for (k = 0; k < FC2_NBOUTPUT; k++) 
    bias[k] = buffer_float[k]; 

  status = H5Dclose (dataset);
  status = H5Fclose (file);

}



