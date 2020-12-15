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

void ReadPgmFile(char *filename, unsigned char *pix)
{
  FILE *pgm_file;
  int i, width, height, max, ret;
  char readChars[256];

  pgm_file = fopen(filename, "rb");
  if (!pgm_file)
  {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }

  ret = fscanf(pgm_file, "%s", readChars);
  ret = fscanf(pgm_file, "%d", &width);
  ret = fscanf(pgm_file, "%d", &height);
  ret = fscanf(pgm_file, "%d", &max);
  //  printf("Reading PGM file %s \t -> Type %s, width %d, height %d, max %d\n", filename, readChars, width, height, max);
  //  if (width != IMG_WIDTH) printf("Warning: Image width mismatch (%d, expecting %d) \t -> Consider rescaling\n", width, IMG_WIDTH);
  //  if (height != IMG_HEIGHT) printf("Warning: Image height mismatch(%d, expecting %d) \t -> Consider rescaling\n", height, IMG_HEIGHT);

  for (i = 0; i < width * height; i++) // DEBUG IF IMG_DEPTH > 1 ??
    ret = fscanf(pgm_file, "%c", &pix[i]);

  fclose(pgm_file);
}

void WritePgmFile(char *filename, float *pix, short width, short height)
{
  FILE *pgm_file;
  short i;

  pgm_file = fopen(filename, "w");
  if (!pgm_file)
  {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }
  fprintf(pgm_file, "P2\n");
  fprintf(pgm_file, "%d %d\n", width, height);
  fprintf(pgm_file, "255\n");

  for (i = 0; i < width * height; i++)
  {                                                         // DEBUG IF IMG_DEPTH > 1 ??
    fprintf(pgm_file, "%d ", (unsigned char)(pix[i] * 64)); // *64 because pix values are too small
    if (i % width == width - 1)
      fprintf(pgm_file, "\n");
  }

  fclose(pgm_file);
}

void ReadTestLabels(char *filename, short size)
{
  FILE *label_file;
  int ret;
  short k;
  unsigned char label;

  label_file = fopen(filename, "r");
  if (!label_file)
  {
    printf("Error: Unable to open file %s.\n", filename);
    exit(1);
  }

  for (k = 0; k < size; k++)
  {
    ret = fscanf(label_file, "%c", &label);
    if (k >= 8)
      printf("img%d -> 0x%x \n", k - 8, label);
  }
  printf("\n");

  fclose(label_file);
}

// Nearest neighbor, linear interpolation
// Based on
// http://courses.cs.vt.edu/~masc1044/L17-Rotation/ScalingNN.html
#define min(a, b) ((a) < (b) ? (a) : (b))
void RescaleImg(unsigned char *input, short width, short height, float *output, short new_width, short new_height)
{
  short x, y;
  short interpol_x, interpol_y;

  for (y = 0; y < new_height; y++)
  {
    for (x = 0; x < new_width; x++)
    {
      interpol_x = (short)(((float)x / (float)new_width) * (float)width + 0.5);
      interpol_x = min(interpol_x, width - 1);
      interpol_y = (short)(((float)y / (float)new_height) * (float)height + 0.5); // MOVE TO Y LOOP
      interpol_y = min(interpol_y, height - 1);
      output[(y * new_width) + x] = input[(interpol_y * width) + interpol_x];
    }
  }
}

void NormalizeImg(unsigned char *input, unsigned char *output, short width, short height)
{
  short x, y;

  for (y = 0; y < height; y++)
    for (x = 0; x < width; x++)
      //for some strange reason, it is faster if I leave this function here... no need for this at all, could use REF_IMG at the beginning
      output[(y * width) + x] = input[(y * width) + x];
}