/**
  ******************************************************************************
  * @file    conv.c
  * @author  Chaitanya Devidas Gore, Bogdan Mihai Nistor, Nelli Nyisztor, Université Côte d'Azur, France
  * @version V1.0
  * @date    07 september 2020
  * @brief   Convolutional layers for modified LeNet5
  */

#include <stdio.h>
#include <stdlib.h>

#include "lenet_cnn_float.h"

float sumProduct( float imgPart[CONV1_DIM][CONV1_DIM], float filter[CONV1_DIM][CONV1_DIM]){
  short y,x;
  float conv_result=0;

  for(y = 0; y < CONV1_DIM; y++){
      for(x = 0; x < CONV1_DIM; x++){
        conv_result+=imgPart[y][x]*filter[y][x];
      }
  }
  return conv_result;
}

void Conv1_28x28x1_5x5x20_1_0(  float input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH], 	                // IN [1][28][28]
				                float kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM], 	// IN [20][1][5][5]
				                float bias[CONV1_NBOUTPUT],						                // IN [20]
				                float output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH])        // OUT [20][24][24]
{
  short o,h,w,x,y;
  float imgPart[CONV1_DIM][CONV1_DIM];
  float conv_px;
  
  for(o = 0; o < CONV1_NBOUTPUT; o++){
    for(h = 0; h < CONV1_HEIGHT; h++){
      for(w = 0; w < CONV1_WIDTH; w++){
        //prepare 5X5 imgPart for the convolution
        for(y = 0; y < CONV1_DIM; y++){
          for(x = 0; x < CONV1_DIM; x++){
            imgPart[y][x]=input[0][h+y][w+x];
          }
        }
        conv_px=sumProduct(imgPart,kernel[o][0]);

        //neuron activation >> if removed: current accuracy is better, but we dont want overtrained stuff...
        if(conv_px+bias[o]<=0){
          output[o][h][w]=0;
        }else{
          output[o][h][w]=conv_px + bias[o];
        }

        output[o][h][w]=conv_px + bias[o];
        
      }
    }
  }
  
}

void Conv2_12x12x20_5x5x40_1_0( float input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH], 	            // IN
				                float kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM], 	// IN
				                float bias[CONV2_NBOUTPUT], 						                    // IN
				                float output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]) 		        // OUT
{
  short f,d,o,h,w,x,y,oh,ow;
  float imgPart[CONV2_DIM][CONV2_DIM];
  float conv_px;
  
  for(f=0;f<CONV2_NBOUTPUT;f++){
    for(d=0;d<POOL1_NBOUTPUT;d++){

      for(h=0;h<CONV2_HEIGHT;h++){
        for(w=0;w<CONV2_WIDTH;w++){
          
          for(y = 0; y < CONV2_DIM; y++){ //prepare 5X5 imgPart for the convolution
            for(x = 0; x < CONV2_DIM; x++){
              imgPart[y][x]=input[d][y+h][x+w];
            }
          }

          conv_px=sumProduct(imgPart,kernel[f][d]);

          if(d==0){ //to initialize first element
            output[f][h][w] = conv_px;
          }else{
            output[f][h][w]+= conv_px;
          }
        }
      } 
    }

    for(oh=0;oh<CONV2_HEIGHT;oh++){
      for(ow=0;ow<CONV2_WIDTH;ow++){
        //neuron activation
        if(output[f][oh][ow]+bias[f]<=0){
          output[f][oh][ow]=0;
        }else{
          output[f][oh][ow]=output[f][oh][ow] + bias[f];
        }
      }
    }
  }
}