#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
//#include <cstdlib>
//#include <ctime>
#include <math.h>  // exp
#include <stdlib.h>// exit(0);
#include <iostream>
using namespace std;
//using namespace cv;
#include "sparse_autoenc.hpp"
#include "CIFAR_test_data.h"

int main()
{
    CIFAR_test_data CIFAR_object;///Data input images use CIFAR data set
    printf("Need CIFAR input data data_batch_1.bin for test\n");
    CIFAR_object.init_CIFAR();///Read CIFAR data_batch_1.bin file
    cv::Mat test;
    sparse_autoenc cnn_autoenc_layer1;
    cnn_autoenc_layer1.colour_mode      = 1;///colour_mode = 1 is ONLY allowed to use at Layer 1
    cnn_autoenc_layer1.patch_side_size  = 15;
    cnn_autoenc_layer1.Lx_IN_depth      = 1;///This is forced inside class to 1 when colour_mode = 1. In gray mode = colour_mode = 0 this number is the size of the input data depth.
                                            ///So if for example the input data come a convolution cube the Lx_IN_depth is the number of the depth of this convolution cube source/input data
                                            ///In a chain of Layer's the Lx_IN_depth will the same size as the Lx_OUT_depth of the previous layer order.
    cnn_autoenc_layer1.Lx_OUT_depth     = 50;///This is the number of atom's in the whole dictionary.
    cnn_autoenc_layer1.stride           = 2;
    cnn_autoenc_layer1.Lx_IN_hight      = CIFAR_object.CIFAR_height;///Convolution cube hight of data
    cnn_autoenc_layer1.Lx_IN_widht      = CIFAR_object.CIFAR_width;///Convolution cube width of data
    cnn_autoenc_layer1.e_stop_threshold = 30.0f;
    cnn_autoenc_layer1.max_atom_use     = cnn_autoenc_layer1.Lx_OUT_depth / 4;
    cnn_autoenc_layer1.use_dynamic_penalty = 1;
    cnn_autoenc_layer1.penalty_add      = 1.5f;
    cnn_autoenc_layer1.init_noise_gain = 0.15f;///

    printf("**********************\n");
    printf("Start init CNN layer 1\n");
    cnn_autoenc_layer1.init();
    printf("**********************\n");

    cv::imshow("Dictionary", cnn_autoenc_layer1.dictionary);
    cv::imshow("L1 IN cube", cnn_autoenc_layer1.Lx_IN_convolution_cube);
    cv::imshow("L1 OUT cube", cnn_autoenc_layer1.Lx_OUT_convolution_cube);

    cv::waitKey();
    while(1){}

    return 0;
}
