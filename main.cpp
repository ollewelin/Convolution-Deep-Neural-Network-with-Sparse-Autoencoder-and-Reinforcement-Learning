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

int main()
{
    cv::Mat test;
    sparse_autoenc cnn_autoenc_layer1;
    cnn_autoenc_layer1.colour_mode      = 1;///colour_mode = 1 is ONLY allowed to use at Layer 1
    cnn_autoenc_layer1.patch_side_size  = 15;
    cnn_autoenc_layer1.Lx_IN_depth      = 1;///This is forced inside class to 1 when colour_mode = 1. In gray mode = colour_mode = 0 this number is the size of the input data depth.
                                            ///So if for example the input data come a convolution cube the Lx_IN_depth is the number of the depth of this convolution cube source/input data
                                            ///In a chain of Layer's the Lx_IN_depth will the same size as the Lx_OUT_depth of the previous layer order.
    cnn_autoenc_layer1.Lx_OUT_depth     = 10;///This is the number of atom's in the whole dictionary.
    cnn_autoenc_layer1.stride           = 2;
    cnn_autoenc_layer1.Lx_IN_hight      = 28;///Convolution cube hight of data
    cnn_autoenc_layer1.Lx_IN_widht      = 28;///Convolution cube width of data
    cnn_autoenc_layer1.e_stop_threshold = 30.0f;
    cnn_autoenc_layer1.max_atom_use     = cnn_autoenc_layer1.Lx_OUT_depth / 4;
    cnn_autoenc_layer1.use_dynamic_penalty = 1;
    cnn_autoenc_layer1.penalty_add      = 1.5f;
    cnn_autoenc_layer1.init_noise_gain = 0.15f;///
    printf("Start init CNN layer 1\n");
    cnn_autoenc_layer1.init();
    cv::imshow("dict", cnn_autoenc_layer1.dictionary);
    cv::waitKey();
    while(1){}

    return 0;
}
