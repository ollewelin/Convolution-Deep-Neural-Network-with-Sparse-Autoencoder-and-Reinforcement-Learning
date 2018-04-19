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
    cnn_autoenc_layer1.layer_nr = 1;
    sparse_autoenc cnn_autoenc_layer2;
    cnn_autoenc_layer2.layer_nr = 2;


    cnn_autoenc_layer1.show_patch_during_run = 1;///Only for debugging
    cnn_autoenc_layer1.use_greedy_enc_method = 1;///
    cnn_autoenc_layer1.show_encoder_on_conv_cube = 1;
    cnn_autoenc_layer1.show_encoder = 0;
    cnn_autoenc_layer1.init_in_from_outside = 0;///When init_in_from_outside = 1 then Lx_IN_data_cube is same poiner as the Lx_OUT_convolution_cube of the previous layer
    cnn_autoenc_layer1.color_mode          = 1;///color_mode = 1 is ONLY allowed to use at Layer 1
    cnn_autoenc_layer1.patch_side_size     = 7;
    cnn_autoenc_layer1.Lx_IN_depth         = 1;///This is forced inside class to 1 when color_mode = 1. In gray mode = color_mode = 0 this number is the size of the input data depth.
                                               ///So if for example the input data come a convolution cube the Lx_IN_depth is the number of the depth of this convolution cube source/input data
                                               ///In a chain of Layer's the Lx_IN_depth will the same size as the Lx_OUT_depth of the previous layer order.
    cnn_autoenc_layer1.Lx_OUT_depth        = 20;///This is the number of atom's in the whole dictionary.
    cnn_autoenc_layer1.stride              = 1;
    cnn_autoenc_layer1.Lx_IN_hight         = CIFAR_object.CIFAR_height;///Convolution cube hight of data
    cnn_autoenc_layer1.Lx_IN_widht         = CIFAR_object.CIFAR_width;///Convolution cube width of data
    cnn_autoenc_layer1.e_stop_threshold    = 30.0f;
    cnn_autoenc_layer1.K_sparse            = cnn_autoenc_layer1.Lx_OUT_depth / 4;
    //cnn_autoenc_layer1.K_sparse            = 2;
    cnn_autoenc_layer1.use_dynamic_penalty = 0;
    cnn_autoenc_layer1.penalty_add         = 0.0f;
    cnn_autoenc_layer1.init_noise_gain     = 0.55f;///
    cnn_autoenc_layer1.enable_denoising    = 0;
    cnn_autoenc_layer1.denoising_percent   = 50;///0..100
    cnn_autoenc_layer1.use_bias = 1;
    cnn_autoenc_layer1.use_leak_relu = 1;
    cnn_autoenc_layer1.score_bottom_level = -1000.0f;
    cnn_autoenc_layer1.use_variable_leak_relu = 1;
    cnn_autoenc_layer1.min_relu_leak_gain = 0.01f;
    cnn_autoenc_layer1.relu_leak_gain_variation = 0.05f;
    cnn_autoenc_layer1.fix_relu_leak_gain = 0.02f;

    cnn_autoenc_layer1.init();
    cnn_autoenc_layer1.k_sparse_sanity_check();
    cnn_autoenc_layer1.copy_dictionary2visual_dict();

    CIFAR_object.insert_L1_IN_data_cube = cnn_autoenc_layer1.Lx_IN_data_cube;///Copy over Mat pointer so CIFAR input image could be load into cnn_autoenc_layer1.Lx_IN_data_cube memory.
    CIFAR_object.insert_a_random_CIFAR_image();
   /// cv::imshow("Dictionary L1", cnn_autoenc_layer1.dictionary);
   /// cv::imshow("Visual dict L1", cnn_autoenc_layer1.visual_dict);
   /// cv::imshow("L1 IN cube", cnn_autoenc_layer1.Lx_IN_data_cube);
   /// cv::imshow("L1 OUT cube", cnn_autoenc_layer1.Lx_OUT_convolution_cube);

    cnn_autoenc_layer2.show_patch_during_run = 0;///Only for debugging
    cnn_autoenc_layer2.use_greedy_enc_method = 1;///
    cnn_autoenc_layer2.show_encoder = 1;
    cnn_autoenc_layer2.show_encoder_on_conv_cube = 1;
    cnn_autoenc_layer2.Lx_IN_data_cube = cnn_autoenc_layer1.Lx_OUT_convolution_cube;///Pointer are copy NOT copy the physical memory. Copy physical memory is not good solution here.
    cnn_autoenc_layer2.init_in_from_outside       = 1;///When init_in_from_outside = 1 then Lx_IN_data_cube is same poiner as the Lx_OUT_convolution_cube of the previous layer
    cnn_autoenc_layer2.color_mode      = 0;///color_mode = 1 is ONLY allowed to use at Layer 1
    cnn_autoenc_layer2.patch_side_size  = 7;
    cnn_autoenc_layer2.Lx_IN_depth      = cnn_autoenc_layer1.Lx_OUT_depth;///This is forced inside class to 1 when color_mode = 1. In gray mode = color_mode = 0 this number is the size of the input data depth.
                                            ///So if for example the input data come a convolution cube the Lx_IN_depth is the number of the depth of this convolution cube source/input data
                                            ///In a chain of Layer's the Lx_IN_depth will the same size as the Lx_OUT_depth of the previous layer order.
    cnn_autoenc_layer2.Lx_OUT_depth     = 20;///This is the number of atom's in the whole dictionary.
    cnn_autoenc_layer2.stride           = 2;///
    cnn_autoenc_layer2.Lx_IN_hight      = cnn_autoenc_layer1.Lx_OUT_hight;///Convolution cube hight of data
    cnn_autoenc_layer2.Lx_IN_widht      = cnn_autoenc_layer1.Lx_OUT_widht;///Convolution cube width of data
    cnn_autoenc_layer2.e_stop_threshold = 30.0f;
    cnn_autoenc_layer2.K_sparse     = cnn_autoenc_layer2.Lx_OUT_depth / 4;
    //cnn_autoenc_layer2.K_sparse     = 2;
    cnn_autoenc_layer2.use_dynamic_penalty = 0;
    cnn_autoenc_layer2.penalty_add      = 0.0f;
    cnn_autoenc_layer2.init_noise_gain = 0.15f;///
    cnn_autoenc_layer2.enable_denoising = 0;
    cnn_autoenc_layer2.denoising_percent = 50;///0..100
    cnn_autoenc_layer2.use_bias = 1;
    cnn_autoenc_layer2.use_leak_relu = 1;
    cnn_autoenc_layer2.score_bottom_level = -1000.0f;
    cnn_autoenc_layer2.use_variable_leak_relu = 0;
    cnn_autoenc_layer2.fix_relu_leak_gain = 0.01;

    cnn_autoenc_layer2.init();
    cnn_autoenc_layer2.k_sparse_sanity_check();
    cnn_autoenc_layer2.copy_dictionary2visual_dict();
  ///  cv::imshow("Dictionary L2", cnn_autoenc_layer2.dictionary);
  ///  cv::imshow("Visual dict L2", cnn_autoenc_layer2.visual_dict);
  ///  cv::imshow("L2 IN cube", cnn_autoenc_layer2.Lx_IN_data_cube);///If no pooling is used between L1-L2 This should be EXACT same image as previous OUT cube layer "Lx OUT cube"
  ///  cv::imshow("L2 OUT cube", cnn_autoenc_layer2.Lx_OUT_convolution_cube);

    cv::waitKey(1);
    while(1)
    {
      //  CIFAR_object.insert_a_random_CIFAR_image();
        cnn_autoenc_layer1.random_change_ReLU_leak_variable();
        cnn_autoenc_layer1.train_encoder();
        cnn_autoenc_layer2.train_encoder();
        cv::imshow("Visual_dict_L2", cnn_autoenc_layer2.visual_dict);
        cv::imshow("L2_IN_cube", cnn_autoenc_layer2.Lx_IN_data_cube);///If no pooling is used between L1-L2 This should be EXACT same image as previous OUT cube layer "Lx OUT cube"
        cv::imshow("L2_OUT_cube", cnn_autoenc_layer2.Lx_OUT_convolution_cube);
        cv::imshow("Visual_dict_L1", cnn_autoenc_layer1.visual_dict);
        cv::imshow("L1_IN_cube", cnn_autoenc_layer1.Lx_IN_data_cube);
        cv::imshow("L1_OUT_cube", cnn_autoenc_layer1.Lx_OUT_convolution_cube);
        imshow("L1 rec", cnn_autoenc_layer1.reconstruct);
        imshow("L2 rec", cnn_autoenc_layer2.reconstruct);
        cv::waitKey(100);
    }

    return 0;
}
