//#include <opencv2/opencv_modules.hpp>
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <stdio.h>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

//#include <cstdlib>
//#include <ctime>
#include <math.h>  // exp
#include <stdlib.h>// exit(0);
#include <iostream>
using namespace std;
using namespace cv;
#include "sparse_autoenc.hpp"
#include "CIFAR_test_data.h"
using namespace cv::cuda;

///************************************************************************
///*************** (GUI) Graphic User Interface **************************
///************************************************************************
int GUI_parameter1_int = 1;///layer_nr
int GUI_parameter2_int = 4;///learning_gain
int GUI_parameter3_int = 90;
int GUI_parameter4_int = 0;///Nois
int GUI_parameter5_int = 25;
int GUI_parameter6_int = 100;
int GUI_parameter7_int = 3000;
int GUI_parameter8_int = 10;
int save_push = 0;
int load_push = 0;
int print_score = 0;
int greedy_mode = 1;
int print_pause_ms = 1;
int autoenc_ON =1;/// 1 = Autoencoder. 0 = Convolution All layer.
int H_MIN = 0;
int H_MAX = 1000;
int layer_MIN = 1;
int layer_MAX = 10;

const string GUI_WindowName = "GUI_Trackbars";

void callbackButton1(int state, void *)
{
    printf("Autoencoder ON/OFF state =%d\n", state);
    autoenc_ON = state;

}

void callbackButton2(int state, void *pointer)
{
    printf("Save button pressed\n");
    save_push = 1;
}
void callbackButton3(int state, void *pointer)
{
    printf("Load button pressed\n");

    load_push = 1;
}
void callbackButton4(int state, void *)
{
    printf("Print score ON/OFF state =%d\n", state);
    print_score = state;

}
void callbackButton5(int state, void *pointer)
{
    printf("Greedy mode ON/OFF state =%d\n", state);
    greedy_mode = state;
}
void callbackButton6(int state, void *pointer)
{
    printf("button6 pressed\n");
}
void callbackButton7(int state, void *pointer)
{
    printf("button7 pressed\n");
}


float GUI_learning_rate = 0.0f;

void action_GUI( int, void* )
{
    GUI_learning_rate = 0.001f * ((float) GUI_parameter2_int);
    printf("GUI_learning_rate = %f\n", GUI_learning_rate);

    if(GUI_parameter1_int <layer_MIN)
    {
        GUI_parameter1_int = layer_MIN;
    }
    printf("layer = %d\n", GUI_parameter1_int);
    if(GUI_parameter7_int < 1)
    {
        print_pause_ms = 1;

    }
    else
    {
        print_pause_ms = GUI_parameter7_int;
    }
}


void create_GUI(void)
{
    cv::namedWindow(GUI_WindowName,0);
    char GUI_TrackbarName[50];
    sprintf(GUI_TrackbarName, "Control values");
    string nameb1 = "Autoencode";
    string nameb2 = "Save dictionary file";
    string nameb3 = "Load dictionary file";
    string nameb4 = "print";
    string nameb5 = "Greedy";
    string nameb6 = "Buttom6";
    string nameb7 = "Buttom7";

    cv::createButton(nameb1,callbackButton1,NULL,QT_CHECKBOX,1);
    cv::createButton(nameb2,callbackButton2,NULL,QT_PUSH_BUTTON,0);
    cv::createButton(nameb3,callbackButton3,NULL,QT_PUSH_BUTTON,2);
    cv::createButton(nameb4,callbackButton4,NULL,QT_CHECKBOX,0);///Print
    cv::createButton(nameb5,callbackButton5,NULL,QT_CHECKBOX,1);///Greedy
    cv::createButton(nameb6,callbackButton6,NULL,QT_PUSH_BUTTON,0);
    cv::createButton(nameb7,callbackButton7,NULL,QT_PUSH_BUTTON,0);

    cv::createTrackbar("Layer numb", GUI_WindowName, &GUI_parameter1_int, layer_MAX, action_GUI);
    cv::createTrackbar("learning g ", GUI_WindowName, &GUI_parameter2_int, 1000, action_GUI);
    cv::createTrackbar("residual g ", GUI_WindowName, &GUI_parameter3_int, 100, action_GUI);
    cv::createTrackbar("noise perc ", GUI_WindowName, &GUI_parameter4_int, 100, action_GUI);
    cv::createTrackbar("K_sparse ", GUI_WindowName, &GUI_parameter5_int, 1000, action_GUI);
    cv::createTrackbar("Bias_level ", GUI_WindowName, &GUI_parameter6_int, 100, action_GUI);
    cv::createTrackbar("ms pause ", GUI_WindowName, &GUI_parameter7_int, 10000, action_GUI);
    cv::createTrackbar("Max node ", GUI_WindowName, &GUI_parameter8_int, 500, action_GUI);


}
///************************************************************************
///************ End of (GUI) **********************************************
///************************************************************************



int main()
{
	printf("GPU ON = %d\n", cv::cuda::getCudaEnabledDeviceCount());
	printf("Have not yet start use GpuMat\n");
	printf("To be continue with GPU cv::cuda\n");
	printf("OpenCV version:\n");
	std::cout << CV_VERSION << std::endl;

    int use_CIFAR = 1;
    printf("Would you like to use CIFAR dataset as input image<Y>/<N> \n");
    char answer_character;
    answer_character = getchar();


    cv::Mat input_jpg_BGR;
    cv::Mat input_jpg_FC3;

    if(answer_character == 'Y' || answer_character == 'y')
    {
        use_CIFAR = 1;
    }
    else
    {
        use_CIFAR = 0;
        printf("now use file:\n");
        printf("input.JPG\n");
        input_jpg_BGR = cv::imread("input.JPG", 1);
        input_jpg_BGR.convertTo(input_jpg_FC3, CV_32FC3, 1.0f/255.0f);
        printf("Start a GPU test\n");
        int GPU_TEST = 1;
        printf("GPU_TEST = %d\n", GPU_TEST);
        if(GPU_TEST == 1 && input_jpg_FC3.cols > 49 && input_jpg_FC3.rows > 49)
        {


///Test GPU things
            cv::cuda::GpuMat test_gpu_mat;///Not yet used
            cv::cuda::GpuMat gpu_roi_part;
            cv::cuda::GpuMat gpu_roi_part_B;
            cv::cuda::GpuMat gpu_roi_part_C;
            test_gpu_mat.create(input_jpg_FC3.rows,input_jpg_FC3.cols,CV_32FC3);
            gpu_roi_part.create(8,8,CV_32FC3);
            gpu_roi_part_B.create(8,8,CV_32FC3);
            gpu_roi_part_C.create(8,8,CV_32FC3);
            cv::Mat part_of_inputJPG;
            part_of_inputJPG.create(8,8,CV_32FC3);

            ///src(Rect(left,top,width, height)).copyTo(dst);
            input_jpg_FC3(Rect(18,10,8,8)).copyTo(part_of_inputJPG);///No effect will be overwritten by gpu_roi_part.download(part_of_inputJPG);
//for(int h=0;h<)
            for(int i=0; i<10; i++)
            {

                test_gpu_mat.upload(input_jpg_FC3);///Data to GPU "device"
                test_gpu_mat(Rect(5+i,10,8,8)).copyTo(gpu_roi_part);///Inside NVIDIA Rect() a part of image in GPU to another GpuMat
                test_gpu_mat(Rect(7+i,26+i,8,8)).copyTo(gpu_roi_part_B);///Inside NVIDIA Rect() a part of image in GPU to another GpuMat
                ///input.JPG test was 50x50
                test_gpu_mat(Rect(10+i,40,8,8)).copyTo(gpu_roi_part_C);///Inside NVIDIA Rect() a part of image in GPU to another GpuMat

///==========================================================
///OpenCV documentation regaring   gpu::multiply
///C++: void gpu::multiply(const GpuMat& a, const GpuMat& b, GpuMat& c, double scale=1, int dtype=-1, Stream& stream=Stream::Null() )
///C++: void gpu::multiply(const GpuMat& a, const Scalar& sc, GpuMat& c, double scale=1, int dtype=-1, Stream& stream=Stream::Null() )¶
///Parameters:
///    a – First source matrix.
///    b – Second source matrix to be multiplied by a elements.
///    sc – A scalar to be multiplied by a elements.
///    c – Destination matrix that has the same size and number of channels as the input array(s). The depth is defined by dtype or a depth.
///    scale – Optional scale factor.
///    dtype – Optional depth of the output array.
///    stream – Stream for the asynchronous version.
///==========================================================
                float scaler = (float) i/10;
                cv::cuda::multiply(gpu_roi_part,gpu_roi_part_B,gpu_roi_part_C, scaler);

                gpu_roi_part_C.download(part_of_inputJPG);///Data back to CPU "host"
                test_gpu_mat.download(input_jpg_FC3);///Data back to CPU "host"

                imshow("input_jpg_FC3", input_jpg_FC3);
                imshow("part_of_inputJPG", part_of_inputJPG);
                waitKey(1000);
            }
///End GPU test
        }
        printf("End GPU test\n");
    }


    if(GUI_parameter7_int < 1)
    {
        print_pause_ms = 1;

    }
    else
    {
        print_pause_ms = GUI_parameter7_int;
    }

    create_GUI();
    CIFAR_test_data CIFAR_object;///Data input images use CIFAR data set
    if(use_CIFAR == 1)
    {
        printf("Need CIFAR input data data_batch_1.bin for test\n");
        CIFAR_object.init_CIFAR();///Read CIFAR data_batch_1.bin file
        CIFAR_object.print_CIFAR_nr = 0;
    }

    cv::Mat test;
    sparse_autoenc cnn_autoenc_layer1;
    cnn_autoenc_layer1.layer_nr = 1;
    sparse_autoenc cnn_autoenc_layer2;
    cnn_autoenc_layer2.layer_nr = 2;

    int layer_control;

    cnn_autoenc_layer1.show_patch_during_run = 0;///Only for debugging
    cnn_autoenc_layer1.use_auto_bias_level = 0;
    cnn_autoenc_layer1.fix_bias_level = 0.01f;

    cnn_autoenc_layer1.use_greedy_enc_method = greedy_mode;///
    cnn_autoenc_layer1.print_greedy_reused_atom = 0;
    cnn_autoenc_layer1.show_encoder_on_conv_cube = 1;
    cnn_autoenc_layer1.use_salt_pepper_noise = 1;///Only depend in COLOR mode. 1 = black..white noise. 0 = all kinds of color noise
    cnn_autoenc_layer1.learning_rate = 0.0;
    cnn_autoenc_layer1.residual_gain = 0.9;
    cnn_autoenc_layer1.init_in_from_outside = 0;///When init_in_from_outside = 1 then Lx_IN_data_cube is same poiner as the Lx_OUT_convolution_cube of the previous layer
    cnn_autoenc_layer1.color_mode          = 1;///color_mode = 1 is ONLY allowed to use at Layer 1
    cnn_autoenc_layer1.patch_side_size     = 13;
    cnn_autoenc_layer1.Lx_IN_depth         = 1;///This is forced inside class to 1 when color_mode = 1. In gray mode = color_mode = 0 this number is the size of the input data depth.
                                               ///So if for example the input data come a convolution cube the Lx_IN_depth is the number of the depth of this convolution cube source/input data
                                               ///In a chain of Layer's the Lx_IN_depth will the same size as the Lx_OUT_depth of the previous layer order.
    cnn_autoenc_layer1.Lx_OUT_depth        = 400;///This is the number of atom's in the whole dictionary.
    cnn_autoenc_layer1.stride              = 1;
    if(use_CIFAR == 1)
    {

        cnn_autoenc_layer1.Lx_IN_hight         = CIFAR_object.CIFAR_height;///Convolution cube hight of data
        cnn_autoenc_layer1.Lx_IN_widht         = CIFAR_object.CIFAR_width;///Convolution cube width of data
    }
    else
    {
        cnn_autoenc_layer1.Lx_IN_hight         = input_jpg_FC3.rows;///Convolution cube hight of data
        cnn_autoenc_layer1.Lx_IN_widht         = input_jpg_FC3.cols;///Convolution cube width of data

    }
    cnn_autoenc_layer1.e_stop_threshold    = 30.0f;
    //cnn_autoenc_layer1.K_sparse            = cnn_autoenc_layer1.Lx_OUT_depth / 15;
    cnn_autoenc_layer1.K_sparse            = GUI_parameter5_int;
    cnn_autoenc_layer1.use_dynamic_penalty = 0;
    cnn_autoenc_layer1.penalty_add         = 0.0f;
    cnn_autoenc_layer1.init_noise_gain     = 0.15f;///
    cnn_autoenc_layer1.enable_denoising    = 1;

    cnn_autoenc_layer1.use_leak_relu = 1;
    cnn_autoenc_layer1.score_bottom_level = -1000.0f;
    cnn_autoenc_layer1.use_variable_leak_relu = 1;
    cnn_autoenc_layer1.min_relu_leak_gain = 0.01f;
    cnn_autoenc_layer1.relu_leak_gain_variation = 0.05f;
    cnn_autoenc_layer1.fix_relu_leak_gain = 0.02f;

    if(use_CIFAR == 1)
    {
        cnn_autoenc_layer1.init();
        CIFAR_object.insert_L1_IN_data_cube = cnn_autoenc_layer1.Lx_IN_data_cube;///Copy over Mat pointer so CIFAR input image could be load into cnn_autoenc_layer1.Lx_IN_data_cube memory.
        CIFAR_object.insert_a_random_CIFAR_image();
    }
    else
    {
        cnn_autoenc_layer1.Lx_IN_data_cube = input_jpg_FC3;
        cnn_autoenc_layer1.init();
    }
    cnn_autoenc_layer1.k_sparse_sanity_check();
    cnn_autoenc_layer1.copy_dictionary2visual_dict();

   /// cv::imshow("Dictionary L1", cnn_autoenc_layer1.dictionary);
   /// cv::imshow("Visual dict L1", cnn_autoenc_layer1.visual_dict);
   /// cv::imshow("L1 IN cube", cnn_autoenc_layer1.Lx_IN_data_cube);
   /// cv::imshow("L1 OUT cube", cnn_autoenc_layer1.Lx_OUT_convolution_cube);

    cnn_autoenc_layer2.show_patch_during_run = 0;///Only for debugging
    cnn_autoenc_layer2.use_greedy_enc_method = greedy_mode;///
    cnn_autoenc_layer2.print_greedy_reused_atom = 1;
    cnn_autoenc_layer2.learning_rate = 0.01;
    cnn_autoenc_layer2.residual_gain = 0.1;
    cnn_autoenc_layer2.show_encoder_on_conv_cube = 1;
    cnn_autoenc_layer2.Lx_IN_data_cube = cnn_autoenc_layer1.Lx_OUT_convolution_cube;///Pointer are copy NOT copy the physical memory. Copy physical memory is not good solution here.
    cnn_autoenc_layer2.init_in_from_outside       = 1;///When init_in_from_outside = 1 then Lx_IN_data_cube is same poiner as the Lx_OUT_convolution_cube of the previous layer
  //  cnn_autoenc_layer2.init_in_from_outside       = 0;
    cnn_autoenc_layer2.color_mode      = 0;///color_mode = 1 is ONLY allowed to use at Layer 1
    cnn_autoenc_layer2.use_salt_pepper_noise = 0;///Only depend in COLOR mode. 1 = black..white noise. 0 = all kinds of color noise

    cnn_autoenc_layer2.patch_side_size  = 7;
    cnn_autoenc_layer2.Lx_IN_depth      = cnn_autoenc_layer1.Lx_OUT_depth;///This is forced inside class to 1 when color_mode = 1. In gray mode = color_mode = 0 this number is the size of the input data depth.
                                            ///So if for example the input data come a convolution cube the Lx_IN_depth is the number of the depth of this convolution cube source/input data
                                            ///In a chain of Layer's the Lx_IN_depth will the same size as the Lx_OUT_depth of the previous layer order.
    cnn_autoenc_layer2.Lx_OUT_depth     = 100;///This is the number of atom's in the whole dictionary.
    cnn_autoenc_layer2.stride           = 2;///
    cnn_autoenc_layer2.Lx_IN_hight      = cnn_autoenc_layer1.Lx_OUT_hight;///Convolution cube hight of data
    cnn_autoenc_layer2.Lx_IN_widht      = cnn_autoenc_layer1.Lx_OUT_widht;///Convolution cube width of data
    cnn_autoenc_layer2.e_stop_threshold = 30.0f;
    cnn_autoenc_layer2.K_sparse     = cnn_autoenc_layer2.Lx_OUT_depth / 4;
    //cnn_autoenc_layer2.K_sparse     = 10;
    cnn_autoenc_layer2.use_dynamic_penalty = 0;
    cnn_autoenc_layer2.penalty_add      = 0.0f;
    cnn_autoenc_layer2.init_noise_gain = 0.25f;///
    cnn_autoenc_layer2.enable_denoising = 1;
    cnn_autoenc_layer2.denoising_percent = 50;///0..100
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

        cnn_autoenc_layer1.random_change_ReLU_leak_variable();
        cnn_autoenc_layer1.train_encoder();
    printf("cnn_autoenc_layer1.use_greedy_enc_method = % d\n", cnn_autoenc_layer1.use_greedy_enc_method);

    //    cnn_autoenc_layer2.train_encoder();
    cv::waitKey(1);
    cv::Mat BGR_L1_dict, BGR_L1_bias_in2hid, BGR_L1_bias_hid2out;
     while(1)
    {
        if(use_CIFAR == 1)
        {
            CIFAR_object.insert_a_random_CIFAR_image();
        }
        else
        {
          //  cnn_autoenc_layer1.Lx_IN_data_cube = input_jpg_FC3;
        }
        cnn_autoenc_layer1.use_greedy_enc_method = greedy_mode;///

        layer_control = GUI_parameter1_int;
        switch(layer_control)
        {
        case(1):
            if(autoenc_ON == 1)
            {
                cnn_autoenc_layer1.show_encoder_on_conv_cube = 0;///
            }
            else
            {
                cnn_autoenc_layer1.show_encoder_on_conv_cube = 1;///
            }
            if(save_push==1)
            {
                cnn_autoenc_layer1.copy_dictionary2visual_dict();
                cnn_autoenc_layer1.visual_dict.convertTo(BGR_L1_dict, CV_8UC3, 255);
                imshow("BGR",  BGR_L1_dict);
                cv::imwrite("L1_dict.bmp", BGR_L1_dict);
                cnn_autoenc_layer1.bias_in2hid.convertTo(BGR_L1_bias_in2hid, CV_8UC1, 255, 128);
                cnn_autoenc_layer1.bias_hid2out.convertTo(BGR_L1_bias_hid2out, CV_8UC1, 255, 128);
                cv::imwrite("L1_bias_in2hid.bmp", BGR_L1_bias_in2hid);
                cv::imwrite("L1_bias_hid2out.bmp", BGR_L1_bias_hid2out);
                save_push=0;
            }
            if(load_push==1)
            {

                BGR_L1_dict = cv::imread("L1_dict.bmp", 1);
                BGR_L1_dict.convertTo(cnn_autoenc_layer1.visual_dict, CV_32FC3, 1.0f/255.0);
                cnn_autoenc_layer1.copy_visual_dict2dictionary();
                BGR_L1_bias_in2hid = cv::imread("L1_bias_in2hid.bmp", 1);
                BGR_L1_bias_hid2out = cv::imread("L1_bias_hid2out.bmp", 1);
                BGR_L1_bias_in2hid.convertTo(cnn_autoenc_layer1.bias_in2hid, CV_32FC1, 1.0f/255.0, -0.5f);
                BGR_L1_bias_hid2out.convertTo(cnn_autoenc_layer1.bias_hid2out, CV_32FC1, 1.0f/255.0, -0.5f);
                load_push=0;
            }
            cnn_autoenc_layer1.denoising_percent   = GUI_parameter4_int;///0..100
            cnn_autoenc_layer1.pause_score_print_ms   = print_pause_ms;///0..100
            cnn_autoenc_layer1.ON_OFF_print_score = print_score;
            if(GUI_parameter8_int == 0)
            {
            cnn_autoenc_layer1.max_ReLU_auto_reset = 1.0f;
            }
            else
            {
            cnn_autoenc_layer1.max_ReLU_auto_reset = (float) GUI_parameter8_int;
            }
            break;
        case(2):
            if(save_push==1)
            {
                cv::imwrite("L2_dict.bin", cnn_autoenc_layer2.dictionary);
                cv::imwrite("L2_bias_in2hid.bin", cnn_autoenc_layer2.bias_in2hid);
                cv::imwrite("L2_bias_hid2out.bin", cnn_autoenc_layer2.bias_hid2out);
                save_push=0;
            }
            if(load_push==1)
            {
                cnn_autoenc_layer2.dictionary = cv::imread("L2_dict.bin", 1);
                cnn_autoenc_layer2.bias_in2hid = cv::imread("L2_bias_in2hid.bin", 1);
                cnn_autoenc_layer2.bias_hid2out = cv::imread("L12_bias_hid2out.bin", 1);
                load_push=0;
            }

            cnn_autoenc_layer2.denoising_percent   = GUI_parameter4_int;///0..100
            break;
        }
        if(autoenc_ON == 0)
        {
            cv::imshow("L2_IN_cube", cnn_autoenc_layer2.Lx_IN_data_cube);///If no pooling is used between L1-L2 This should be EXACT same image as previous OUT cube layer "Lx OUT cube"
            cv::imshow("L2_OUT_cube", cnn_autoenc_layer2.Lx_OUT_convolution_cube);
            cv::imshow("L1_IN_cube", cnn_autoenc_layer1.Lx_IN_data_cube);
            cv::imshow("L1_OUT_cube", cnn_autoenc_layer1.Lx_OUT_convolution_cube);
        }
        switch(layer_control)
        {
        case(1):
            if(cnn_autoenc_layer1.K_sparse != GUI_parameter5_int)
            {
                if(GUI_parameter5_int < 1)
                {
                    GUI_parameter5_int = 1;
                }
                if(GUI_parameter5_int > cnn_autoenc_layer1.Lx_OUT_depth-1)
                {
                    GUI_parameter5_int = cnn_autoenc_layer1.Lx_OUT_depth-1;
                }

                cnn_autoenc_layer1.K_sparse = GUI_parameter5_int;
                printf("K_sparse change to = %d\n", cnn_autoenc_layer1.K_sparse);
                cnn_autoenc_layer1.k_sparse_sanity_check();///This should be called every time K_sparse changes
            }
            cnn_autoenc_layer1.learning_rate = ((float) GUI_parameter2_int) * 0.001f;
            cnn_autoenc_layer1.residual_gain = ((float) GUI_parameter3_int) * 0.01f;
            cnn_autoenc_layer1.fix_bias_level = ((float) GUI_parameter6_int) * 0.01f;
            cnn_autoenc_layer1.random_change_ReLU_leak_variable();
            cnn_autoenc_layer1.copy_dictionary2visual_dict();
            cnn_autoenc_layer1.train_encoder();
            imshow("L1 rec", cnn_autoenc_layer1.reconstruct);
            imshow("L1 enc_input", cnn_autoenc_layer1.encoder_input);
            imshow("L1 enc_error", cnn_autoenc_layer1.enc_error);
            imshow("L1 noise resid", cnn_autoenc_layer1.denoised_residual_enc_input);
            imshow("L1 bias hid2out", cnn_autoenc_layer1.visual_b_hid2out);
            cv::imshow("Visual_dict_L1", cnn_autoenc_layer1.visual_dict);
            break;
        case(2):
            if(cnn_autoenc_layer2.K_sparse != GUI_parameter5_int)
            {
                if(GUI_parameter5_int < 1)
                {
                    GUI_parameter5_int = 1;
                }
                if(GUI_parameter5_int > cnn_autoenc_layer2.Lx_OUT_depth-1)
                {
                    GUI_parameter5_int = cnn_autoenc_layer2.Lx_OUT_depth-1;
                }
                cnn_autoenc_layer2.K_sparse = GUI_parameter5_int;
                printf("K_sparse change to = %d\n", cnn_autoenc_layer2.K_sparse);
                cnn_autoenc_layer2.k_sparse_sanity_check();///This should be called every time K_sparse changes
            }
            cnn_autoenc_layer2.learning_rate = ((float) GUI_parameter2_int) * 0.001f;
            cnn_autoenc_layer2.residual_gain = ((float) GUI_parameter3_int) * 0.01f;
            cnn_autoenc_layer2.copy_dictionary2visual_dict();
            cnn_autoenc_layer2.train_encoder();
            imshow("L2 rec", cnn_autoenc_layer2.reconstruct);
            imshow("L2 enc_input", cnn_autoenc_layer2.encoder_input);
            imshow("L2 enc_error", cnn_autoenc_layer2.enc_error);
            imshow("L2 noise resid", cnn_autoenc_layer2.denoised_residual_enc_input);
            imshow("L2 bias hid2out", cnn_autoenc_layer2.visual_b_hid2out);
            cv::imshow("Visual_dict_L2", cnn_autoenc_layer2.visual_dict);
            break;
        }


     //   cv::waitKey(1);
     //  imshow("L1 rec", cnn_autoenc_layer1.reconstruct);
cv::waitKey(1);
    }

    return 0;
}
