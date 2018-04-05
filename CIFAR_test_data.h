#ifndef CIFAR_TEST_DATA_H
#define CIFAR_TEST_DATA_H

#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)


class CIFAR_test_data
{
    public:
        CIFAR_test_data();
        ~CIFAR_test_data();
        void init_CIFAR(void);
        void insert_a_random_CIFAR_image(void);
        int CIFAR_height;///Output parameter
        int CIFAR_width;///Output parameter
        int nr_of_CIFAR_file_bytes;///Output parameter. Filled with relevant value when a function call get_CIFAR_file_size() is done.
        char* CIFAR_data;
        cv::Mat insert_L1_IN_convolution_cube;///Mat pointer Copied from Lx_IN_convolution_cube from first layer outside this class
    protected:

    private:
        int CIFAR_nr_of_img_p_batch;///
        int CIFAR_RGB_pixels;///
        int CIFAR_nr;///
        int CIFAR_row_size;///Internal parameter 1 byte label, 1024 RED ch, 1024 GREEN ch, 1024 BLUE ch

};



#endif // CIFAR_TEST_DATA_H

