#ifndef SPARSE_AUTOENC_H
#define SPARSE_AUTOENC_H

#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <cstdlib>///rand() srand()
//#include <ctime>
#include <math.h>  // exp
#include <stdlib.h>// exit(0);
#include <iostream>
using namespace std;

class sparse_autoenc
{
     public:
        sparse_autoenc() {}
        virtual ~sparse_autoenc() {}
        void init(void);
        void train_encoder(void);
        int training_iterations;///Input parameter. Set how many training iterations (randomized positions) on one learn_encoder() function calls
        void convolve_operation(void);
        int color_mode;///1 = CV_32FC3 color mode. 0 = CV_32FC1 gray mode. color_mode = 1 is ONLY allowed to use at Layer 1
        int init_in_from_outside;///Input Parameter. Must set to 0 when this object is the FIRST layer. MUST be = 1 when a deeper layer
        ///When init_in_from_outside = 1 then Lx_IN_convolution_cube is same poiner as the Lx_OUT_convolution_cube or Lx_pooling_cube of the previous layer
        int patch_side_size;///Example 7 will set up 7x7 patch at 2 Dimensions of the 3D
        int Lx_IN_depth;///This is set to 1 when color mode. When gray mode this is the depth of the input data
        int Lx_OUT_depth;///This is the number of atom's in the whole dictionary.
        cv::Mat Lx_IN_convolution_cube;///The input convolution cube. Pick a spatial patch data from this Input Convolution cube when learning autoencoder
        cv::Mat Lx_OUT_convolution_cube;///The output convolution cube. one point in this data is one encoder node output data
        int stride;///Convolution stride level
        int Lx_IN_widht;///Input parameter how wide will the Lx_IN_convolution_cube be.
        int Lx_IN_hight;///Input parameter how high will the Lx_IN_convolution_cube be.
        int Lx_OUT_widht;///Output parameter how wide will the Lx_OUT_convolution_cube be. Depend on patch_side_side and Lx_IN_widht
        int Lx_OUT_hight;///Output parameter how high will the Lx_OUT_convolution_cube be. Depend on patch_side_side and Lx_IN_hight
        ///No padding option implemented
        ///Pooling layer is outside this class
        cv::Mat dictionary;///
        cv::Mat visual_activation;///Same as dictionary Mat but add on a activation visualization on top of the image.
        float init_noise_gain;
        ///--- This 2 parameters below is the Stop condition parameters to use more atom's. ----
        float e_stop_threshold;///If the sparse coding result is below this threshold then stop use more numbers of atoms.
        int K_sparse;///Max atom use. This is the maximum number of atom from the dictionary used in the sparse coding. max atom use
        int use_dynamic_penalty;///0 = only fix level of the 2 parameter e_stop_threshold & K_sparse above used. 1 = penalty_add will add on the e_stop_threshold etch time a new atom is selected
        float penalty_add;///Used when use_dynamic_penalty = 1. This value will add on the e_stop_threshold each time a new atom is selected
        ///-------------------------------------------------------------------------------------
        int enable_denoising;///Input parameter
        int denoising_percent;///Input parameter 0..100
    protected:

    private:
        int dict_hight;
        int dict_width;
        int slide_count;///0..(slide_steps-1) count from 0 up to max when slide the convolution patch fit in
        int slide_steps;///This will be set to a max constant value at init
        int convolution_mode;///1 = do convolution operation not do sparse patch learning. 0= Do sparse autoenc learning process. 0= Not full convolution process.
        ///When convolve_operation() function called convolution_mode will set to = 1
        ///When train_encoder()      function called convolution_mode will set to = 0
        int* score_table;///Set up a score table of the order of strength of each atom's. Only used in sparse mode. When K_sparse = Lx_OUT_depth this not used.

};

void sparse_autoenc::convolve_operation(void)
{
    convolution_mode = 1;
}
void sparse_autoenc::train_encoder(void)
{
    convolution_mode = 0;
}

void sparse_autoenc::init(void)
{
    score_table = new int[Lx_OUT_depth];///Set up the size of the score_table will then contain the order of strength of each atom's. Only used in sparse mode. When K_sparse = Lx_OUT_depth this not used.
    convolution_mode = 0;
    int sqrt_nodes_plus1 = 0;
    if(patch_side_size > Lx_IN_hight)
    {
        printf("Parameter ERROR!\n");
        printf("patch_side_size is less the Lx_IN_hight \n");
        printf("patch_side_size =%d\n", patch_side_size);
        printf("Lx_IN_hight =%d\n", Lx_IN_hight);
        printf("patch_side_size must fit inside Lx_IN_hight because\n");
        printf("padding is NOT supported in this version.\n");
        printf("Suggest: Increase Lx_IN_hight or Decrease patch_side_size\n");
        exit(0);
    }
    if(patch_side_size > Lx_IN_widht)
    {
        printf("Parameter ERROR!\n");
        printf("patch_side_size is less the Lx_IN_widht \n");
        printf("patch_side_size =%d\n", patch_side_size);
        printf("Lx_IN_widht =%d\n", Lx_IN_widht);
        printf("patch_side_size must fit inside Lx_IN_widht because\n");
        printf("padding is NOT supported in this version.\n");
        printf("Suggest: Increase Lx_IN_widht or Decrease patch_side_size\n");
        exit(0);
    }
    if(init_noise_gain < 0.0f || init_noise_gain > 1.0f)
    {
        printf("Parameter ERROR!\n");
        printf("init_noise_gain = %f is out of range 0..1.0f\n", init_noise_gain);
        exit(0);
    }
    if(color_mode == 1)///Only
    {
        if(init_in_from_outside == 1)
        {
            printf("ERROR! color_mode is ONLY allowed at first layer, init_in_from_outside = %d\n", init_in_from_outside);
            printf("Suggest on First layer: Set init_in_from_outside = 0\n");
            printf("Suggest on deeper layer: Set color_mode = 0\n");
            exit(0);
        }
        sqrt_nodes_plus1 = sqrt(Lx_OUT_depth);///Set up a square of nodes many small patches organized in a square
        sqrt_nodes_plus1 += 1;///Plus 1 ensure that the graphic square is large enough if the sqrt() operation get round of
        dict_hight = patch_side_size * sqrt_nodes_plus1;
        dict_width = patch_side_size * sqrt_nodes_plus1;
        dictionary.create(dict_hight, dict_width, CV_32FC3);
        visual_activation.create(dict_hight, dict_width, CV_32FC3);///Show activation overlay marking on each patch.
        if(Lx_IN_depth != 1)
        {
            printf("********\n");
            printf("WARNING Lx_IN_depth should be = 1 when color_mode = 1\n");
            printf("Set Lx_IN_depth = 1 to remove this warning when color_mode = 1\n");
            printf("Lx_IN_depth of this Layer is NEVER used because color_mode = 1 \n");
            printf("then the input data is always use RGB component and Lx_IN_depth now FORCED to 1\n");
            Lx_IN_depth = 1;
            printf("Lx_IN_depth = %d\n", Lx_IN_depth );
            printf("********\n");
        }
        printf("This layer First Layer init_in_from_outside = %d\n", init_in_from_outside);
        Lx_IN_convolution_cube.create(Lx_IN_hight, Lx_IN_widht, CV_32FC3);
        printf("Lx_IN_convolution_cube are now created in COLOR mode CV_32FC3\n");

    }
    else
    {
        ///Gray mode now a deep mode is used instead with number of deep layers
        ///This graphical setup consist of many small patches (boxes) with many (boxes) rows.
        dict_hight = patch_side_size * Lx_OUT_depth;///Each patches (boxes) row correspond to one encode node
        dict_width = patch_side_size * Lx_IN_depth;///Each column of small patches (boxes) correspond to each depth level.
        dictionary.create(dict_hight, dict_width, CV_32FC1);///Only gray
        visual_activation.create(dict_hight, dict_width, CV_32FC3);/// Color only for show activation overlay marking on the gray (green overlay)
        if(init_in_from_outside == 1)
        {
            printf("This layer is a init_in_from_outside = %d\n", init_in_from_outside);
            printf("NOTE: Lx_IN_hight, Lx_IN_widht, Lx_IN_depth and \n");
            printf("Lx_OUT_convolution_cube MUST be initialized outside this class\n");
            printf("with related input from previous convolution\n");
            printf("or pool layer object to work proper\n");
        }
        else
        {
            printf("This layer should be the First Layer because init_in_from_outside = %d\n", init_in_from_outside);
            Lx_IN_convolution_cube.create(Lx_IN_hight * Lx_IN_depth, Lx_IN_widht, CV_32FC1);
            printf("Lx_IN_convolution_cube are now created in GRAY mode CV_32FC1\n");
        }
    }
    printf("pixel dict_hight = %d\n", dict_hight);
    printf("pixel dict_width = %d\n", dict_width);
    printf("Width of Lx_IN_convolution_cube, Lx_IN_widht = %d\n", Lx_IN_widht);
    printf("Hight of Lx_IN_convolution_cube, Lx_IN_hight = %d\n", Lx_IN_hight);
    printf("Pixel Size of feature patch square side, patch_side_size = %d\n", patch_side_size);
    printf("color_mode = %d\n", color_mode);
    printf("Lx_IN_depth = %d\n", Lx_IN_depth);
    printf("Lx_OUT_depth = %d\n", Lx_OUT_depth);
    printf("K_sparse = %d\n", K_sparse); ///This may changed during operation by the user control
    printf("stride = %d\n", stride);

    Lx_OUT_widht = (Lx_IN_widht - patch_side_size + 1) / stride;///No padding option is implemented in this version
    Lx_OUT_hight = (Lx_IN_hight - patch_side_size + 1) / stride;///No padding option is implemented in this version
    if(Lx_OUT_widht < 1)
    {
        printf("Error Lx_OUT_hight = %d to small\n", Lx_OUT_hight);
        exit(0);
    }
    if(Lx_OUT_hight < 1)
    {
        printf("Error Lx_OUT_hight = %d to small\n", Lx_OUT_hight);
        exit(0);
    }
    printf("Width of Lx_OUT_convolution_cube, Lx_OUT_widht = %d\n", Lx_OUT_widht);
    printf("Hight of Lx_OUT_convolution_cube, Lx_OUT_hight = %d\n", Lx_OUT_hight);
    Lx_OUT_convolution_cube.create(Lx_OUT_hight * Lx_OUT_depth, Lx_OUT_widht, CV_32FC1);
    slide_steps = Lx_OUT_widht * Lx_OUT_hight;
    printf("Convolution slide_steps (Lx_OUT_widht * Lx_OUT_hight) = %d\n", slide_steps);
    printf("use_dynamic_penalty = %d\n", use_dynamic_penalty);
    if(color_mode == 0)///Only
    {
        printf("NOTE: Lx_IN/OUT_convolution_cube is only show 2D so the depth of the cube\n");
        printf("is represented as several boxes on the vertical directions\n");
        printf("so if Lx_IN/OUT_depth is large the image of IN/OUT_cube will go below the screen\n");
        printf("enable_denoising = %d\n", enable_denoising);
    }
    srand (static_cast <unsigned> (time(0)));///Seed the randomizer
    float noise = 0.0f;
    //cv_D_mat.at<float>(cv_Row, cv_Col) =
    int is_on_patch_nr=0;

    ///Initialize random data into the dictionary
    for(int i=0; i<dict_hight; i++)///i will count up for each line on the Mat dictionary
    {
        for(int j=0; j<(dict_width * dictionary.channels()); j++)///j will count up each column on the Mat dictionary. When color mode 3-step (3-color) for each pixel
        {
            if(color_mode == 1)
            {
                is_on_patch_nr = ((i/patch_side_size)*sqrt_nodes_plus1) + (j%(dict_width * dictionary.channels()))/(patch_side_size * dictionary.channels());
//                printf("is_on_patch_nr =%d  i=%d j=%d\n ", is_on_patch_nr, i, j);
                if(is_on_patch_nr < Lx_OUT_depth)///Lx_OUT_depth is the number of atom's in the whole dictionary.
                {
                    noise = (float) (rand() % 65535) / 65536;///0..1.0 range
                    noise -= 0.5;
                    noise *= init_noise_gain;
                    noise += 0.5;
                }
                else
                {
                    noise = 0.5;
                }
            }
            else
            {
                noise = (float) (rand() % 65535) / 65536;///0..1.0 range
                noise -= 0.5;
                noise *= init_noise_gain;
                noise += 0.5;
            }
            dictionary.at<float>(i, j) = noise;
        }
    }
    printf("Init CNN layer object Done!\n");
}



#endif // SPARSE_AUTOENC_H
