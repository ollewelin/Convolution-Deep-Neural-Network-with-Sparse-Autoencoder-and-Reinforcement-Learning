#ifndef SPARSE_AUTOENC_H
#define SPARSE_AUTOENC_H

#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
//#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <cstdlib>///rand() srand()
//#include <ctime>
#include <math.h>  // exp
#include <stdlib.h>// exit(0);
#include <iostream>
using namespace std;
const int MAX_DEPTH = 9999;
///#define ALWAYS_PRINT_RELU_MAX
///#define USE_RAND_RESET_MAX_NODE
/// ======== Things for evaluation only ==========
const int ms_patch_show = 1;
int print_variable_relu_leak = 0;
/// ==============================================
class sparse_autoenc
{
public:
    sparse_autoenc() {}
    virtual ~sparse_autoenc() {}
    int use_auto_bias_level;
    float fix_bias_level;
    int pause_score_print_ms;
    int ON_OFF_print_score;
    int layer_nr;
    float learning_rate;
    float momentum;
    float residual_gain;
    float max_ReLU_auto_reset;
    void init(void);
    void train_encoder(void);
    int show_patch_during_run;///Only for evaluation/debugging
    int use_salt_pepper_noise;///Only depend in COLOR mode. 1 = black..white noise. 0 = all kinds of color noise
    int show_encoder_on_conv_cube;///Only for debugging
    void copy_visual_dict2dictionary(void);
    void copy_dictionary2visual_dict(void);
    float ReLU_function(float);
    void k_sparse_sanity_check(void);
    int training_iterations;///Input parameter. Set how many training iterations (randomized positions) on one learn_encoder() function calls
    void convolve_operation(void);
    int color_mode;///1 = CV_32FC3 color mode. 0 = CV_32FC1 gray mode. color_mode = 1 is ONLY allowed to use at Layer 1
    int init_in_from_outside;///Input Parameter. Must set to 0 when this object is the FIRST layer. MUST be = 1 when a deeper layer
    ///When init_in_from_outside = 1 then Lx_IN_data_cube is same poiner as the Lx_OUT_convolution_cube or Lx_pooling_cube of the previous layer
    int patch_side_size;///Example 7 will set up 7x7 patch at 2 Dimensions of the 3D
    int Lx_IN_depth;///This is set to 1 when color mode. When gray mode this is the depth of the input data
    int Lx_OUT_depth;///This is the number of atom's in the whole dictionary.
    cv::Mat Lx_IN_data_cube;///The input convolution cube. Pick a spatial patch data from this Input Convolution cube when learning autoencoder
    cv::Mat Lx_OUT_convolution_cube;///The output convolution cube. one point in this data is one encoder node output data
    int stride;///Convolution stride level
    int Lx_IN_widht;///Input parameter how wide will the Lx_IN_data_cube be.
    int Lx_IN_hight;///Input parameter how high will the Lx_IN_data_cube be.
    int Lx_OUT_widht;///Output parameter how wide will the Lx_OUT_convolution_cube be. Depend on patch_side_side and Lx_IN_widht
    int Lx_OUT_hight;///Output parameter how high will the Lx_OUT_convolution_cube be. Depend on patch_side_side and Lx_IN_hight
    ///No padding option implemented
    ///Pooling layer is outside this class
    //cv::cuda::GpuMat dictionary;///Straight follow Boxes Downwards memory of dictionary
    cv::Mat dictionary;///Straight follow Boxes Downwards memory of dictionary
    cv::Mat visual_dict;///Visual organization of the Mat dictionary
    cv::Mat visual_activation;///Same as visual_dict Mat but add on a activation visualization on top of the image.
    cv::Mat encoder_input;///depth Z = Lx_IN_depth. layer size X, Y = patch_size * patch_size
    cv::Mat denoised_residual_enc_input;///same size as encoder_input. Used when enable_denoising = 1 for noise or if Greedy encoder mode = 1 for residual
    cv::Mat reconstruct;///same size as encoder_input
    cv::Mat enc_error;///same size as encoder_input
    cv::Mat eval_indata_patch;
    cv::Mat eval_atom_patch;
    cv::Mat bias_in2hid;///Column 0 = the weight. Column 1 = change weight
    cv::Mat bias_hid2out;
    cv::Mat visual_b_hid2out;///Same as bias_hid2out but + 0.5 for visualize
    cv::Mat visual_b_in2hid;///Same as bias_in2hid but + 0.5 for visualize
    float init_noise_gain;

    ///Regarding ReLU Leak function
    void random_change_ReLU_leak_variable(void);///Should be called to change the variable variable_relu_leak variable
    int use_leak_relu;///if = 1 then the ReLU (Rectify Linear function) function have a negative leak also to prevent dying node's
    float fix_relu_leak_gain;///0.0< X <1.0. Should have some low value close to 0.0 This is used when use_variable_leak_relu = 0
    int use_variable_leak_relu;///it = 1 Advanced Leak ReLU function with stochastic negative leak
    float relu_leak_gain_variation;///used when use_variable_leak_relu = 1. relu_leak_gain_variation + min_relu_leak_gain must bee <1.0f
    float min_relu_leak_gain;///used when use_variable_leak_relu = 1.
    float score_bottom_level;///Should be 0.0f or less. Start bottom level of score table. This is forced to 0.0f when use_leak_relu = 0
    ///-----------------------------

    ///--- This 2 parameters below is the Stop condition parameters to use more atom's. ----
    float e_stop_threshold;///If the sparse coding result is below this threshold then stop use more numbers of atoms.
    int K_sparse;///Max atom use. This is the maximum number of atom from the dictionary used in the sparse coding. max atom use
    int use_dynamic_penalty;///0 = only fix level of the 2 parameter e_stop_threshold & K_sparse above used. 1 = penalty_add will add on the e_stop_threshold etch time a new atom is selected
    float penalty_add;///Used when use_dynamic_penalty = 1. This value will add on the e_stop_threshold each time a new atom is selected
    ///-------------------------------------------------------------------------------------

///TODO denoising not implemented yet
    int enable_denoising;///Input parameter. If enabled the cv::Mat denoised_residual_enc_input used
    int denoising_percent;///Input parameter 0..100
///
    int use_greedy_enc_method;///Use Greedy encoder selection of atom's take longer time but better solver. If enabled the cv::Mat denoised_residual_enc_input used
    int print_greedy_reused_atom;
    float encoder_loss;
    float get_noise(void);
protected:

private:
    cv::Mat change_w_dict;///Change weight's used for update weight's
 ///   cv::Mat change_w_b_in2hid;///Change weight's used for update weight's
 ///   change_w_b_in2hid replaced by Column 0 = the weight. Column 1 = change weight on bias_in2hid
    cv::Mat change_w_b_hid2out;///Change weight's used for update weight's
    float bias_node_level;
    int v_dict_hight;
    int v_dict_width;
    int dict_h;
    int dict_w;
    float variable_relu_leak;///
    float actual_relu_leak_level;///The actual relu leak level
    int slide_count;///0..(slide_steps-1) count from 0 up to max when slide the convolution patch fit in
    int slide_steps;///This will be set to a max constant value at init
    int convolution_mode;///1 = do convolution operation not do sparse patch learning. 0= Do sparse autoenc learning process. 0= Not full convolution process.
    ///When convolve_operation() function called convolution_mode will set to = 1
    ///When train_encoder()      function called convolution_mode will set to = 0
    int *score_table;///Set up a score table of the order of strength of each atom's. Only used in sparse mode. When K_sparse = Lx_OUT_depth this not used.
    int sqrt_nodes_plus1;
    int max_patch_h_offset;
    int max_patch_w_offset;
    float *train_hidden_node;///Only used in train_encoder() function
    float *temp_hidden_node;///Used in Greedy method
    float *train_hidden_deleted_max;///Same size as train_hidden_node but erase all max value when fill score table
    ///======== Set up pointers for Mat direct address (fastest operation) =============
    float *zero_ptr_dict;///Set up pointer for fast direct address of Mat
    float *index_ptr_dict;///Set up pointer for fast direct address of Mat
    float *zero_ptr_vis_act;///Set up pointer for fast direct address of Mat
    float *index_ptr_vis_act;///Set up pointer for fast direct address of Mat
    float *zero_ptr_Lx_IN_data;///Set up pointer for fast direct address of Mat
    float *index_ptr_Lx_IN_data;///Set up pointer for fast direct address of Mat
    float *zero_ptr_Lx_OUT_conv;///Set up pointer for fast direct address of Mat
    float *index_ptr_Lx_OUT_conv;///Set up pointer for fast direct address of Mat
    float *sanity_check_ptr;///Only for test
    float *zero_ptr_encoder_input;///Set up pointer for fast direct address of Mat
    float *index_ptr_encoder_input;///Set up pointer for fast direct address of Mat
    float *zero_ptr_deno_residual_enc;///Set up pointer for fast direct address of Mat
    float *index_ptr_deno_residual_enc;///Set up pointer for fast direct address of Mat
    float *zero_ptr_reconstruct;///Set up pointer for fast direct address of Mat
    float *index_ptr_reconstruct;///Set up pointer for fast direct address of Mat
    float *zero_ptr_chan_w_dict;
    float *index_ptr_chan_w_dict;
    float *zero_ptr_chan_w_b_hid2out;
    float *index_ptr_chan_w_b_hid2out;
    float *zero_ptr_enc_error;
    float *index_ptr_enc_error;
    float *zero_ptr_bias_hid2out;
    float *index_ptr_bias_hid2out;
    ///=================================================================================
    void insert_patch_noise(void);
    void check_dictionary_ptr_patch(void);
    void print_score_table_f(void);

    float check_remove_Nan(float);
    void lerning_autencoder(void);
    void insert_enc_noise(int);
    int noise_probablity;/// noise_probablity = (65535 * denoising_percent) / 100;

};
void sparse_autoenc::lerning_autencoder(void)
{

    if(color_mode==1)
    {
        reconstruct = cv::Scalar(0.5f, 0.5f, 0.5f);///Start with a neutral (gray) image
       // reconstruct = cv::Scalar(0.0f, 0.0f, 0.0f);///Start with a neutral (black) image
    }
    else
    {
        reconstruct = cv::Scalar(0.0f);///Start with a neutral (gray) image
    }

    ///Add bias signal to reconstruction
    reconstruct += bias_hid2out * bias_node_level;
    for(int i=0; i<K_sparse; i++) ///Search through the most strongest atom's and do ReLU non linear activation function of hidden nodes
    {
        int at_node = 0;
        ///K_sparse could be set up to Lx_OUT_depth. If K_sparse = LxOUT_depth then there is not sparse mode
        if(K_sparse != Lx_OUT_depth)///Check if this encoder are set in sparse mode
        {
            ///Sparse mode
            if((score_table[i]) == -1)///Break if the score table tell that the atom's in use is less then up to K_sparse
            {
                break;///atom's in use this time is fewer then K_sparse
                ///Note this break event will probably never happen if score_bottom_level is set to something less then 0.0f
            }
            at_node = (score_table[i]);
        }
        else
        {
            ///No sparsity Mode setting. All atom's used
            at_node = i;
        }
        ///Train selected atom's procedure
        ///Step 1. Make reconstruction
        ///Step 2. Calculate each pixel's error. Sum up the total loss for report. Also set the hidden_delta for step 4
        ///Step 3. Update patch weights (and bias_hid2out weights also)
        ///Step 4. Update bias_in2hid regarding the hidden_delta

        ///============================================================
        ///Step 1. Make reconstruction from one hidden node each turn
        ///============================================================
        ///COLOR or GRAY mode reconstruction
        index_ptr_reconstruct = zero_ptr_reconstruct;
        index_ptr_dict = zero_ptr_dict + at_node * patch_side_size*patch_side_size*reconstruct.channels()*Lx_IN_depth;
        for(int j=0; j<Lx_IN_depth; j++)
        {
            for(int k=0; k<patch_side_size*patch_side_size*reconstruct.channels(); k++)
            {
                *index_ptr_reconstruct += train_hidden_node[at_node] * (*index_ptr_dict);
                index_ptr_dict++;
                index_ptr_reconstruct++;
            }
        }
    }
    ///============================================================
    ///Step 1. complete reconstruction from one hidden node
    ///============================================================

    ///============================================================
    ///Step 2. Calculate each pixel's error. Sum up the total loss for report
    ///============================================================
    enc_error = encoder_input - reconstruct;       ///Calculate pixel error. encoder_input - reconstruction

    ///============================================================
    ///Step 2. complete
    ///============================================================

    ///====== First do weight update for bias_hid2out weights and make encoder_loss ==========
    encoder_loss=0.0f;
    index_ptr_enc_error     = zero_ptr_enc_error;
    index_ptr_chan_w_b_hid2out = zero_ptr_chan_w_b_hid2out;
    index_ptr_bias_hid2out  = zero_ptr_bias_hid2out;

    ///Update autoencoder weight in this way;
    ///change_weight = learning_rate * node * delta_out + momentum * change_weight;
    ///weight        = weight + change_weight;
    ///In this case when node = 1.0f because bias node is ALWAYS = 1. delta is = pixel error in this case when encoder used
    ///NOTE in this case (autoencoder case) delta_out = encoder_error = (*index_ptr_enc_error)

    ///COLOR or GRAY mode
    for(int j=0; j<Lx_IN_depth; j++)
    {
        for(int k=0; k<(patch_side_size*patch_side_size*encoder_input.channels()); k++)
        {
            encoder_loss += (*index_ptr_enc_error) * (*index_ptr_enc_error);///Loss = error²
            ///change_weight = learning_rate * node * delta_out + momentum * change_weight
            ///weight       += change_weight
            (*index_ptr_chan_w_b_hid2out) = learning_rate * bias_node_level * (*index_ptr_enc_error) + momentum * (*index_ptr_chan_w_b_hid2out);
            (*index_ptr_bias_hid2out)    += (*index_ptr_chan_w_b_hid2out);

            index_ptr_bias_hid2out++;
            index_ptr_chan_w_b_hid2out++;
            index_ptr_enc_error++;
        }
    }
    ///======= bias_hid2out weights updated and make encoder_loss finish ===============
    float deriv_gradient_decent = 0.0f;
    for(int i=0; i<K_sparse; i++) ///
    {
        int addr_offset = 0;
        float temp_train_hidd_node = 0.0;
        ///K_sparse could be set up to Lx_OUT_depth. If K_sparse = LxOUT_depth then there is not sparse mode
        if(K_sparse != Lx_OUT_depth)///Check if this encoder are set in sparse mode
        {
            ///Sparse mode
            if((score_table[i]) == -1)///Break if the score table tell that the atom's in use is less then up to K_sparse
            {
                break;///atom's in use this time is fewer then K_sparse
                ///Note this break event will probably never happen if score_bottom_level is set to something less then 0.0f
            }
            temp_train_hidd_node = train_hidden_node[ (score_table[i]) ];///
            addr_offset = (score_table[i]) * patch_side_size*patch_side_size*encoder_input.channels()*Lx_IN_depth;
        }
        else
        {
            ///No sparsity Mode setting. All atom's used
            temp_train_hidd_node = train_hidden_node[i];///Use all hidden nodes
            addr_offset = i * patch_side_size*patch_side_size*encoder_input.channels()*Lx_IN_depth;
        }

        ///====== Do dictionary weight update, make encoder_loss and set hidden_delta
        index_ptr_enc_error     = zero_ptr_enc_error;
        index_ptr_chan_w_dict   = zero_ptr_chan_w_dict + addr_offset;
        index_ptr_dict          = zero_ptr_dict + addr_offset;

        float temp_hidden_delta = 0.0f;
        ///COLOR or GRAY mode
        if(temp_train_hidd_node < 0.0f)
        {
            deriv_gradient_decent = actual_relu_leak_level;
        }
        else
        {
            deriv_gradient_decent = 1.0f;
        }
        for(int j=0; j<Lx_IN_depth; j++)
        {
            for(int k=0; k<(patch_side_size*patch_side_size*encoder_input.channels()); k++)
            {
                ///delta_hid = delta_out * weight;
                ///change_weight = learning_rate * node * delta_out + momentum * change_weight;
                ///weight        = weight + change_weight;
                ///NOTE in this case (autoencoder case) delta_out = encoder_error = (*index_ptr_enc_error)

                ///First backprop delta to hidden_delta so it is possible to update bias_in2hid weight later.
                temp_hidden_delta += (*index_ptr_enc_error) * (*index_ptr_chan_w_dict) * deriv_gradient_decent;

                ///Now update tied weight's
                (*index_ptr_chan_w_dict) = learning_rate * temp_train_hidd_node * deriv_gradient_decent * (*index_ptr_enc_error) + momentum * (*index_ptr_chan_w_dict);///change_weight = learning_rate * node * delta + momentum * change_weight;
                (*index_ptr_dict)       += (*index_ptr_chan_w_dict);///weight        = weight + change_weight;
                index_ptr_enc_error++;
                index_ptr_chan_w_dict++;
                index_ptr_dict++;
            }
        }
        ///  Update bias_in2hid weight's from hidden_delta data
        ///  bias_in2hid.create(Lx_OUT_depth, 2, CV_32FC1);///Column 0 = the weight. Column 1 = change weight

        ///change_weight = learning_rate * node * delta_hid + momentum * change_weight;
        ///weight        = weight + change_weight;
        ///In this case when node = 1.0f because bias node is ALWAYS = 1. delta_hid is backpropagate above

        if(K_sparse != Lx_OUT_depth)///Check if this encoder are set in sparse mode
        {
            bias_in2hid.at<float>((score_table[i]), 1)  = learning_rate * bias_node_level * temp_hidden_delta + momentum * bias_in2hid.at<float>((score_table[i]), 1);///Column 1 is the change weight data
            bias_in2hid.at<float>((score_table[i]), 0) += bias_in2hid.at<float>((score_table[i]), 1);///Column 0 is the weight data. Column 1 is the change weigh
        }
        else
        {
            bias_in2hid.at<float>(i, 1)  = learning_rate * bias_node_level * temp_hidden_delta + momentum * bias_in2hid.at<float>(i, 1);///Column 1 is the change weight data
            bias_in2hid.at<float>(i, 0) += bias_in2hid.at<float>(i, 1);///Column 0 is the weight data. Column 1 is the change weight data
        }
     }
}

inline float sparse_autoenc::check_remove_Nan(float f_input)
{
    float f_output = f_input;
    if(f_input != f_input)
    {
        printf("WARNING! float NaN occur Set to 0.001f\n");
        f_output = 0.001f;
    }
    return f_output;
}
inline float sparse_autoenc::ReLU_function(float input_value)
{
    float ReLU_result = 0.0f;
    float temp_random = 0.0f;
    if(input_value < 0.0f)
    {
        ReLU_result = actual_relu_leak_level;
    }
    else
    {
        ReLU_result = input_value;
    }

    if(ReLU_result > max_ReLU_auto_reset)
    {
#ifdef USE_RAND_RESET_MAX_NODE
        temp_random = (float) (rand() % 65535) / 65536;///0..1.0 range
        ReLU_result = temp_random * max_ReLU_auto_reset * 1.0f;
#else
        ReLU_result = max_ReLU_auto_reset * 0.95f;
#endif // USE_RAND_RESET_MAX_NODE
        if(ON_OFF_print_score == 1)
        {
            printf("reach Max ReLU set to %f\n", ReLU_result);
        }
#ifdef ALWAYS_PRINT_RELU_MAX
printf("reach Max ReLU set to %f\n", ReLU_result);
#endif // ALWAYS_PRINT_RELU_MAX

    }
    return ReLU_result;
}

void sparse_autoenc::random_change_ReLU_leak_variable(void)
{
    if(use_variable_leak_relu != 1)
    {
        printf("ERROR! use_variable_leak_relu must be = 1 to allow call random_change_ReLU_leak_variable function\n");
        exit(0);
    }
    float temporary_random = 0.0f;
    float random_range = relu_leak_gain_variation - min_relu_leak_gain;
    temporary_random = (float) (rand() % 65535) / 65536;///0..1.0 range
    temporary_random *= random_range;
    variable_relu_leak = min_relu_leak_gain + temporary_random;
    actual_relu_leak_level = variable_relu_leak;

    if(print_variable_relu_leak == 1)
    {
        printf("actual_relu_leak_level = %f\n", actual_relu_leak_level);
    }
}

void sparse_autoenc::copy_dictionary2visual_dict(void)
{
///Why is this need ?..
///dictionary is Straight follow memory that is good for high speed Dot product operation.
///dictionary is organized in a long (long of many features choses) graphic row of patches.
///Therefor there it is more suitable to show this dictionary data in a more square like image with several patches in both X and Y direction
    int dict_ROW = 0;
    int dict_COL = 0;

    if(color_mode == 1)
    {
        ///COLOR mode the input depth is 1 with 3 COLOR
        index_ptr_dict = zero_ptr_dict;
        for(int i=0;i<Lx_OUT_depth;i++)
        {
            for(int k=0;k<patch_side_size*patch_side_size*dictionary.channels();k++)
            {
                dict_ROW = (patch_side_size * (i/sqrt_nodes_plus1)) + (k/(patch_side_size*dictionary.channels()));
                dict_COL = ((i%sqrt_nodes_plus1) * patch_side_size * dictionary.channels()) + (k%(patch_side_size*dictionary.channels()));

                visual_dict.at<float>(dict_ROW, dict_COL) = *index_ptr_dict + 0.5f;
                index_ptr_dict++;///Direct is fast but no sanity check. Must have control over this pointer otherwise Segmentation Fault could occur.
            }
        }
        ///Now we could check the sanity of index_ptr_dict pointer address
        check_dictionary_ptr_patch();
    }
    else
    {
        ///GRAY mode the input depth is arbitrary
        index_ptr_dict = zero_ptr_dict;
        for(int i=0;i<Lx_OUT_depth;i++)
        {
            for(int j=0; j<Lx_IN_depth; j++)///IN depth is arbitrary in GRAY mode
            {
                for(int k=0; k<patch_side_size*patch_side_size*dictionary.channels(); k++)
                {
                    dict_ROW = (j * patch_side_size) + (k/patch_side_size);
                    dict_COL = (i * patch_side_size) + (k%patch_side_size);
                    visual_dict.at<float>(dict_ROW, dict_COL) = *index_ptr_dict + 0.5f;
                    index_ptr_dict++;///Direct is fast but no sanity check. Must have control over this pointer otherwise Segmentation Fault could occur.
                }
            }
        }
        ///Now we could check the sanity of index_ptr_dict pointer address
        check_dictionary_ptr_patch();
    }

    visual_b_in2hid = bias_in2hid + cv::Scalar(0.5f);
    if(color_mode == 1)
    {
        visual_b_hid2out = bias_hid2out + cv::Scalar(0.5f, 0.5f, 0.5f);
    }
    else
    {
        visual_b_hid2out = bias_hid2out + cv::Scalar(0.5f);
    }
}
void sparse_autoenc::copy_visual_dict2dictionary(void)
{
///Why is this need ?..
///dictionary is Straight follow memory thats good for high speed Dot product operation.
///dictionary is organized in a long (long of many features choses) graphic column of patches.
///Therefor there is more suitable to show this dictionary data in a more square like image with several patches in both X and Y direction
    int dict_ROW = 0;
    int dict_COL = 0;

    if(color_mode == 1)
    {
        ///COLOR mode the input depth is 1 with 3 COLOR
        index_ptr_dict = zero_ptr_dict;
        for(int i=0;i<Lx_OUT_depth;i++)
        {
            for(int k=0;k<patch_side_size*patch_side_size*dictionary.channels();k++)
            {
                dict_ROW = (patch_side_size * (i/sqrt_nodes_plus1)) + (k/(patch_side_size*dictionary.channels()));
                dict_COL = ((i%sqrt_nodes_plus1) * patch_side_size * dictionary.channels()) + (k%(patch_side_size*dictionary.channels()));
                *index_ptr_dict = visual_dict.at<float>(dict_ROW, dict_COL) - 0.5f;
                index_ptr_dict++;///Direct is fast but no sanity check. Must have control over this pointer otherwise Segmentation Fault could occur.
            }
        }
        ///Now we could check the sanity of index_ptr_dict pointer address
        check_dictionary_ptr_patch();
    }
    else
    {
        ///GRAY mode the input depth is arbitrary
        index_ptr_dict = zero_ptr_dict;
        for(int i=0;i<Lx_OUT_depth;i++)
        {
            for(int j=0; j<Lx_IN_depth; j++)///IN depth is arbitrary in GRAY mode
            {
                for(int k=0; k<patch_side_size*patch_side_size*dictionary.channels(); k++)
                {
                    dict_ROW = (j * patch_side_size) + (k/patch_side_size);
                    dict_COL = (i * patch_side_size) + (k%patch_side_size);
                    *index_ptr_dict = visual_dict.at<float>(dict_ROW, dict_COL) - 0.5f;
                    index_ptr_dict++;///Direct is fast but no sanity check. Must have control over this pointer otherwise Segmentation Fault could occur.
                }
            }
        }
        ///Now we could check the sanity of index_ptr_dict pointer address
        check_dictionary_ptr_patch();
    }
}
void sparse_autoenc::convolve_operation(void)
{
    convolution_mode = 1;
}
inline void sparse_autoenc::print_score_table_f(void)
{
                /// ======= Only for evaluation =========
            if(ON_OFF_print_score == 1)
            {
                ///print table
                for(int i=0; i<Lx_OUT_depth; i++)
                {
                    if((score_table[i]) == -1)
                    {
                        break;
                       // printf("score_table[%d] = %d\n", i, score_table[i]);
                    }
                    else
                    {
                        printf("score_table[%d] = %d node = %f\n", i, score_table[i], train_hidden_node[(score_table[i])]);
                    }
                }
                printf("=========================\n");
                cv::waitKey(pause_score_print_ms);
            }
            /// ======= End evaluation ===========
}
inline void sparse_autoenc::insert_enc_noise(int k)
{
    float input_offset = -0.5f;
    static int chose_nois_dice = 0;
    static float salt_pepper_noise = 0.0f;
    if(enable_denoising == 1)
    {
        if(dictionary.channels() == 3 && use_salt_pepper_noise == 1)///Color mode
        {
            if((k%3) == 0)
            {
                chose_nois_dice = (rand() % 65535);//0..65535 range
                salt_pepper_noise = (float) (rand() % 65535) / 65536;//
            }

            if(chose_nois_dice < noise_probablity)
            {
                ///This pixel should contain noise;
                *index_ptr_deno_residual_enc = salt_pepper_noise;//0..1.0 range
            }
            else
            {
                *index_ptr_deno_residual_enc = *index_ptr_Lx_IN_data + input_offset;///This is for prepare for the autoencoder
            }

        }
        else
        {
            int chose_nois_dice = (rand() % 65535);//0..65535 range
            if(chose_nois_dice < noise_probablity)
            {
                ///This pixel should contain noise;
                *index_ptr_deno_residual_enc = (float) (rand() % 65535) / 65536;//0..1.0 range
            }
            else
            {
                *index_ptr_deno_residual_enc = *index_ptr_Lx_IN_data + input_offset;///This is for prepare for the autoencoder
            }

        }
    }
    else
    {
        *index_ptr_deno_residual_enc = *index_ptr_Lx_IN_data + input_offset;///This is for prepare for the autoencoder
    }
}

void sparse_autoenc::train_encoder(void)
{
    if(use_auto_bias_level == 1)
    {
        bias_node_level =  ((float) K_sparse) / ((float) Lx_OUT_depth);
    }
    else
    {
        bias_node_level = fix_bias_level;
    }
    float dot_product = 0.0f;
    int patch_row_offset=0;///This will point where the start upper row of the part of input data how will be dot product with the patch atom
    int patch_col_offset=0;///This will point where the start left column of the part of input data how will be dot product with the patch
    patch_row_offset = (int) (rand() % (max_patch_h_offset +1));///Randomize a start row of where input data patch will dot product with patch.
    patch_col_offset = (int) (rand() % (max_patch_w_offset +1));

    float max_temp = score_bottom_level;
    index_ptr_dict              = zero_ptr_dict;///Set dictionary Mat pointer to start point
    index_ptr_encoder_input     = zero_ptr_encoder_input;///
    index_ptr_deno_residual_enc = zero_ptr_deno_residual_enc;///
    noise_probablity = (65535 * denoising_percent) / 100;
    if(use_greedy_enc_method == 1)
    {

        for(int i=0; i<Lx_OUT_depth; i++) ///-1 tell that this will not used
        {
            score_table[i] = -1;///Clear the table
        }
        ///First copy over Lx_IN_data to Mat encoder_input and denoised_residual_enc_input
        for(int j=0; j<Lx_IN_depth; j++)
        {
            for(int k=0; k<(patch_side_size*patch_side_size*dictionary.channels()); k++)
            {
                if(color_mode == 1)
                {
                    index_ptr_Lx_IN_data = zero_ptr_Lx_IN_data + ((patch_row_offset + k/(patch_side_size*dictionary.channels())) * (Lx_IN_widht * Lx_IN_data_cube.channels()) + (k%(patch_side_size*dictionary.channels())) + (patch_col_offset * Lx_IN_data_cube.channels()));
                }
                else
                {
                    index_ptr_Lx_IN_data = zero_ptr_Lx_IN_data + ((j * Lx_IN_hight * Lx_IN_widht) + ((patch_row_offset + k/patch_side_size) * Lx_IN_widht) + (k%patch_side_size) + (patch_col_offset));
                }
                ///=========== Copy over the input data to encoder_input =========
                *index_ptr_encoder_input     = *index_ptr_Lx_IN_data;///This is for prepare for the autoencoder
                insert_enc_noise(k);
                index_ptr_encoder_input++;
                index_ptr_deno_residual_enc++;
                ///=========== End copy over the input data to encoder_input =====
            }
        }

        int node_already_selected_befor = 0;
        int search_turn = 0;
        for(int h=0; h<K_sparse || node_already_selected_befor == 1; h++)///Run through K_sparse time and select by Greedy method and make residual each h turn
        {
            if(search_turn > (10*K_sparse))
            {
                printf("Break search turn reach max search = %d\n", search_turn);
                break;
            }
            search_turn++;
            if(node_already_selected_befor == 1)
            {
                h--;///search strongest again on this score table place
            }
            index_ptr_dict          = zero_ptr_dict;///Set dictionary Mat pointer to start point
            ///COLOR or GRAY dictionary access
            for(int i=0; i<Lx_OUT_depth; i++)
            {
                index_ptr_deno_residual_enc = zero_ptr_deno_residual_enc;///
                dot_product = 0.0f;
                ///Make dot product between dictionary and index_ptr_deno_residual_enc
                ///and store the result in
                ///temp_hidden_node[i] = dot_product;
                for(int j=0; j<Lx_IN_depth; j++)
                {

                    for(int k=0; k<(patch_side_size*patch_side_size*dictionary.channels()); k++)
                    {
                        dot_product += (*index_ptr_deno_residual_enc) * (*index_ptr_dict);
                        index_ptr_deno_residual_enc++;
                        index_ptr_dict++;
                        if(show_patch_during_run == 1)///Only for debugging)
                        {
                            int eval_ROW = k/(patch_side_size*Lx_IN_data_cube.channels());
                            int eval_COL = k%(patch_side_size*Lx_IN_data_cube.channels());
                            eval_indata_patch.at<float>(eval_ROW, eval_COL)   = *index_ptr_Lx_IN_data;
                            eval_atom_patch.at<float>(eval_ROW, eval_COL)     = *index_ptr_dict + 0.5f;
                        }
                    }
                }
                ///and store the result in
                dot_product += bias_in2hid.at<float>(i, 0) * bias_node_level;
                temp_hidden_node[i] = ReLU_function(dot_product);
                //temp_hidden_node[i] = (dot_product);
            }///i<Lx_OUT_depth loop end
            ///Do the score table
            ///Make the score table, select out by score on order the K_sparse strongest atom's of the dictionary
            ///h will Search through the most strongest atom's

            max_temp = score_bottom_level;
            int strongest_atom_nr = -1;
            for(int j=0; j<Lx_OUT_depth; j++)
            {
                if(max_temp < (temp_hidden_node[j]))
                {
                    max_temp = (temp_hidden_node[j]);
                    strongest_atom_nr = j;
                }
            }
            node_already_selected_befor = 0;
            if(strongest_atom_nr != -1 && h>0)
            {
                for(int i=0; i<h; i++)
                {
                    if(score_table[i] == strongest_atom_nr)
                    {
                        node_already_selected_befor = 1;
                        if(ON_OFF_print_score == 1)
                        {
                            printf(" h = %d i = %d Rerun atom %d was already find before. search_turn %d\n", h, i, strongest_atom_nr, search_turn);
                        }
                    }
                }
            }
            if(strongest_atom_nr == -1)
            {
                score_table[h] = -1;///No atom's was stronger then score_bottom_level.
                break;///No more sparse all atom's was below score_bottom_level.
            }
            else
            {
                score_table[h] = strongest_atom_nr;///h is the K_sparse loop counter
            }
            if(-1 < strongest_atom_nr && strongest_atom_nr < Lx_OUT_depth)
            {
                index_ptr_dict              = zero_ptr_dict + strongest_atom_nr * (patch_side_size*patch_side_size*dictionary.channels()*Lx_IN_depth);///Set dictionary Mat pointer to start point
            }
            else
            {
                printf("ERROR! strongest_atom_nr = %d is outside range\n", strongest_atom_nr);
                exit(0);
            }

            index_ptr_deno_residual_enc = zero_ptr_deno_residual_enc;///
            ///Update residual data regarding the last selected atom's
            for(int i=0; i<Lx_IN_depth; i++)
            {
                for(int j=0; j<(patch_side_size*patch_side_size*dictionary.channels()); j++)
                {
                    *index_ptr_deno_residual_enc -= (*index_ptr_dict) * temp_hidden_node[strongest_atom_nr] * residual_gain;
                    index_ptr_deno_residual_enc++;
                    index_ptr_dict++;
                }
            }
            train_hidden_node[strongest_atom_nr] = temp_hidden_node[strongest_atom_nr];///Store the greedy strongest atom's in train_hidden_node[]
        }///h<K_sparse loop end
        /// ======= Only for evaluation =========
        print_score_table_f();
        /// ======= End evaluation ===========
///============= End Dot product and score table in Greedy mode ============
///=============

        encoder_loss = 0.0f;///Clear
        /// lerning_autencoder() will do this:
        ///Train selected atom's procedure
        ///Step 1. Make reconstruction
        ///Step 2. Calculate each pixel's error. Sum up the total loss for report. Also set the hidden_delta[] for step 4
        ///Step 3. Update patch weights (and bias_hid2out weights also)
        ///Step 4. Update bias_in2hid regarding the hidden_delta
        lerning_autencoder();///function do Step1..4

    }

    else
    {
        ///NON greedy method
        if(color_mode == 1)///When color mode there is another data access of the dictionary
        {

            ///COLOR or GRAY dictionary access
            for(int i=0; i<Lx_OUT_depth; i++)
            {
                ///Do the dot product (scalar product) of all the atom's in the dictionary with the input data on Lx_IN_data_cube
                dot_product = 0.0f;
                for(int j=0; j<Lx_IN_depth; j++)
                {

                    for(int k=0; k<(patch_side_size*patch_side_size*dictionary.channels()); k++)
                    {
                        if(color_mode == 1)
                        {
                            index_ptr_Lx_IN_data = zero_ptr_Lx_IN_data + ((patch_row_offset + k/(patch_side_size*dictionary.channels())) * (Lx_IN_widht * Lx_IN_data_cube.channels()) + (k%(patch_side_size*dictionary.channels())) + (patch_col_offset * Lx_IN_data_cube.channels()));
                        }
                        else
                        {
                            index_ptr_Lx_IN_data = zero_ptr_Lx_IN_data + ((j * Lx_IN_hight * Lx_IN_widht) + ((patch_row_offset + k/patch_side_size) * Lx_IN_widht) + (k%patch_side_size) + (patch_col_offset));
                        }
                        dot_product += (*index_ptr_Lx_IN_data) * (*index_ptr_dict);
                        index_ptr_dict++;///
                        if(show_patch_during_run == 1)///Only for debugging)
                        {
                            int eval_ROW = k/(patch_side_size*Lx_IN_data_cube.channels());
                            int eval_COL = k%(patch_side_size*Lx_IN_data_cube.channels());
                            eval_indata_patch.at<float>(eval_ROW, eval_COL)   = *index_ptr_Lx_IN_data;
                            eval_atom_patch.at<float>(eval_ROW, eval_COL)     = *index_ptr_dict + 0.5f;
                        }
                        ///=========== Copy over the input data to encoder_input =========
                        if(i==0)///Do this copy input data to encoder_input ones on Lx_OUT_depth 0, not for every Lx_OUT_depth turn
                        {
                            ///=========== Copy over the input data to encoder_input =========
                            *index_ptr_encoder_input     = *index_ptr_Lx_IN_data;///This is for prepare for the autoencoder
                            insert_enc_noise(k);
                            index_ptr_encoder_input++;
                            index_ptr_deno_residual_enc++;
                            ///=========== End copy over the input data to encoder_input =====
                        }
                        ///=========== End copy over the input data to encoder_input =====
                    }
                }
                dot_product += bias_in2hid.at<float>(i, 0) * bias_node_level;

                if(show_patch_during_run == 1)///Only for debugging)
                {
                    imshow("patch", eval_indata_patch);
                    imshow("atom", eval_atom_patch);
                    cv::waitKey(ms_patch_show);
                }
                ///Put this dot product into train_hidden_node
                train_hidden_node[i] = ReLU_function(dot_product);
                //train_hidden_node[i] = dot_product;
                train_hidden_deleted_max[i] = train_hidden_node[i];

            }

        }

        encoder_loss = 0.0f;///Clear
        if(K_sparse != Lx_OUT_depth)///Check if this encoder are set in sparse mode
        {
            for(int i=0; i<Lx_OUT_depth; i++) ///-1 tell that this will not used
            {
                score_table[i] = -1;///Clear the table
            }
            ///Do sparse constraints
            ///Make the score table, select out by score on order the K_sparse strongest atom's of the dictionary
            for(int i=0; i<K_sparse; i++) ///Search through the most strongest atom's
            {
                float max_temp = score_bottom_level;
                for(int j=0; j<Lx_OUT_depth; j++)
                {
                    if(max_temp < (train_hidden_deleted_max[j]))
                    {
                        max_temp = (train_hidden_deleted_max[j]);
                        score_table[i] = j;
                    }
                }
                int index_delete_this = score_table[i];
                train_hidden_deleted_max[index_delete_this] = score_bottom_level;///Delete this max value so next check will search for next strongest atom's and mark in score table
            }

            /// ======= Only for evaluation =========
            print_score_table_f();
            /// ======= End evaluation ===========

            /// lerning_autencoder() will do this:
            ///Train selected atom's procedure
            ///Step 1. Make reconstruction
            ///Step 2. Calculate each pixel's error. Sum up the total loss for report. Also set the hidden_delta[] for step 4
            ///Step 3. Update patch weights (and bias_hid2out weights also)
            ///Step 4. Update bias_in2hid regarding the hidden_delta
            lerning_autencoder();///function do Step1..4

        }
        else
        {
            ///Not in sparse mode. No sparse constraints this mean's that all atom's in dictionary will be used to represent the reconstruction
            ///and all atom's will also be trained every cycle.
            lerning_autencoder();///function do Step1..4
        }


    }

    if(show_encoder_on_conv_cube==1)///If safe CPU time turn of this during Autoencoder learning
    {
        for(int h=0;h<K_sparse;h++)
        {
            int i=0;
            if(score_table[h] == -1)
            {
                break;
            }
            else
            {
                if(K_sparse != Lx_OUT_depth)
                {
                i = score_table[h];

                }
                else
                {
                    i = h;
                }
                index_ptr_Lx_OUT_conv = zero_ptr_Lx_OUT_conv + (i * Lx_OUT_widht * Lx_OUT_hight) + (patch_row_offset * Lx_OUT_widht) + (patch_col_offset);
                *index_ptr_Lx_OUT_conv = train_hidden_node[i];
            }
        }
    }

    convolution_mode = 0;

}

void sparse_autoenc::insert_patch_noise(void)
{
    float noise = 0.0f;
    for(int k=0; k<patch_side_size*patch_side_size*dictionary.channels(); k++)
    {
        noise = (float) (rand() % 65535) / 65536;///0..1.0 range
        noise -= 0.5;
        noise *= init_noise_gain;
     //   noise += 0.5;

        *index_ptr_dict = noise;
        index_ptr_dict++;///Direct is fast but no sanity check. Must have control over this pointer otherwise Segmentation Fault could occur.
    }
}
float sparse_autoenc::get_noise(void)
{
    float noise=0.0f;
    noise = (float) (rand() % 65535) / 65536;///0..1.0 range
    noise -= 0.5;
    noise *= init_noise_gain;
    return noise;
}


void sparse_autoenc::check_dictionary_ptr_patch(void)
{
    sanity_check_ptr = zero_ptr_dict + (dictionary.rows * dictionary.cols * dictionary.channels());
    if(sanity_check_ptr != index_ptr_dict)
    {
        printf("ERROR! index_ptr_dict is NOT point at the same place as last pixel in dictionary Mat\n");
        printf("sanity_check_ptr =%p\n",sanity_check_ptr);
        printf("index_ptr_dict =%p\n",index_ptr_dict);
        exit(0);
    }
    else
    {
   ///     printf("OK sanity_check_ptr = index_ptr_dict =%p\n", index_ptr_dict);
    }
}
void sparse_autoenc::init(void)
{
    srand (static_cast <unsigned> (time(0)));///Seed the randomizer
    printf("\n");
    printf("*** Parameter settings of ***\n");
    printf("Layer_nr = %d\n", layer_nr);
    printf("*****************************\n");

    if(K_sparse == Lx_OUT_depth && use_greedy_enc_method == 1)
    {
       K_sparse = Lx_OUT_depth - 1;
       printf("WARNING K_sparse is subtracted by 1 to = %d\n", K_sparse);
       printf("because use_greedy_enc_method = 1 and K_sparse was set equal to Lx_OUT_depth \n");
    }
    if(layer_nr < 1 || layer_nr > 99)
    {
        printf("ERROR! layer_nr = %d is out of range 1..99\n", layer_nr);
        exit(0);
    }

    if(Lx_OUT_depth < 1)
    {
        printf("ERROR! Lx_OUT_depth = %d < 1\n", Lx_OUT_depth);
        exit(0);
    }
    if(Lx_OUT_depth > MAX_DEPTH)
    {
        printf("ERROR! to deep Lx_OUT_depth = %d > MAX_DEPTH = %d\n", Lx_OUT_depth, MAX_DEPTH);
        exit(0);
    }
    score_table = new int[Lx_OUT_depth];///Set up the size of the score_table will then contain the order of strength of each atom's. Only used in sparse mode. When K_sparse = Lx_OUT_depth this not used.
    train_hidden_node  = new float[Lx_OUT_depth];///
    temp_hidden_node = new float[Lx_OUT_depth];///Used in greedy method
    train_hidden_deleted_max = new float[Lx_OUT_depth];///
    convolution_mode = 0;

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
    if(color_mode == 1)///Only allowed at First Layer
    {
        if(init_in_from_outside == 1)
        {
            printf("ERROR! color_mode is ONLY allowed at first layer, init_in_from_outside = %d\n", init_in_from_outside);
            printf("Suggest on First layer: Set init_in_from_outside = 0\n");
            printf("Suggest on deeper layer: Set color_mode = 0\n");
            exit(0);
        }
        if(show_patch_during_run == 1)///Only for debugging)
        {
            eval_indata_patch.create(patch_side_size, patch_side_size, CV_32FC3);
            eval_atom_patch.create(patch_side_size, patch_side_size, CV_32FC3);
        }
        sqrt_nodes_plus1 = sqrt(Lx_OUT_depth);///Set up a square of nodes many small patches organized in a square
        sqrt_nodes_plus1 += 1;///Plus 1 ensure that the graphic square is large enough if the sqrt() operation get round of
        v_dict_hight = patch_side_size * sqrt_nodes_plus1;
        v_dict_width = patch_side_size * sqrt_nodes_plus1;
        dictionary.create(patch_side_size * Lx_OUT_depth, patch_side_size, CV_32FC3);///The first atom is one box patch_side_size X patch_side_size in COLOR. the second atom is in box below the the first atom then it fit Dot product better then the visual_dict layout
        change_w_dict.create(patch_side_size * Lx_OUT_depth, patch_side_size, CV_32FC3);
        change_w_dict = cv::Scalar(0.0f, 0.0f, 0.0f);
        visual_dict.create(v_dict_hight, v_dict_width, CV_32FC3);
        visual_activation.create(v_dict_hight, v_dict_width, CV_32FC3);///Show activation overlay marking on each patch.
        visual_dict = cv::Scalar(0.5f, 0.5f, 0.5f);
        visual_activation = cv::Scalar(0.5f, 0.5f, 0.5f);
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
        encoder_input.create     (patch_side_size * Lx_IN_depth, patch_side_size, CV_32FC3);
        denoised_residual_enc_input.create(patch_side_size * Lx_IN_depth, patch_side_size, CV_32FC3);
        reconstruct.create       (patch_side_size * Lx_IN_depth, patch_side_size, CV_32FC3);
        enc_error.create       (patch_side_size * Lx_IN_depth, patch_side_size, CV_32FC3);
        printf("This layer First Layer init_in_from_outside = %d\n", init_in_from_outside);
        Lx_IN_data_cube.create(Lx_IN_hight, Lx_IN_widht, CV_32FC3);
        printf("Lx_IN_data_cube are now created in COLOR mode CV_32FC3\n");
        bias_hid2out.create(patch_side_size, patch_side_size, CV_32FC3);
        visual_b_hid2out.create(patch_side_size, patch_side_size, CV_32FC3);
        change_w_b_hid2out.create(patch_side_size, patch_side_size, CV_32FC3);
        change_w_b_hid2out = cv::Scalar(0.0f, 0.0f, 0.0f);
        for(int i=0; i<patch_side_size; i++)
        {
            for(int j=0; j<patch_side_size*3; j++)
            {
                bias_hid2out.at<float>(i,j) = get_noise();
            }
        }
        printf("bias_hid2out are now created in COLOR mode CV_32FC3\n");
        bias_in2hid.create(Lx_OUT_depth, 2, CV_32FC1);///Column 0 = the weight. Column 1 = change weight
        visual_b_in2hid.create(Lx_OUT_depth, 2, CV_32FC1);///Column 0 = the weight. Column 1 = change weight

        for(int i=0; i<Lx_OUT_depth; i++)
        {
            bias_in2hid.at<float>(i, 0) = get_noise();
        }
    }
    else
    {
        ///Gray mode now a deep mode is used instead with number of deep layers
        ///This graphical setup consist of many small patches (boxes) with many (boxes) rows.
        v_dict_hight = patch_side_size * Lx_IN_depth;///Each patches (boxes) row correspond to each LxIN depth level.
        v_dict_width = patch_side_size * Lx_OUT_depth;///Each column of small patches (boxes) correspond to each encode node = each Lx OUT depth.
        dictionary.create(patch_side_size * Lx_IN_depth * Lx_OUT_depth, patch_side_size, CV_32FC1);///The first atom is one box patch_side_size X patch_side_size in COLOR. the second atom is in box below the the first atom then it fit Dot product better then the visual_dict layout
        change_w_dict.create(patch_side_size * Lx_IN_depth * Lx_OUT_depth, patch_side_size, CV_32FC1);
        change_w_dict = cv::Scalar(0.0f);
        visual_dict.create(v_dict_hight, v_dict_width, CV_32FC1);///Only gray
        visual_activation.create(v_dict_hight, v_dict_width, CV_32FC3);/// Color only for show activation overlay marking on the gray (green overlay)
        visual_dict = cv::Scalar(0.5f);
        visual_activation = cv::Scalar(0.5f, 0.5f, 0.5f);
        encoder_input.create     (patch_side_size * Lx_IN_depth, patch_side_size, CV_32FC1);
        denoised_residual_enc_input.create(patch_side_size * Lx_IN_depth, patch_side_size, CV_32FC1);
        reconstruct.create       (patch_side_size * Lx_IN_depth, patch_side_size, CV_32FC1);
        enc_error.create       (patch_side_size * Lx_IN_depth, patch_side_size, CV_32FC1);
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
            Lx_IN_data_cube.create(Lx_IN_hight * Lx_IN_depth, Lx_IN_widht, CV_32FC1);
            printf("Lx_IN_data_cube are now created in GRAY mode CV_32FC1\n");
        }
        bias_hid2out.create(patch_side_size  * Lx_IN_depth, patch_side_size, CV_32FC1);
        visual_b_hid2out.create(patch_side_size, patch_side_size, CV_32FC1);
        change_w_b_hid2out.create(patch_side_size  * Lx_IN_depth, patch_side_size, CV_32FC1);
        change_w_b_hid2out = cv::Scalar(0.0f);
        for(int h=0; h<Lx_IN_depth; h++)
        {
            for(int i=0; i<patch_side_size; i++)
            {
                for(int j=0; j<patch_side_size; j++)
                {
                    bias_hid2out.at<float>(h*patch_side_size + i,j) = get_noise();
                }
            }
        }
        printf("bias_hid2out are now created in GRAY mode CV_32FC1\n");
        bias_in2hid.create(Lx_OUT_depth, 2, CV_32FC1);///Column 0 = the weight. Column 1 = change weight
        visual_b_in2hid.create(Lx_OUT_depth, 2, CV_32FC1);///Column 0 = the weight. Column 1 = change weight
        for(int i=0; i<Lx_OUT_depth; i++)
        {
            bias_in2hid.at<float>(i, 0) = get_noise();
        }
        if(show_patch_during_run == 1)///Only for debugging)
        {
            eval_indata_patch.create(patch_side_size, patch_side_size, CV_32FC1);
            eval_atom_patch.create(patch_side_size, patch_side_size, CV_32FC1);
        }

    }
    printf("color_mode = %d\n", color_mode);
    printf("Lx_IN_depth = %d\n", Lx_IN_depth);
    printf("Lx_OUT_depth = %d\n", Lx_OUT_depth);
    printf("K_sparse = %d\n", K_sparse); ///This may changed during operation by the user control
    printf("stride = %d\n", stride);
    printf("Pixel Size of feature patch square side:\n");
    printf("patch_side_size = %d\n", patch_side_size);
    printf("dictionary.rows = %d\n", dictionary.rows);
    printf("dictionary.cols = %d\n", dictionary.cols);
    printf("dictionary.type() = %d\n", dictionary.type());
    printf("dictionary.channels() = %d\n", dictionary.channels());
    printf("visual_dict.rows = %d\n", visual_dict.rows);
    printf("visual_dict.cols = %d\n", visual_dict.cols);
    printf("Width of Lx_IN_data_cube, Lx_IN_widht = %d\n", Lx_IN_widht);
    printf("Hight of Lx_IN_data_cube, Lx_IN_hight = %d\n", Lx_IN_hight);

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
    max_patch_h_offset = (Lx_IN_hight - patch_side_size) / stride;
    max_patch_w_offset = (Lx_IN_widht - patch_side_size) / stride;
    printf("max_patch_h_offset = %d\n", max_patch_h_offset);
    printf("max_patch_w_offset = %d\n", max_patch_w_offset);

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
    ///======== Set up pointers for Mat direct address (fastest Mat access operation) =============
    zero_ptr_dict          = dictionary.ptr<float>(0);///Set up pointer for fast direct address of Mat
    index_ptr_dict         = zero_ptr_dict;///Set up pointer for fast direct address of Mat
    zero_ptr_vis_act       = visual_activation.ptr<float>(0);///Set up pointer for fast direct address of Mat
    index_ptr_vis_act      = zero_ptr_vis_act;///Set up pointer for fast direct address of Mat
    zero_ptr_Lx_IN_data    = Lx_IN_data_cube.ptr<float>(0);///Set up pointer for fast direct address of Mat
    index_ptr_Lx_IN_data   = zero_ptr_Lx_IN_data;///Set up pointer for fast direct address of Mat
    zero_ptr_Lx_OUT_conv   = Lx_OUT_convolution_cube.ptr<float>(0);///Set up pointer for fast direct address of Mat
    index_ptr_Lx_OUT_conv  = zero_ptr_Lx_OUT_conv;///Set up pointer for fast direct address of Mat
    zero_ptr_encoder_input  = encoder_input.ptr<float>(0);///Set up pointer for fast direct address of Mat
    index_ptr_encoder_input = zero_ptr_encoder_input;///Set up pointer for fast direct address of Mat
    zero_ptr_deno_residual_enc  = denoised_residual_enc_input.ptr<float>(0);///Set up pointer for fast direct address of Mat
    index_ptr_deno_residual_enc = zero_ptr_deno_residual_enc;///Set up pointer for fast direct address of Mat
    zero_ptr_reconstruct   = reconstruct.ptr<float>(0);///Set up pointer for fast direct address of Mat
    index_ptr_reconstruct  = zero_ptr_reconstruct;///Set up pointer for fast direct address of Mat
    zero_ptr_enc_error     = enc_error.ptr<float>(0);///
    index_ptr_enc_error    = zero_ptr_enc_error;
    zero_ptr_bias_hid2out  = bias_hid2out.ptr<float>(0);///
    index_ptr_bias_hid2out = zero_ptr_bias_hid2out;
    zero_ptr_chan_w_b_hid2out = change_w_b_hid2out.ptr<float>(0);///
    index_ptr_chan_w_b_hid2out= zero_ptr_chan_w_b_hid2out;
    zero_ptr_chan_w_dict   = change_w_dict.ptr<float>(0);///
    index_ptr_chan_w_dict  = zero_ptr_chan_w_dict;
    ///=================================================================================
    if(color_mode == 1)
    {
        ///COLOR mode the input depth is 1 with 3 COLOR
        index_ptr_dict = zero_ptr_dict;
        for(int i=0;i<Lx_OUT_depth;i++)
        {
            insert_patch_noise();
        }
        ///Now we could check the sanity of index_ptr_dict pointer address
        check_dictionary_ptr_patch();
    }
    else
    {
        ///GRAY mode the input depth is arbitrary
        index_ptr_dict = zero_ptr_dict;
        for(int i=0;i<Lx_OUT_depth;i++)
        {
            for(int j=0; j<Lx_IN_depth; j++)///IN depth is arbitrary in GRAY mode
            {
                insert_patch_noise();
            }
        }
        ///Now we could check the sanity of index_ptr_dict pointer address
        check_dictionary_ptr_patch();
    }
    printf("use_leak_relu = %d\n", use_leak_relu);
    if(use_leak_relu == 1)
    {
        printf("use_variable_leak_relu = %d\n", use_variable_leak_relu);
        if(use_variable_leak_relu == 1)
        {
            ///Check min_relu_leak_gain and relu_leak_gain_variation parameter
            if(min_relu_leak_gain > 0.0f && min_relu_leak_gain < 1.0f)
            {
                printf("min_relu_leak_gain = %f is in allowed range 0.0< to <1.0\n", min_relu_leak_gain);
            }
            else
            {
                printf("ERROR! parameter min_relu_leak_gain = %f is outside allowed range 0.0< to <1.0\n", min_relu_leak_gain);
                exit(0);
            }
            if(relu_leak_gain_variation > 0.0f && (relu_leak_gain_variation + min_relu_leak_gain) < 1.0f)
            {
                printf("relu_leak_gain_variation = %f is in allowed range 0.0< to ..\n", relu_leak_gain_variation);
                printf("relu_leak_gain_variation + min_relu_leak_gain = %f is in range <1.0\n", relu_leak_gain_variation + min_relu_leak_gain);
            }
            else
            {
                printf("ERROR! parameter relu_leak_gain_variation = %f  + min_relu_leak_gain is outside range 0.0< to <1.0\n", relu_leak_gain_variation);
                exit(0);
            }
            variable_relu_leak = min_relu_leak_gain;///Default start value
            actual_relu_leak_level = variable_relu_leak;
        }
        else
        {
            ///Check fix_relu_leak_gain parameter
            if(fix_relu_leak_gain > 0.0f && fix_relu_leak_gain < 1.0f)
            {
                printf("fix_relu_leak_gain = %f is in allowed range 0.0< to <1.0\n", fix_relu_leak_gain);
            }
            else
            {
                printf("ERROR! parameter fix_relu_leak_gain = %f is outside allowed range 0.0< to <1.0\n", fix_relu_leak_gain);
                exit(0);
            }
            actual_relu_leak_level = fix_relu_leak_gain;
            ///END Check fix_relu_leak_gain parameter
        }
        float min_score_b_level = -100000.0f;
        if(score_bottom_level > 0.0f || score_bottom_level < min_score_b_level)
        {
            printf("WARNING! score_bottom_level = %f should be in range 0.0f to %f\n", score_bottom_level, min_score_b_level);
        }
     }
    else
    {
        score_bottom_level = 0.0f;
        actual_relu_leak_level = 0.0f;
    }
    printf("Init CNN layer object Done!\n");
    printf("*** END settings of *********\n");
    printf("Layer_nr = %d\n", layer_nr);
    printf("*****************************\n");
    printf("\n");

}

void sparse_autoenc::k_sparse_sanity_check(void)
{
    ///============= K_sparse sanity check =============
    if(K_sparse == Lx_OUT_depth && use_greedy_enc_method == 1)
    {
       K_sparse = Lx_OUT_depth - 1;
       printf("WARNING K_sparse is subtracted by 1 to = %d\n", K_sparse);
       printf("because use_greedy_enc_method = 1 and K_sparse was set equal to Lx_OUT_depth \n");
    }

    if(sparse_autoenc::K_sparse < 1)
    {
        printf("ERROR! K_sparse = %d < 1\n", sparse_autoenc::K_sparse);
        exit(0);
    }
    if(sparse_autoenc::K_sparse > Lx_OUT_depth)
    {
        printf("ERROR! K_sparse = %d > Lx_OUT_depth\n", sparse_autoenc::K_sparse);
        exit(0);
    }
    if(K_sparse > MAX_DEPTH)
    {
        printf("ERROR! to deep K_sparse = %d > MAX_DEPTH = %d\n", K_sparse, MAX_DEPTH);
        exit(0);
    }
    ///============= End K_sparse sanity check =============
    //printf("K_sparse OK = %d\n", K_sparse);


}



#endif // SPARSE_AUTOENC_H
