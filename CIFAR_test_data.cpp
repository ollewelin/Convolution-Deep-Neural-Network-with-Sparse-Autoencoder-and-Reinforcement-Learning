#include "CIFAR_test_data.h"
#include <stdio.h>
#include <stdlib.h>// exit(0);

CIFAR_test_data::CIFAR_test_data()
{
    //ctor
    printf("CIFAR_test_data Constructor\n");
    CIFAR_height = 32;
    CIFAR_width  = 32;
    CIFAR_nr_of_img_p_batch = 10000;
    CIFAR_RGB_pixels = CIFAR_height*CIFAR_width;
    CIFAR_row_size = 3073;
}

CIFAR_test_data::~CIFAR_test_data()
{
    //dtor
}

void CIFAR_test_data::init_CIFAR(void)
{
    printf("**** Start read data size of CIFAR data set file data_batch_1.bin  **** \n");
    FILE *fp1;
    fp1= fopen("data_batch_1.bin", "r");
    if (fp1 == NULL)
    {
        puts("Error while opening file data_batch_1.bin");
        exit(0);
    }
    fseek(fp1, 0L, SEEK_END);
    nr_of_CIFAR_file_bytes = ftell(fp1);
    rewind(fp1);
    printf("Byte size of data_batch_1.bin = %d\n", nr_of_CIFAR_file_bytes);
    CIFAR_data = new char[nr_of_CIFAR_file_bytes];
    int MN_index=0;
    char c_data=0;
    for(int i=0; i<nr_of_CIFAR_file_bytes; i++)
    {
        c_data = fgetc(fp1);
        if( feof(fp1) )
        {
            break;
        }
        //printf("c_data %d\n", c_data);
        CIFAR_data[MN_index] = c_data;
        MN_index++;
    }
    fclose(fp1);
    printf("data_batch_1.bin data is put into CIFAR_data[0..%d]\n", nr_of_CIFAR_file_bytes-1);
}

void CIFAR_test_data::insert_a_random_CIFAR_image(void)
{
    /// Sanity check of the insert_L1_IN_data_cube
    if(insert_L1_IN_data_cube.type() != 21)
    {
        printf("ERROR! Mat type is wrong insert_L1_IN_data_cube.type() = %d is not = CV_32FC3 = 21\n", insert_L1_IN_data_cube.type());
        exit(0);
    }
    if(insert_L1_IN_data_cube.cols != CIFAR_width)
    {
        printf("ERROR! insert_L1_IN_data_cube.cols = %d != CIFAR_width\n", insert_L1_IN_data_cube.cols);
        exit(0);
    }

    if(insert_L1_IN_data_cube.rows != CIFAR_height)
    {
        printf("ERROR! insert_L1_IN_data_cube.rows = %d != CIFAR_height\n", insert_L1_IN_data_cube.rows);
        exit(0);
    }
    ///End Mat Sanity check

    ///==============================================
    ///Test insert COLOR data to the L1_IN_convolution_cube
    ///   example_mat.at<float>(ROW, COLUMN) = 1.0f;
///    insert_L1_IN_data_cube = cv::Scalar(0,0,0);
///    insert_L1_IN_data_cube.at<float>(5, 0) = 1.0;/// 0 column = Blue
///    insert_L1_IN_data_cube.at<float>(6, 1) = 1.0;/// 1 column = Green
///    insert_L1_IN_data_cube.at<float>(7, 2) = 1.0;/// 2 column = Red
    ///==============================================
    CIFAR_nr = (int) (rand() % CIFAR_nr_of_img_p_batch);
    CIFAR_nr = 0;
    printf("Load CIFAR_nr = %d to insert_L1_IN_data_cube\n", CIFAR_nr);

    float pixel_data=0.0f;
    uint8_t pixel_8d=0;

    for (int i=0; i<(CIFAR_height*CIFAR_width); i++)
    {
        for(int BGR=0; BGR<3; BGR++)
        {
            pixel_8d = CIFAR_data[(CIFAR_nr*CIFAR_row_size) + ((2-BGR)*CIFAR_RGB_pixels) + i +1];///(2-BGR) will swap Blue and Red order
            pixel_data = (float) pixel_8d;///(2-BGR) will swap Blue and Red order
            pixel_data = pixel_data * (1.0/255.0f);
            ///   example_mat.at<float>(ROW, COLUMN) = 1.0f;
            insert_L1_IN_data_cube.at<float>(i/(CIFAR_height*CIFAR_width), (i%(CIFAR_height*CIFAR_width)*3) + BGR) = pixel_data;
        }
    }

}
