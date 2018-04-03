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

