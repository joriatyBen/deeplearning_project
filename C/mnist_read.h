#ifndef MNIST_READ_H_
#define MNIST_READ_H_

#define TRAIN_IMAGE "P:/mlp_test/Playground/mnist_c_test/data/train-images.idx3-ubyte"
#define TRAIN_LABEL "P:/mlp_test/Playground/mnist_c_test/data/train-labels.idx1-ubyte"
#define TEST_IMAGE "P:/mlp_test/Playground/mnist_c_test/data/t10k-images.idx3-ubyte"
#define TEST_LABEL "P:/mlp_test/Playground/mnist_c_test/data/t10k-labels.idx1-ubyte"

#define SIZE 784 // 28*28 pixels	
#define NUM_TRAIN 60000
#define NUM_TEST 10000

double train_image[NUM_TRAIN][SIZE];
double test_image[NUM_TEST][SIZE];
double  train_label[NUM_TRAIN];
double test_label[NUM_TEST];


#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

	double* get_mnist_image(const char *filepath);
	double* get_mnist_label(const char *filepath);
		
#ifdef __cplusplus
}
#endif

#endif 