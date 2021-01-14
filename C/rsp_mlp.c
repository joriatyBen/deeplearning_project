#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "P:/mlp_test/Playground/final/nloh_json.h"

#define CUST_EST "C:/Users/bajorat_benjamin/Documents/codenv/estimator_cust_chkpts/custom_esti.json"
#define PRE_EST "C:/Users/bajorat_benjamin/Documents/codenv/estimator_chkpts/premade_esti.json"
#define PY_MLP "C:/Users/bajorat_benjamin/Documents/codenv/mlp_gen/datas.json"
#define TF_MNIST "C:/Users/bajorat_benjamin/Documents/codenv/mnist_examp/mnist.json"

enum preconfig {

	IN = 4,  //for MNIST image 28*28 = 784 pixels
	OUT = 3, //logits, classes ...
	H1 = 4,
	H2 = 4,
	BIAS = 0,
	N = 3,  //Number of Input sets
	flag = 4, //activation function
};


//defined flags for activation function
//flag=1 logistic activation function
//int flag = 1;

//initialization of MLP architecture
double b1[H1];		//array of bias of nodes of the 1st hidden layer
double b2[H2];		//array of bias of nodes of the 2nd hidden layer
double b3[OUT];		//array of bias of nodes of the ouput layer

//weight arrays
double w1[H1][IN];	//array of weights from input to  1st hidden layer
double w2[H2][H1];	//array of weights from 1st hidden to 2nd hidden layer
double w3[OUT][H2];	//array of weights from 2nd hidden to output

//outputs
double y[OUT]; //output of MLP
double z1[H1]; //output of 1st layer
double z2[H2]; //output of the 2nd layer

//products s=x*w
double s1[H1];
double s2[H2];
double s3[OUT];

//arrays for the training set
double x_testset[N][IN] = { { 5.9, 3.0, 4.2, 1.5 }, { 6.9, 3.1, 5.4, 2.1 }, { 5.1, 3.3, 1.7, 0.5 } };

//arrays for the MNIST set
//double x_mnist[TEST][IN];	//input

//FUNCTIONS

//extern double* getWeights_filepath(const char *filepath);

void initialize_bias_PYMLP(double *bweights)
{
	int i;
	double *pw;
	//bweights = getWeights_filepath("C:/Users/bajorat_benjamin/Documents/codenv/datas.json");
	pw = bweights;
	//nullptr check;
	if (pw == NULL) { EXIT_FAILURE; }

	int *pcount;
	int count = 0;
	pcount = &count;

	if (BIAS == 1)
	{
		//bias H1
		count = IN;
		for (i = 0; i < H1; i++)
		{
			b1[i] = *(pw + count);
			printf("Bias1: %f\n", b1[i]);
			//++*pcount;
			*pcount = *pcount + (IN+1);
		}

		//bias H2
		count = (H1 * IN + H1) + (H1);
		for (i = 0; i < H2; i++)
		{
			b2[i] = *(pw + count);
			printf("Bias2: %f\n", b2[i]);
			//++*pcount;
			*pcount = *pcount + (H1+1);
		}

		//bias Out
		count = (H1 * IN + H1) + (H2 * H1 + H2) + (H2);
		for (i = 0; i < OUT; i++)
		{
			b3[i] = *(pw + count);
			printf("Bias3: %f\n", b3[i]);
			//++*pcount;
			*pcount = *pcount + (H2+1);
		}
	}

	else
	{
		return;
	}
}

void initialize_PYMLPweights(double * weights)
{
	int i, j;
	double *pw;
	//weights = getWeights_filepath("C:/Users/bajorat_benjamin/Documents/codenv/datas.json");
	pw = weights;
	//nullptr check;
	if (pw == NULL) { EXIT_FAILURE; }
	int *pcount;
	int count = 0;
	pcount = &count;
	int skiph1 = 0, skiph2 = 0, skipout = 0;

	if (BIAS == 1)
	{
		for (i = 0; i < H1; i++)
		{
			for (j = 0; j < IN; j++)
			{	
				w1[i][j] = *(pw + count +skiph1);
				printf("weight1[%i][%i] IN -> H1: %f\n", i, j, w1[i][j]);
				++*pcount;
	
			}
			skiph1++;
		}

		count = count + H1;
		for (i = 0; i < H2; i++)
		{	
			for (j = 0; j < H1; j++)
			{
				w2[i][j] = *(pw + count + skiph2);
				printf("weight2[%i][%i] H1 -> H2: %f\n", i, j, w2[i][j]);
				++*pcount;
			}
			skiph2++;
		}

		count = count + H2;
		for (i = 0; i < OUT; i++)
		{
			for (j = 0; j < H2; j++)
			{
				w3[i][j] = *(pw + count + skipout);
				printf("weight3[%i][%i] H2 -> OUT: %f\n", i, j, w3[i][j]);
				++*pcount;
			}
			skipout++;
		}
	}
	else
	{
		for (i = 0; i < H1; i++)
		{
			for (j = 0; j < IN; j++)
			{
				w1[i][j] = *(pw + count);
				printf("weight1[%i][%i] IN -> H1: %f\n", i, j, w1[i][j]);
				++*pcount;
			}
		}
		for (i = 0; i < H2; i++)
		{
			for (j = 0; j < H1; j++)
			{
				w2[i][j] = *(pw + count);
				printf("weight2[%i][%i] H1 -> H2: %f\n", i, j, w2[i][j]);
				++*pcount;
			}
		}
		for (i = 0; i < OUT; i++)
		{
			for (j = 0; j < H2; j++)
			{
				w3[i][j] = *(pw + count);
				printf("weight3[%i][%i] H2 -> OUT: %f\n", i, j, w3[i][j]);
				++*pcount;
			}
		}
	}
}


//void initialize_bias_weight(double *bweights)
//{
//	int i;
//	double *pw;
//	//bweights = getWeights_filepath("C:/Users/bajorat_benjamin/Documents/codenv/datas.json");
//	pw = bweights;
//	//nullptr check;
//	if (pw == NULL) { EXIT_FAILURE; }
//
//	int *pcount;
//	int count = 0;
//	pcount = &count;
//
//	if (BIAS == 1)
//	{
//		//bias H1
//		count = H1 * IN;
//		for (i = 0; i < H1; i++)
//		{
//			b1[i] = *(pw + count);
//			printf("Bias1: %f\n", b1[i]);
//			++*pcount;
//		}
//
//		//bias H2
//		count = (H1 * IN + H1) + (H2 * H1);
//		for (i = 0; i < H2; i++)
//		{
//			b2[i] = *(pw + count);
//			printf("Bias2: %f\n", b2[i]);
//			++*pcount;
//		}
//
//		//bias Out
//		count = (H1 * IN + H1) + (H2 * H1 + H2) + (OUT * H2);
//		for (i = 0; i < OUT; i++)
//		{
//			b3[i] = *(pw + count);
//			printf("Bias3: %f\n", b3[i]);
//			++*pcount;
//		}
//	}
//
//	else
//	{
//		return;
//	}
//}

//void initialize_weights(double * weights)
//{
//	int i, j;
//	double *pw;
//	//weights = getWeights_filepath("C:/Users/bajorat_benjamin/Documents/codenv/datas.json");
//	pw = weights;
//	//nullptr check;
//	if (pw == NULL) { EXIT_FAILURE; }
//	int *pcount;
//	int count = 0;
//	pcount = &count;
//
//	if (BIAS == 1)
//	{
//		for (i = 0; i < H1; i++)
//		{
//			for (j = 0; j < IN; j++)
//			{
//				w1[i][j] = *(pw + count);
//				printf("weight1[%i][%i] IN -> H1: %f\n", i, j, w1[i][j]);
//				++*pcount;
//			}
//		}
//		count = count + H1;
//		for (i = 0; i < H2; i++)
//		{
//			for (j = 0; j < H1; j++)
//			{
//				w2[i][j] = *(pw + count);
//				printf("weight2[%i][%i] H1 -> H2: %f\n", i, j, w2[i][j]);
//				++*pcount;
//			}
//		}
//
//		count = count + H2;
//		for (i = 0; i < OUT; i++)
//		{
//			for (j = 0; j < H2; j++)
//			{
//				w3[i][j] = *(pw + count);
//				printf("weight3[%i][%i] H2 -> OUT: %f\n", i, j, w3[i][j]);
//				++*pcount;
//			}
//		}
//	}
//	else
//	{
//		for (i = 0; i < H1; i++)
//		{
//			for (j = 0; j < IN; j++)
//			{
//				w1[i][j] = *(pw + count);
//				printf("weight1[%i][%i] IN -> H1: %f\n", i, j, w1[i][j]);
//				++*pcount;
//			}
//		}
//		for (i = 0; i < H2; i++)
//		{
//			for (j = 0; j < H1; j++)
//			{
//				w2[i][j] = *(pw + count);
//				printf("weight2[%i][%i] H1 -> H2: %f\n", i, j, w2[i][j]);
//				++*pcount;
//			}
//		}
//		for (i = 0; i < OUT; i++)
//		{
//			for (j = 0; j < H2; j++)
//			{
//				w3[i][j] = *(pw + count);
//				printf("weight3[%i][%i] H2 -> OUT: %f\n", i, j, w3[i][j]);
//				++*pcount;
//			}
//		}
//	}
//}

double activation_function(double x, int f)
{
	//f = flag;
	//sigmoid, logistic u function	
	if (f == 1)
	{
		return (1 / (1 + exp(-x)));
	}
	//tanh
	if (f == 2)
	{
		return ((2.0 / (1.0 + exp(-2 * x))) - 1);
	}
	//relu
	if (f == 3)
	{
		//return log(1 + exp(x)); //softplus
		if (x <= 0.0)
		{
			return 0.0;
		}
		else
		{
			return x;
		}
	}
	//leaky relu
	if (f = 4)
	{
		//return log(1 + exp(x)); //softplus
		if (x <= 0.0)
		{
			return 0.01*x;
		}
		else
		{
			return x;
		}
	}
}

void forward_propagation(double * x) 
{
	int i, j;
	double si, sii, siii;
	//nullptr check;
	
	//double *px = &x_testset[0][0];
	//double *px = &test_image[0][0];
	
	double *px;
	px = x;
	//if (px == NULL) { EXIT_FAILURE; }

	//for (int z = 0; z < IN; z++)
	//{
	//	printf("%f\n", x[z]);
	//
	//}


	if (px == NULL) { EXIT_FAILURE; }

	for (i = 0; i < H1; i++)
	{
		si = 0;
		for (j = 0; j < IN; j++)
		{
			si = si + w1[i][j] * x[j];
			//printf("inputvariable %f\n", x[j]);
		}

		s1[i] = si + b1[i];
		z1[i] = activation_function(s1[i], flag);
		//printf("z1[%i]: %f\n", i, z1[i]);
	}
	for (i = 0; i < H2; i++)
	{
		sii = 0;
		for (j = 0; j < H1; j++)
		{
			sii = sii + w2[i][j] * z1[j];
		}
		s2[i] = sii + b2[i];
		z2[i] = activation_function(s2[i], flag);
		//printf("z2[%i]: %f\n",i, z2[i]);
	}
	for (i = 0; i < OUT; i++)
	{
		siii = 0;
		for (j = 0; j < H2; j++)
		{
			siii = siii + w3[i][j] * z2[j];
		}
		s3[i] = siii + b3[i];
		y[i] = activation_function(s3[i], flag);
		//printf("y[%i]: %f\n", i, y[i]);
	}
}



void output_softmax(double * y, int length)
{
	int i;
	double sum, max;

	for (i = 1, max = y[0]; i < length; i++) {
		if (y[i] > max) {
			max = y[i];
		}
	}

	for (i = 0, sum = 0; i < length; i++) {
		y[i] = exp(y[i] - max);
		sum += y[i];
	}

	for (i = 0; i < length; i++) {
		y[i] /= sum;
	}
}

int main() 
{
		
	int i, j, neurons;

	if (BIAS == 1) 
	{
		neurons = (H1*IN + H1) + (H2*H1 + H2) + (OUT*H2 + OUT);
	}
	else 
	{
		neurons = (H1*IN) + (H2*H1) + (OUT*H2);
	}


	//Choose filepath to JSON weight-file bei pointer
	double *py;
	double *pw;
	//ALTERNATIV JSON FILES
	//custom-estimator
	//pw = getWeights(CUST_EST);		
	//premade-estimator
	//pw = getWeights(PRE_EST);
	//python-MLP
	pw = getWeights(PY_MLP);
	//MNIST
	//pw = getWeights(TF_MNIST);


	//initialize json_parser
	//for (int i = 0; i < neurons; i++) {
	//	*(p + i);
	//	printf("hier gehts: %f\n", *(p + i));
	//}

	//initialize weights
	initialize_bias_PYMLP(pw);
	initialize_PYMLPweights(pw);
	
	//initialize_weights(pw);
	//initialize_bias_weight(pw);
	
	//fprop
	double max;
	int step;

	for (i = 0; i < N; i++) 
	{
		max = 0;
		step = 0;
		py = y;
		
		// choose for fprop x_mnist[i], x_testset[i]
		forward_propagation(x_testset[i]); //results y[0,1,2]
		//output_softmax(py, N);

		for (j = 0; j < OUT; j++) 
		{ 
			int count = j;	
			
			printf("Output: %f, number: %i\n", y[j], j);
			
			if (y[j] > max) 
			{
				max = y[j];
				step = j;
				//printf("max = %f, position = %i\n", max, step);
			}
			else {
				max;
				//printf("max = %f, position = %i\n", max, step);
			}
			if (count == (OUT-1)) 
			{
	
				if (step == 0) 
				{
					printf("Iris Sentosa -> 0\n");
				}
				if (step == 1) 
				{
					printf("Iris Versicolor -> 1\n");
				}
				if (step == 2) 
				{
					printf("Iris Virginica -> 2\n");
				}
			}
		}	
	}
}
