#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include "P:/mlp_test/Playground/final/mnist_read.h"

using namespace std;

// reduces memory claim
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(const char *filepath, vector<vector<double> > &vec_1)
{
	std::ifstream file(filepath, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		//std::cout << "magic: " << magic_number << endl; //for train_set magic_number = 60000
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			vector<double> tp;
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					//tp.push_back((double)temp) ; //instead of if/else prints [0, 255]
					
					if ((double)temp > 0.5) 
					{
						tp.push_back((double)1.0);
					}
					else 
					{
						tp.push_back((double)0.0); // recscale to 0 xor 1
						//tp.push_back(((double)temp)*(1.0 / 255.0)); //rescale into [0, 1]
					}
					
				}
			}
			vec_1.push_back(tp);
		}
	}
}


double* read_Mnist_Label(const char *filepath, vector<double> vec_2)
{
	std::ifstream file(filepath, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec_2[i] = (double)temp;
			test_label[i] = vec_2[i];
			//std::cout << "Label: " << vec_2[i] << endl; //console output
		}
	}
	return test_label;
}

 
//wrap stuff for use in C
double* get_mnist_image(const char *filepath) 
{	
	vector<vector<double> > vec_1;
	read_Mnist(filepath, vec_1);
	/*vec_2_arr(vec_1, NUM_TEST, SIZE);
	return test_image[0];*/
	for (int x = 0; x < NUM_TEST; x++)
	{
		for (int y = 0; y < SIZE; y++)
		{
			test_image[x][y] = vec_1[x][y];
		}
	}


}

double* get_mnist_label(const char *filepath)
{
	vector<double> vec_2(NUM_TEST);
	read_Mnist_Label(filepath, vec_2);
 

	return test_label;

}



//int main()
//{
//	//final functions:
//	get_mnist_image(TEST_IMAGE);
//	get_mnist_label(TEST_LABEL);
//
//	for (int x = 0; x < NUM_TEST; x++)
//	{
//		cout << "label: " << test_label[x] << ' ' << endl;
//		for (int y = 0; y < SIZE; y++)
//		{	
//			cout << test_image[x][y] << ' ';
//			if ((y + 1) % 28 == 0)
//			{
//				putchar('\n');
//			}
//
//		}
//	}
//	
//	return 0;
//}