#include "nloh_json.h"
#include "P:/mlp_test/Dependencies/json/single_include/nlohmann/json.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <stdint.h>

using namespace std;
using json = nlohmann::json;

static std::vector<double> weight_vec;
static double* ret_weight_vec;


double* getWeights(const char *filepath) {
	using std::copy;
	
	std::ifstream f(filepath);
	json j = json::parse(f);  

	for (auto& x : j.items()) {
		//		std::cout << "key: " << x.key() << ", value: " << x.value() << '\n';
		weight_vec.push_back(x.value());
	}

	/*for (std::vector<double>::const_iterator i = weight_vec.begin();
		i != weight_vec.end(); ++i)
		std::cout << *i << ' ';*/


	ret_weight_vec = &weight_vec[0];
	return ret_weight_vec;
}

//int main() {
//
//	getWeights("C:/Users/bajorat_benjamin/Documents/codenv/mnist_examp/esti_mnist.json");
//}
//;



