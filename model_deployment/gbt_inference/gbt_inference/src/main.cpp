#include <chrono>
#include <iostream>
#include "xgb.hpp"
#include "m2cgen.hpp"


// https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c/22387757
class Timer {
public:
Timer() : t1_(std::chrono::high_resolution_clock::now()) {}

~Timer() {
    auto t2 = std::chrono::high_resolution_clock::now();
    auto timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1_);
    std::cout << "elapsed: " << timeSpan.count() << "\n";
}

private:
std::chrono::high_resolution_clock::time_point t1_;
};


void testMultiClassPrediction(const std::string& modelDir) {
    std::cout << "multi class prediction: " << std::endl;

    std::vector<std::string> features {
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    };
    std::vector<float> inputs {5.1, 3.5, 1.4, 0.2};
    int numClass = 3;
    std::string modelCheckpoint = modelDir + "/multi_class.txt";

    gbt::XGBoostPredictor gbtPredictor(modelCheckpoint, features, numClass);
    std::vector<float> prediction = gbtPredictor.predict(inputs);
    gbt::softmax(prediction);

    for (float prob: prediction) {
        std::cout << prob << " ";
    }
    std::cout << std::endl;
};


void testBinaryClassPrediction(const std::string& modelDir) {
    std::cout << "binary class prediction: " << std::endl;

    std::vector<std::string> features {"f0",  "f1",  "f2",  "f3",  "f4"};
    std::vector<float> inputs {0.0, 0.2, 0.4, 0.6, 0.8};
    int numClass = 1;
    std::string modelCheckpoint = modelDir + "/binary_class.txt";

    gbt::XGBoostPredictor gbtPredictor(modelCheckpoint, features, numClass);
    std::vector<float> prediction = gbtPredictor.predict(inputs);

    for (float score: prediction) {
        float prob = gbt::logistic(score);
        std::cout << prob << " ";
    }
    std::cout << std::endl;
};


void testRegressionPrediction(const std::string& modelDir) {
    std::cout << "regression prediction: " << std::endl;

    std::vector<std::string> features {"age", "sex", "bmi", "bp"};
    std::vector<float> inputs {0.038, 0.051, 0.062, 0.022};
    int numClass = 1;
    std::string modelCheckpoint = modelDir + "/regression.txt";

    gbt::XGBoostPredictor gbtPredictor(modelCheckpoint, features, numClass);
    std::vector<float> prediction = gbtPredictor.predict(inputs);

    for (float score: prediction) {
        std::cout << score << " ";
    }
    std::cout << std::endl;

    std::cout << "m2cgen regression prediction: " << std::endl;
    // m2cgen expects double instead of float
    std::vector<double> inputs2(inputs.begin(), inputs.end());
    double score = gbt::score(inputs2.data());
    std::cout << score << std::endl;

    // time the predicton using two different inferencing implementation
    size_t iterations = 1000;
    {
        Timer timer;
        for (size_t i = 0; i < iterations; i++) {
            prediction = gbtPredictor.predict(inputs);
        }
    }

    {
        Timer timer;
        for (size_t i = 0; i < iterations; i++) {
            score = gbt::score(inputs2.data());
        }
    }
};


int main(int, char**) {
    std::string modelDir = "/Users/mingyuliu/machine-learning/model_deployment/gbt_inference";
    testRegressionPrediction(modelDir);
    testBinaryClassPrediction(modelDir);
    testMultiClassPrediction(modelDir);
    return 0;
}
