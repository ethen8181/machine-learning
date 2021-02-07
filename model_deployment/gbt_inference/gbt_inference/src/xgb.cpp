#include <cmath>
#include <fstream>
#include "xgb.hpp"

namespace gbt {

XGBoostPredictor::XGBoostPredictor(
    const std::string& modelCheckpoint,
    const std::vector<std::string>& features,
    int numClass
) {
    numClass_ = numClass;
    
    // e.g. 0:[f0<-0.0629209131] yes=1,no=2,missing=1
    nodeRegexExpression_ = "([0-9]+):\\[(.*)<(.*)\\] yes=([0-9]+),no=([0-9]+),.*";
    // e.g. 7:leaf=-0.556454778
    leafRegexExpression_ = "([0-9]+):leaf=(.*)";

    int numFeatures = 0;
    for (std::string feature: features) {
        featureIndices_[feature] = numFeatures;
        numFeatures++;
    }

    std::string line;
    std::ifstream in(modelCheckpoint);
    std::unordered_map<int, TreeNode> tree;
    while (std::getline(in, line)) {
        parseLine(line, tree);
    }
    addTree(tree);
    in.close();
};

void XGBoostPredictor::parseLine(
    const std::string& line,
    std::unordered_map<int, TreeNode>& tree
) {
    if (line.rfind("booster", 0) == 0) {
        addTree(tree);
    } else {
        TreeNode node;
        int currentNode;

        std::smatch nodeRegexMatch;
        bool matched = std::regex_search(line, nodeRegexMatch, nodeRegexExpression_);
        if (matched) {
            currentNode = std::stoi(nodeRegexMatch[1]);
            // yes goes to left, no goes to right
            node.left_ = std::stoi(nodeRegexMatch[4]);
            node.right_ = std::stoi(nodeRegexMatch[5]);
            node.splitIndex_ = featureIndices_[nodeRegexMatch[2]];
            node.splitValue_ = std::stof(nodeRegexMatch[3]);
        } else {
            std::smatch leafRegexMatch;
            std::regex_search(line, leafRegexMatch, leafRegexExpression_);

            currentNode = std::stoi(leafRegexMatch[1]);
            node.response_ = std::stof(leafRegexMatch[2]);
            node.isLeaf_ = true;
        }
        tree[currentNode] = node;
    }
};

void XGBoostPredictor::addTree(std::unordered_map<int, TreeNode>& tree) {
    if (!tree.empty()) {
        trees_.push_back(tree);
        tree.clear();
    }
};

std::vector<float> XGBoostPredictor::predict(const std::vector<float>& inputs) {
    int index;
    TreeNode node;
    std::vector<float> prediction(numClass_, 0.0f);
    int numTree = 0;

    for (std::unordered_map<int, TreeNode>& tree: trees_) {
        index = 0;
        do {
            node = tree[index];
            index = inputs[node.splitIndex_] > node.splitValue_ ? node.right_ : node.left_;
        } while (!node.isLeaf_);

        prediction[numTree % numClass_] += node.response_;
        numTree++;
    }
    return prediction;
};


float logistic(float score) {
    return 1. / (1. + std::exp(-score));
};


void softmax(std::vector<float>& prediction) {
    float norm = 0.;

    for (int i = 0; i < prediction.size(); ++i) {
        float& score = prediction[i];
        score = std::exp(score);
        norm += score;
    }

    for (int i = 0; i < prediction.size(); ++i) {
        float& score = prediction[i];
        score /= norm;
    }
};

};  // namespace gbt