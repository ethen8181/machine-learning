#pragma once

#include <regex>
#include <vector>
#include <string>
#include <unordered_map>

namespace gbt {

struct TreeNode {
    int left_ = -1;
    int right_ = -1;
    bool isLeaf_ = false;
    float response_ = 0.0f;
    int splitIndex_ = -1;
    float splitValue_ = 0.0f;
};

class XGBoostPredictor {
public:

/*
 * Initialize the model given the .txt model checkpoint (i.e. from dump_model),
 * the input features, and the number of classes
 */
XGBoostPredictor(
    const std::string& modelCheckpoint,
    const std::vector<std::string>& features,
    int numClass
);

/*
 * Parse each line of the XGBoost text model dump.
 * Each line can either indicate a new tree, a split node or a leaf node.
 */
void parseLine(const std::string& line, std::unordered_map<int, TreeNode>& tree);

/*
 * After reaching the end of a given tree, we add all the nodes of the tree
 * before moving on to processing the next tree.
 */
void addTree(std::unordered_map<int, TreeNode>& tree);

/*
 * Given the input features, generate the output raw prediction.
 * Note that it is a raw prediction, for binary or multi-class classification,
 * we will need to apply additional transformation on top of the raw prediction
 * to get the actual probabilities
 */
std::vector<float> predict(const std::vector<float>& inputs);

private:
// number of class to determine the output vector size
int numClass_;

// gbt is an ensemble tree model, stores all the trees in the model
// where each tree is stored as a node index to TreeNode map
std::vector<std::unordered_map<int, TreeNode>> trees_;

// maps the feature name to feature index
std::unordered_map<std::string, int> featureIndices_;

// regex to parse the leaf node
std::regex leafRegexExpression_;

// regex to parse the split node
std::regex nodeRegexExpression_;
};

/*
 * Applies the logistic transformation for a given score
 */
float logistic(float score);

/*
 * Applies the softmax transformation in place
 */
void softmax(std::vector<float>& prediction);

};  // namespace gbt