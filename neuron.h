#include <vector>
#include <string>
using namespace std;
class neuron
{
    public:
        neuron(int numFeatures, int randSeed);
        void setFeatures(vector<double> featureArr);
        void receiveInput(double feature);
        void updateWeights(double errG);
        double generateDouble();
        double activate();
        double getOutput(int index);
        double getWeight(int index);
    private:
        void generateWeights(int numFeatures);
        double bias;
        vector<double> weights;
        vector<double> input;
        vector<double> output;
};