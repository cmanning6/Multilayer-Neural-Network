#include "neuron.h"
#include <math.h>
using namespace std;

/**
 * Constructs a neuron
 * @param numFeature : number of expected features for neuron
 */
neuron::neuron(int numWeights, int randSeed)
{
    srand(time(NULL));

    generateWeights(numWeights);
    bias = generateDouble();
}

/**
 * Returns Double between -0.5 and 0.5
*/
double neuron::generateDouble()
{
    double val = 0.0;
    val = (((double)rand()) / RAND_MAX) - 0.5;
    return floor(val * 100) / 100;
}

/**
 * Generates weights for the perceptron
*/
void neuron::generateWeights(int numFeatures)
{
    for (int i = 0; i < numFeatures; i++)
    {
        weights.push_back(generateDouble());
    }
}

void neuron::receiveInput(double feature)
{
    if (input.size() > 2)
    {
        output.clear();
        input.clear();
    }
    else
    {
        input.push_back(feature);
    }
}

void neuron::updateWeights(double errG)
{
    double delta = 0.0;
    for (unsigned int i = 0; i < input.size(); ++i)
    {
        delta = 0.01 * input[i] * errG;
        weights[i] += delta;
    }
    bias += 0.01 * -1 * errG;
    input.clear();
    return;
}

double neuron::activate()
{
    double prediction = 0.0;
    for (unsigned int i = 0; i < input.size(); i++)
    {
        prediction += (input.at(i) * weights.at(i));
    }
    prediction -= bias;
    output.push_back(prediction);
    return prediction;
}

double neuron::getOutput(int index)
{
    return output[index];
}

double neuron::getWeight(int index)
{
    return weights[index];
}
