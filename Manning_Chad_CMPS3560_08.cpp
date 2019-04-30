/**
 * @author: Chad Manning
 * Course: CMPS 3560
 *
 * Multilayer nueral network
*/
#include <iostream>
#include "neuron.h"
#include <fstream>
#include <sstream>
#include <cmath>
using namespace std;

/*
*	Function that reads data from a CSV into a 2-D vector.
*/
void readCSV(vector<vector<double>> &db,
             vector<vector<double>> &dDb, string filename)
{
    ifstream file(filename);

    string line, val;
    int i = -1, numFeatures = 0;
    while (getline(file, line))
    {
        numFeatures = 0;
        stringstream iss(line);
        db.push_back(vector<double>());
        dDb.push_back(vector<double>());
        ++i;
        while (getline(iss, val, ','))
        {
            stringstream convertor(val);
            double num = 0.0;
            convertor >> num;
            if (numFeatures++ > 3)
                dDb[i].push_back(num);
            else
                db[i].push_back(num);
        }
    }
}

void generateneuronLayer(vector<neuron> &layer, int num, int features)
{
    srand(time(NULL));
    for (int i = 0; i < num; i++)
    {
        layer.push_back(neuron(features, rand()));
    }
}

double sigma(double sum)
{
    double result = 1 + exp(sum * -1);
    return 1 / result;
}

int main(int argc, char *argv[])
{
    vector<vector<double>> featuresArr, desiredArr;
    vector<neuron> hiddenLayer;
    vector<neuron> outputLayer;
    double manhattan = 0.0;
    double errG = 0.0;
    double mad = 1.0;
    double output[3] = {0.0, 0.0, 0.0};

    readCSV(featuresArr, desiredArr, argv[1]);

    generateneuronLayer(hiddenLayer, 3, 4);
    generateneuronLayer(outputLayer, 3, 3);

    int epoch = 0;
    while (!(mad - 0.1 < 0.01))
    {
        manhattan = 0.0;
        for (unsigned int i = 0; i < featuresArr.size(); i++)
        {
            errG = 0.0;
            for (unsigned int j = 0; j < featuresArr[i].size(); j++)
            {
                for (unsigned int k = 0; k < hiddenLayer.size(); k++)
                {
                    hiddenLayer[j].receiveInput(featuresArr[i][j]);
                }
            }

            for (unsigned int j = 0; j < hiddenLayer.size(); j++)
            {
                for (unsigned int k = 0; k < outputLayer.size(); k++)
                {
                    outputLayer[k].receiveInput(sigma(hiddenLayer[j].activate()));
                }
            }

            for (unsigned int j = 0; j < 3; j++)
            {
                output[j] = sigma(outputLayer[j].activate());
                manhattan += abs(desiredArr[i][j] - output[j]);
            }

            //Comment out area below to see Epoch details only
            // ostringstream prediction;
            // cout << "Epoch " << epoch << ", Iteration " << i << " ";
            // cout << "Prediction is [";
            // for (unsigned int j = 0; j < 3; j++) {
            //     if (j < 2)
            //         cout << to_string(output[j]) + ", ";
            //     else
            //         cout << to_string(output[j]);
            // }
            // cout << "].\n";
            //

            for (unsigned int j = 0; j < 3; j++)
            {
                errG = output[j] * (1 - output[j]) * (desiredArr[i][j] - output[j]);
                outputLayer[j].updateWeights(errG);
            }

            errG = 0.0;
            for (unsigned int j = 0; j < 3; j++)
            {
                errG += output[j] * (1 - output[j]) * (desiredArr[i][j] - output[j]);
            }
            for (unsigned int j = 0; j < 3; j++)
            {
                for (unsigned int k = 0; k < 3; ++k)
                {
                    hiddenLayer[j].updateWeights(hiddenLayer[j].getOutput(k) * (1 - hiddenLayer[j].getOutput(k)) * errG * hiddenLayer[j].getWeight(k));
                }
            }
        }
        //Use below to view even faster results
        // if (manhattan / 150 - mad < 0.01)
        // {
        mad = manhattan / 150;
        std::printf("Epoch %d Results:  MAD = %2.3f\n\n", ++epoch, mad);
        // }
    }

    return 0;
}