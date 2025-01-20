#include <stdio.h>

#include "ANN.h"
#include "MicroGrad.h"
#include "TrainingTest.h"
#include "Utils.h"

using namespace std;

const double MAX_VALUE = 2;
const double MIN_STEP_SIZE = 0.001;
const unsigned int TOTAL_ITR = 15000;

double get_desired(const double&);
vector<Double> prepare_inputs(const unsigned int&);
vector<Double> prepare_outputs(const vector<Double>&);
void print(const Double&, const unsigned int&);

void start_training()
{
	initUtils();

	const unsigned int batchCount = 20;
	const unsigned int inputCount = 1;
	const unsigned int numberOfLayers = 3;
	const unsigned int finalOutCount = 1;
	unsigned int layerOutputs[numberOfLayers] = { 4, 4, finalOutCount };

    vector<Double> inputs[batchCount];
	vector<Double> desiredOutputs[batchCount];

	for (int i = 0; i < batchCount; i++)
	{
		inputs[i] = prepare_inputs(inputCount);
		desiredOutputs[i] = prepare_outputs(inputs[i]);

		for (int j = 0; j < inputs[i].size(); j++)
		{
			printf("Input: %lf, output: %lf\n", (double)inputs[i].at(j), (double)desiredOutputs[i].at(j));
		}
	}

	getchar();

    MLP mlp(inputCount, numberOfLayers, layerOutputs, Actiavtions::LeakyReLU, Actiavtions::LeakyReLU);
	
	char ch = '\0';
	for (int itr = 0; itr < TOTAL_ITR || ch == 'a'; itr++)
	{
		Double totalBatchLoss;
		for (int i = 0; i < batchCount; i++)
		{
			vector<Double> outs = mlp.forward(inputs[i]);
			Double totalLossPerBatch;
			for (int j = 0; j < outs.size(); j++)
			{
				Double loss = (outs.at(j) - desiredOutputs[i][j]).get_pow(2.0);
				if (j == 0)
				{
					totalLossPerBatch = loss;
				}
				else

				{
					totalLossPerBatch = totalLossPerBatch + loss;
				}
			}

			totalLossPerBatch = totalLossPerBatch / (double)outs.size();

			if (i == 0)
			{
				totalBatchLoss = totalLossPerBatch;
			}
			else
			{
				totalBatchLoss = totalBatchLoss + totalLossPerBatch;
			}
		}

		totalBatchLoss = totalBatchLoss / batchCount;
		printf("Itr: %d, Loss: %.10lf\n", itr, (double)totalBatchLoss);


		totalBatchLoss.zero_grad();
		backpropagate(totalBatchLoss);
		mlp.update(MIN_STEP_SIZE);
	}

	ch = '\0';

	while (ch != 'a') {
		double v = 0;
		printf("Enter a number: \n");
		scanf("%lf", &v);
		printf("\n");
		vector<Double> input = { v };
		vector<Double> out = mlp.forward(input);
		printf("Out = %lf\n", (double)out.at(0));
	}
}

double get_desired(const double& x)
{
	return x * x;
}


vector<Double> prepare_inputs(const unsigned int& inputCount)
{
	vector<Double> inputs;
	for (unsigned int i = 0; i < inputCount; i++)
	{
		inputs.push_back(getRandomBetween(-MAX_VALUE, MAX_VALUE));
	}

	return inputs;
}

vector<Double> prepare_outputs(const vector<Double>& inputs)
{
	vector<Double> outputs;
	for (unsigned int i = 0; i < inputs.size(); i++)
	{
		outputs.push_back(get_desired((double)inputs.at(i)));
	}

	return outputs;
}

void print(const Double& data, const unsigned int& batchCount)
{
    printf("Value = %f, Grad = %f\n", (double)data, data.get_grad());
}
