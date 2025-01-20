#include "ANN.h"
#include "Utils.h"

Double Actiavtions::LeakyReLU(const Double& d)
{
	if ((double)d < 0)
	{
		return d * 0.1;
	}

	return d;
}

Double Actiavtions::Linear(const Double & d)
{
    return d;
}

Double Actiavtions::ReLU(const Double& d)
{
	if ((double)d < 0)
	{
		return 0;
	}

	return d;
}

Double Actiavtions::Tanh(const Double & d)
{
    return d.get_tanh();
}

Neuron::Neuron() : Neuron(0, nullptr)
{
}

Neuron::Neuron(const unsigned int& numberOfInputs, Double (*activation)(const Double&))
: mNumberOfInputs(numberOfInputs), mActivation(activation)
{
	if (mNumberOfInputs > 0)
	{
		mBias = getRandomBetween(-0.1, 0.1);
		mWeights = shared_ptr<Double[]>(new Double[mNumberOfInputs]);

		for (int i = 0; i < mNumberOfInputs; i++)
		{
			mWeights[i] = getRandomBetween(-0.25, 0.25);
		}
	}
	else
	{
		mBias = 0;
		mWeights = nullptr;
	}
}

const Double Neuron::forward(vector<Double> inputs) const
{
	Double result;

	for (int i = 0; i < mNumberOfInputs; i++)
	{
		if (i == 0)
		{
            result = inputs.at(i) * mWeights[i];
		}
		else
		{
            result = result + inputs.at(i) * mWeights[i];
		}
	}

    return mActivation(result + mBias);
}

void Neuron::update(const double& factor)
{
	for (int i = 0; i < mNumberOfInputs; i++)
	{
		mWeights[i] = (double)mWeights[i] - (mWeights[i].get_grad() * factor);
	}

	mBias = (double)mBias - (mBias.get_grad() * factor);
}

Layer::Layer() : Layer(0, 0, nullptr)
{
}

Layer::Layer(const unsigned int& numberOfInputs, const unsigned int& numberOfOutputs, Double (*activation)(const Double&))
	: mNumberOfInputs(numberOfInputs), mNumberOfOutputs(numberOfOutputs)
{
	if (mNumberOfInputs > 0 && mNumberOfOutputs > 0)
	{
		mNeurons = shared_ptr<Neuron[]>(new Neuron[mNumberOfOutputs]);

		for (int i = 0; i < mNumberOfOutputs; i++)
		{
            mNeurons[i] = Neuron(mNumberOfInputs, activation);
		}
	}
	else
	{
		mNeurons = nullptr;
	}
}

vector<Double> Layer::forward(vector<Double> inputs)
{
    vector<Double> outputs;
	for (int i = 0; i < mNumberOfOutputs; i++)
	{
        outputs.push_back(mNeurons[i].forward(inputs));
	}

	return outputs;
}

void Layer::update(const double& factor)
{
	for (int i = 0; i < mNumberOfOutputs; i++)
	{
		mNeurons[i].update(factor);
	}
}

MLP::MLP(const unsigned int& numberOfInputs, const unsigned int& numberOfLayers, const unsigned int* numberOfOutputs, Double (*hidden_activation)(const Double&), Double (*output_activation)(const Double&))
{
	mNumberOfInputs = numberOfInputs;
	mNumberOfLayers = numberOfLayers;

	mLayers = shared_ptr<Layer[]>(new Layer[mNumberOfLayers]);

	for (int i = 0; i < mNumberOfLayers; i++)
	{
		unsigned int inputCount = i == 0 ? mNumberOfInputs : numberOfOutputs[i - 1];
        mLayers[i] = Layer(inputCount, numberOfOutputs[i], i == numberOfLayers - 1 ? output_activation : hidden_activation);
	}
}

vector<Double> MLP::forward(vector<Double> inputs)
{
    vector<Double> outputs;
	for (int i = 0; i < mNumberOfLayers; i++)
	{
		if (i == 0)
		{
			outputs = mLayers[i].forward(inputs);
		}
		else
		{
			outputs = mLayers[i].forward(outputs);
		}
	}

	return outputs;
}

void MLP::update(const double& factor)
{
	for (int i = 0; i < mNumberOfLayers; i++)
	{
		mLayers[i].update(factor);
	}
}
