#ifndef ANN_H
#define ANN_H

#include <memory>
#include <vector>

#include "MicroGrad.h"

using namespace std;

class Actiavtions {
public:
	static Double LeakyReLU(const Double&);
    static Double Linear(const Double&);
    static Double ReLU(const Double&);
    static Double Tanh(const Double&);
};

class Neuron
{
private:
    Double (*mActivation)(const Double&);
	unsigned int mNumberOfInputs;
	shared_ptr<Double[]> mWeights;
	Double mBias;

public:
	Neuron();
	Neuron(const unsigned int&, Double (*activation)(const Double&));

    const Double forward(vector<Double>) const;
	void update(const double&);
};

class Layer
{
public:
	Layer();
	Layer(const unsigned int&, const unsigned int&, Double (*activation)(const Double&));

    vector<Double> forward(vector<Double> inputs);
	void update(const double&);

private:
	unsigned int mNumberOfInputs;
	unsigned int mNumberOfOutputs;
	shared_ptr<Neuron[]> mNeurons;
};

class MLP
{
public:
    MLP(const unsigned int&, const unsigned int&, const unsigned int*, Double (*hidden_activation)(const Double&), Double (*output_activation)(const Double&));

    vector<Double> forward(vector<Double> inputs);
	void update(const double&);

private:
	shared_ptr<Layer[]> mLayers;
	unsigned int mNumberOfInputs;
	unsigned int mNumberOfLayers;
};

#endif // !ANN_H
