#include <queue>

#include "MicroGrad.h"

using namespace std;

Double::Double()
	: Double(0, 0, nullptr, nullptr)
{ }

Double::Double(const double& value)
	: Double(value, 0, nullptr, nullptr)
{ }

Double::Double(const double& value, const unsigned int& parentsCount, Double* parents, void (*bp)(const Double&, const double&))
	: mBackPropagate(bp), mGrad(new double(1)), mParentsCount(parentsCount), mParents(parents), mValue(new double(value))
{
	if (mParentsCount > 0)
	{
		for (int i = 0; i < mParentsCount; i++)
		{
			*mParents.get()[i].mGrad = 0;
		}
	}
}

void Double::BackPropagateAddition(const Double& d, const double& grad)
{
	if (d.mParentsCount != 2)
	{
		return;
	}

	*d.mParents.get()[0].mGrad += 1 * grad;
	*d.mParents.get()[1].mGrad += 1 * grad;
}

void Double::BackPropagateDivision(const Double& d, const double& grad)
{
	if (d.mParentsCount != 2)
	{
		return;
	}

	*d.mParents.get()[0].mGrad += (1.0 / (double)d.mParents.get()[1]) * grad;
	*d.mParents.get()[1].mGrad += (1.0 / (double)d.mParents.get()[0]) * grad;
}

void Double::BackPropagateMultiplication(const Double& d, const double& grad)
{
	if (d.mParentsCount != 2)
	{
		return;
	}

	*d.mParents.get()[0].mGrad += (double)d.mParents.get()[1] * grad;
	*d.mParents.get()[1].mGrad += (double)d.mParents.get()[0] * grad;
}

void Double::BackPropagatePower(const Double& d, const double& grad)
{
	if (d.mParentsCount != 2)
	{
		return;
	}

	*d.mParents.get()[0].mGrad += ((double)d.get_parents()[1] * pow((double)d.get_parents()[0], (double)d.get_parents()[1] - 1)) * grad;
}

void Double::BackPropagate_E_Power(const Double& d, const double& grad)
{
	if (d.mParentsCount != 1)
	{
		return;
	}

	*d.mParents.get()[0].mGrad += (double)*d.mValue * grad;
}

void Double::BackPropagateSubtraction(const Double& d, const double& grad)
{
	if (d.mParentsCount != 2)
	{
		return;
	}

	*d.mParents.get()[0].mGrad += 1 * grad;
	*d.mParents.get()[1].mGrad -= 1 * grad;
}

void Double::BackPropagateTan(const Double& d, const double& grad)
{
	if (d.mParentsCount != 1)
	{
		return;
	}

	*d.mParents.get()[0].mGrad += (1 + pow(tan((double)d.get_parents()[0]), 2)) * grad;
}

void Double::BackPropagateTanh(const Double& d, const double& grad)
{
	if (d.mParentsCount != 1)
	{
		return;
	}

	*d.mParents.get()[0].mGrad += (1 - pow(tanh((double)d.get_parents()[0]), 2)) * grad;
}

void Double::backpropagate()
{
	if (mBackPropagate == nullptr)
	{
		return;
	}

	mBackPropagate(*this, get_grad());
}

double Double::get_grad() const
{
	const double limit = 5;
	return *mGrad > limit ? limit : (*mGrad < -limit ? -limit : *mGrad);
}

Double* Double::get_parents() const
{
	return mParents.get();
}

unsigned int Double::get_parents_count() const
{
	return mParentsCount;
}

Double Double::get_e_pow() const
{
	const unsigned int parents_count = 1;
	Double* parents = new Double[parents_count];
	if (parents)
	{
		parents[0] = *this;
	}

	return Double(exp(*this->mValue), parents_count, parents, BackPropagate_E_Power);
}

Double Double::get_pow(const double& p) const
{
	const unsigned int parents_count = 2;
	Double* parents = new Double[parents_count];
	if (parents)
	{
		parents[0] = *this;
		parents[1] = Double(p);
	}

	return Double(pow(*this->mValue, p), parents_count, parents, BackPropagatePower);
}

Double Double::get_tan() const
{
	const unsigned int parents_count = 1;
	Double* parents = new Double[parents_count];
	if (parents)
	{
		parents[0] = *this;
	}

	return Double(tan(*this->mValue), parents_count, parents, BackPropagateTan);
}

Double Double::get_tanh() const
{
	const unsigned int parents_count = 1;
	Double* parents = new Double[parents_count];
	if (parents)
	{
		parents[0] = *this;
	}

	return Double(tanh(*this->mValue), parents_count, parents, BackPropagateTanh);
}

void Double::zero_grad()
{
	if (mParentsCount > 0)
	{
		for (int i = 0; i < mParentsCount; i++)
		{
			*mParents.get()[i].mGrad = 0;
			mParents.get()[i].zero_grad();
		}
	}
}

Double::operator double() const
{
	return *this->mValue;
}

bool Double::operator==(const Double& d) const
{
	return d.mValue.get() == mValue.get();
}

bool Double::operator!=(const Double& d) const
{
	return d.mValue.get() != mValue.get();
}

Double Double::operator*(const Double& rhs) const
{
	const unsigned int parents_count = 2;
	Double* parents = new Double[parents_count];
	if (parents)
	{
		parents[0] = *this;
		parents[1] = rhs;
	}

	return Double(*this->mValue * *rhs.mValue, parents_count, parents, BackPropagateMultiplication);
}

Double Double::operator/(const Double& rhs) const
{
	const unsigned int parents_count = 2;
	Double* parents = new Double[parents_count];
	if (parents)
	{
		parents[0] = *this;
		parents[1] = rhs;
	}

	return Double(*this->mValue / *rhs.mValue, parents_count, parents, BackPropagateDivision);
}

Double Double::operator+(const Double& rhs) const
{
	const unsigned int parents_count = 2;
	Double* parents = new Double[parents_count];
	if (parents)
	{
		parents[0] = *this;
		parents[1] = rhs;
	}

	return Double(*this->mValue + *rhs.mValue, parents_count, parents, BackPropagateAddition);
}

Double Double::operator-(const Double& rhs) const
{
	const unsigned int parents_count = 2;
	Double* parents = new Double[parents_count];
	if (parents)
	{
		parents[0] = *this;
		parents[1] = rhs;
	}

	return Double(*this->mValue - *rhs.mValue, parents_count, parents, BackPropagateSubtraction);
}

void backpropagate(const Double& d)
{
	vector<Double> sorted = get_sorted(d);
	for (int i = 0; i < sorted.size(); i++)
	{
		sorted[i].backpropagate();
	}
}

vector<Double> get_sorted(const Double& d) {
	queue<Double> openlist;
	vector<Double> openvec;
	queue<Double> result;
	vector<Double> sorted;

	if (d.get_parents_count() == 0)
	{
		return sorted;
	}

	openlist.push(d);

	while (openlist.size() > 0)
	{
		Double value = openlist.front();
		openlist.pop();

		result.push(value);

		for (int i = 0; i < value.get_parents_count(); i++)
		{
			if (!is_present(openvec, value.get_parents()[i]))
			{
				openvec.push_back(value.get_parents()[i]);
				openlist.push(value.get_parents()[i]);
			}
		}
	}

	unsigned int resultCount = result.size();
	if (resultCount != 0)
	{
		while (result.size() > 0)
		{
			sorted.push_back(result.front());
			result.pop();
		}
	}

	return sorted;
}

bool is_present(const vector<Double>& vec, const Double& d) {
	auto it = find(vec.begin(), vec.end(), d);
	return it != vec.end();
}
