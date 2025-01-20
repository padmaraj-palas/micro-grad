#ifndef MICRO_GRAD_H
#define MICRO_GRAD_H

#include <memory>
#include <vector>

using namespace std;

class Double;

void backpropagate(const Double&);
vector<Double> get_sorted(const Double&);
bool is_present(const vector<Double>&, const Double&);

class Double {
private:
	void (*mBackPropagate)(const Double&, const double&);
	shared_ptr<double> mGrad;
	shared_ptr<Double[]> mParents;
	unsigned int mParentsCount;
 	shared_ptr<double> mValue;

protected:
	Double(const double&, const unsigned int&, Double*, void (*bp)(const Double&, const double&));

	static void BackPropagateAddition(const Double&, const double&);
	static void BackPropagateDivision(const Double&, const double&);
	static void BackPropagateMultiplication(const Double&, const double&);
	static void BackPropagatePower(const Double&, const double&);
	static void BackPropagate_E_Power(const Double&, const double&);
	static void BackPropagateSubtraction(const Double&, const double&);
	static void BackPropagateTan(const Double&, const double&);
	static void BackPropagateTanh(const Double&, const double&);

public:
	Double();
	Double(const double&);

	void backpropagate();
	double get_grad() const;
	Double* get_parents() const;
	unsigned int get_parents_count() const;
	Double get_e_pow() const;
	Double get_pow(const double&) const;
	Double get_tan() const;
	Double get_tanh() const;
	void zero_grad();

	explicit operator double() const;

	bool operator==(const Double&) const;
	bool operator!=(const Double&) const;
	Double operator*(const Double&) const;
	Double operator/(const Double&) const;
	Double operator+(const Double&) const;
	Double operator-(const Double&) const;
};

#endif // !MICRO_GRAD_H


