#include <vector>
#include <queue>

#include "DoubleGradTest.h"
#include "MicroGrad.h"

Double calculate(const Double&, const Double&, const Double&);
void print_values(const Double&);

int test_micrograd() {
	Double d1 = 1;
	Double d2 = d1;
	Double d3 = d1 + d2 + d1 + d1 + d2;
	d3 = (d1.get_tan() * d1.get_e_pow() + d1.get_pow(4)) * d3;

	vector<Double> sorted = get_sorted(d3);


	printf("Result is %lf\n", (double)d3);
	printf("Size of sorted = %ld\n", sorted.size());

	for (int i = 0; i < sorted.size(); i++)
	{
		sorted[i].backpropagate();
	}

	for (int i = 0; i < sorted.size(); i++)
	{
		printf("gradient of %lf = %lf\n", (double)sorted[i], sorted[i].get_grad());
	}

	return 0;
}

unsigned int get_length(const char* str) {
	unsigned int count = 0;
	while (str != nullptr && str[count] != '\0') count++;
	return count;
}

void test_gradient_descentend() {
    double factor = 0.01;
    Double i1 = 2;
    Double i2 = 100;
    Double w = 0.001;
    Double b = 0.001;
    Double t1 = 4;
    Double t2 = 10000;
    
    char ch = '\0';
    while (ch != 'a') {
        Double out1 = calculate(i1, w, b);
        Double loss1 = (out1 - t1).get_tanh().get_pow(2);
        Double out2 = calculate(i2, w, b);
        Double loss2 = (out2 - t2).get_tanh().get_pow(2);
        Double loss = (loss1 + loss2) / Double(2.0);
        printf("Out1 = %lf, Out2 = %lf, Loss = %lf\n", (double)out1, (double)out2, (double)loss);
        loss.zero_grad();
        //print_values(out1);
        //print_values(out2);
        backpropagate(loss);
        //print_values(out1);
        //print_values(out2);
        w = (double)w - (w.get_grad() * factor);
        b = (double)b - (b.get_grad() * factor);
        ch = getchar();
    }
    
    ch = '\0';
    while (ch != 'a') {
        double v = 0;
        printf("Enter a number: \n");
        scanf("%lf", &v);
        printf("\n");
        Double out = calculate(v, w, b);
        printf("Out = %lf\n", (double)out);
    }
}

Double calculate(const Double& input, const Double& weight, const Double& bias) {
    return input * weight + bias;
}

void print_values(const Double& value) {
    const vector<Double>& values = get_sorted(value);
    for (int i = 0; i < values.size(); i++)
    {
        if (i > 0)
        {
            printf(", ");
        }
        
        printf("{%lf, %lf}", (double)values.at(i), values.at(i).get_grad());
    }
    
    printf("\n");
}
