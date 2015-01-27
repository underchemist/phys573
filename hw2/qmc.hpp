// Copyright 2015 Yann-Sebastien Tremblay-Johnston
#ifndef QMC_H
#define QMC_H

#include <cmath>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using std::cin;
using std::cout;
using std::endl;

using std::vector;

const size_t DIM = 3;  // dimension
const size_t NEL = 2;  // number of electrons for He
const size_t N = 5;  // number of random walkers

typedef vector<double> coord;
typedef vector<coord> electrons;
typedef vector<electrons> walkers;

double alpha = 0.05;  // variational parameter
double delta;  // trial step size

// function prototypes
void init(walkers &r);

double psi(const coord &c1, const coord &c2);



#endif
