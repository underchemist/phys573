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
using std::ofstream;

using std::vector;

const size_t DIM = 3;  // dimension
const size_t NEL = 2;  // number of electrons for He
const size_t N = 100;  // number of random walkers

typedef vector<double> coord;
typedef vector<coord> electrons;
typedef vector<electrons> walkers;

double alpha = 0.0;  // variational parameter
double delta = 1.0;  // trial step size
double Esum = 0.0;
double Esum_sqr = 0.0;
int accept_count = 0;  // number of accepted Metropolis steps
int MCsteps = 10000;
int thermsteps = static_cast<int>(MCsteps * 0.2);

// function prototypes
void init(walkers &r);

double psi(const coord &c1, const coord &c2);

double elocal(const coord &c1, const coord &c2);

void adjust_delta();

void mstep(electrons &e);

void singleMCstep(walkers &r);

void MCtherm(walkers &r);

void MCrun(walkers &r);


#endif
