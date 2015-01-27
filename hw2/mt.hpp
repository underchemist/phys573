// Copyright 2015 Yann-Sebastien Tremblay-Johnston
#ifndef MT_H
#define MT_H

#include <chrono>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>

using boost::random::uniform_real_distribution;
using boost::variate_generator;
using boost::mt19937;

// set seed for mersenne twister RNG
typedef std::chrono::high_resolution_clock hrclock;
auto seed = hrclock::now().time_since_epoch().count();

// produce random numbers on the interval [0.0, 1.0)
mt19937  rng(seed);
uniform_real_distribution<double> dist(0.0, 1.0);
variate_generator<mt19937&, uniform_real_distribution<double> > rnd_gen(rng, dist);

#endif
