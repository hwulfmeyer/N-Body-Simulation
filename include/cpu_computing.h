

#ifndef _CPU_COMPUTING_H_
#define _CPU_COMPUTING_H_

#include "body.h"
#include <vector>
#include <iostream>
#include <thread>
#include <mutex>


class Cpu_Computing {
public:

	Cpu_Computing(std::vector<Body> &bodies);

	~Cpu_Computing() { 
		delete[] positions;
		delete[] masses;
		delete[] velocities;
	};

	void setThreads();

	void setThreads(unsigned int nthreads);

	void compute(float dt);

	void computeTile(const int tid, const int &num_threads, float dt);


	float *getPositions();

	size_t getSize() const;

	// array of coords
	float *positions;

private:

	// number of bodies
	const size_t size;

	// array of masses
	float *masses;
	// array of velocities
	float *velocities;
	// number of threads
	unsigned int numThreads;
	// mutex to control access to the arrays
	std::mutex mutexTiles;


private:

	Cpu_Computing();



};

#endif