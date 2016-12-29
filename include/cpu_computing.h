

#ifndef _CPU_COMPUTING_H_
#define _CPU_COMPUTING_H_

#include "body.h"
#include <vector>
#include <iostream>
#include <thread>
#include <mutex>
#include <glm/gtc/type_ptr.hpp>


class Cpu_Computing {
public:

	Cpu_Computing(std::vector<Body> &bodies);

	~Cpu_Computing() { 
		delete[] masses;
		delete[] velocities;
	};

	void setThreads();

	void setThreads(unsigned int nthreads);

	void compute(float dt);

	void computeTile(const int tid, const int &num_threads, float dt);

	const float *getPositions() const;

	size_t getSize() const;



private:

	// number of bodies
	const size_t size;
	// array of coords
	std::vector<glm::vec3> positions;
	// array of masses
	float *masses;
	// array of velocities
	glm::vec3 *velocities;
	// number of threads
	unsigned int numThreads;
	// mutex to control access to the arrays
	std::mutex mutexTiles;


private:

	Cpu_Computing();



};

#endif