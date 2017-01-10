

#ifndef _CPU_COMPUTING_H_
#define _CPU_COMPUTING_H_

#include <vector>
#include <iostream>
#include <thread>
#include <mutex>
#include <glm/gtc/type_ptr.hpp>
#include "body.h"



class Cpu_Computing {
public:

	//constructor
	Cpu_Computing(std::vector<Body> &bodies);

	//destructor
	~Cpu_Computing();


	void setThreads();

	void setThreads(unsigned int nthreads);

	void compute_forces(float dtG);

	void computeTile(const int tid, const int &num_threads, float dtG);

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

	//def constructor
	Cpu_Computing();



};

#endif