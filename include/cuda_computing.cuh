#ifndef _CUDA_COMPUTING_H_
#define _CUDA_COMPUTING_H_

#include <vector>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include "body.h"



class Cuda_Computing {
public:

	//constructor
	Cuda_Computing(std::vector<Body> &bodies);

	//destructor
	~Cuda_Computing();

	bool initDevice();

	bool initDeviceMemory();

	void compute_forces(float dt);

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


private:

	//def constructor
	Cuda_Computing();

};

#endif