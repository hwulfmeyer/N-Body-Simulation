


#ifndef _CUDA_COMPUTING_H_
#define _CUDA_COMPUTING_H_

#include <vector>
#include <iostream>
#include <algorithm>
#include <glm/gtc/type_ptr.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuda_util.h"
#include "body.h"



class Cuda_Computing {
public:

	//constructor
	Cuda_Computing(std::vector<Body> &bodies);

	//destructor
	~Cuda_Computing();

	bool initDevice();

	bool initDeviceMemory();

	void computeForces(float dt);

	const float *getPositions() const;

	size_t getSize() const;

private:

	// number of bodies
	const size_t size;
	// array of coords
	glm::vec3 *positions;
	// array of masses
	float *masses;
	// array of velocities
	glm::vec3 *velocities;
	// threads per block on cuda device
	unsigned int num_threads_per_block;
	unsigned int num_blocks;


private:

	//def constructor
	Cuda_Computing();

};

#endif