


#ifndef _CUDA_COMPUTING_H_
#define _CUDA_COMPUTING_H_

#include <vector>
#include <iostream>
#include <algorithm>


#include "glew.h"
#include "device_launch_parameters.h"
#include "vector_types.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__


#include "body.h"

#define errorCheckCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

class Cuda_Computing {
public:

	//constructor
	Cuda_Computing(std::vector<Body> &bodies);

	//destructor
	~Cuda_Computing() {};

	bool initDevice();

	bool initVertexBuffer();

	float computeNewPositions();

	size_t getSize() const;

private:
	const int N;			// number of bodies
	float3 *positions;		// array of coords
	float *masses;			// array of masses
	float3 *velocities;		// array of velocities
	dim3 gridSize;			// The actual grid size needed, based on input size (num blocks)
	dim3 gridSizeTAO;
	dim3 gridSizeXAO;
	dim3 blockSize;			// Suggested block size to achieve maximum occupancy.(num threads per block)

private:

	//def constructor
	Cuda_Computing();

};

#endif