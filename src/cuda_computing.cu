


#include "cuda_computing.cuh"

namespace Device {
	// array of coords
	glm::vec3 *positions;
	// array of masses
	float *masses;
	// array of velocities
	glm::vec3 *velocities;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, copies all the stuff to this class
////////////////////////////////////////////////////////////////////////////////////////////////////
Cuda_Computing::Cuda_Computing(std::vector<Body> &bodies) : size(bodies.size()) {
	this->positions = new glm::vec3[size];
	this->masses = new float[size];
	this->velocities = new glm::vec3[size];

	for (unsigned int i = 0; i < size; ++i)
	{
		positions[i] = bodies[i].position;

		masses[i] = bodies[i].mass;

		velocities[i] = bodies[i].velocity;
	}

	std::cout << "Cuda_Computing::Cuda_Computing() - Copying of " << size << " bodies done." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor, deletes our dynamic arrays & frees memory on cuda device
////////////////////////////////////////////////////////////////////////////////////////////////////
Cuda_Computing::~Cuda_Computing() {
	// free dynamic arrays
	delete[] masses;
	delete[] velocities;

	//free arrays on cuda device
	checkErrorsCuda(cudaFree(Device::positions));
	checkErrorsCuda(cudaFree(Device::masses));
	checkErrorsCuda(cudaFree(Device::velocities));
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// initializes device, detects hardware, number of threads per block
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
Cuda_Computing::initDevice() {
	//check execution environment
	int deviceCount = 0;
	int device_handle = 0;
	checkErrorsCuda(cudaGetDeviceCount(&deviceCount));

	if (0 == deviceCount) {
		std::cerr << "initDevice() : No CUDA device found." << std::endl;
		return false;
	}

	if (deviceCount > 1) {
		std::cerr << "initDevice() : Multiple CUDA devices found. Using first one." << std::endl;
	}

	// set the device
	checkErrorsCuda(cudaSetDevice(device_handle));

	cudaDeviceProp device_props;
	checkErrorsCuda(cudaGetDeviceProperties(&device_props, device_handle));
	//std::cout << "Max CC: " << device_props.major << "   Min CC: " << device_props.minor << std::endl;

	// determine thread layout
	unsigned int max_threads_per_block = 128;

	num_threads_per_block = std::min(size, max_threads_per_block);
	num_blocks = size / max_threads_per_block;
	if (0 != size % max_threads_per_block) {
		num_blocks++;
	}
	std::cout << "num_blocks = " << num_blocks << " :: "
		<< "num_threads_per_block = " << num_threads_per_block << std::endl;

	// initialize memory
	Device::positions = nullptr;
	Device::masses = nullptr;
	Device::velocities = nullptr;
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// allocating device memory and copying memory to device
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
Cuda_Computing::initDeviceMemory() {
	Device::positions = this->positions;
	Device::masses = this->masses;
	Device::velocities = this->velocities;

	// allocate device memory
	checkErrorsCuda(cudaMalloc((void **)&Device::positions,
		size * sizeof(glm::vec3))
	);
	checkErrorsCuda(cudaMalloc((void **)&Device::masses,
		size * sizeof(float))
	);
	checkErrorsCuda(cudaMalloc((void **)&Device::velocities,
		size * sizeof(glm::vec3))
	);

	// copy device memory
	checkErrorsCuda(cudaMemcpy((void *)Device::positions, (void *)positions,
		size * sizeof(glm::vec3),
		cudaMemcpyHostToDevice)
	);
	checkErrorsCuda(cudaMemcpy((void *)Device::masses, (void *)masses,
		size * sizeof(float),
		cudaMemcpyHostToDevice)
	);
	checkErrorsCuda(cudaMemcpy((void *)Device::velocities, (void *)velocities,
		size * sizeof(glm::vec3),
		cudaMemcpyHostToDevice)
	);

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// device physics calculations
////////////////////////////////////////////////////////////////////////////////////////////////////
__device__
glm::vec3
bodyBodyInteraction(glm::vec3 pos_body_cur, glm::vec3 pos_body_oth, float mass_oth, float EPS2) {
	glm::vec3 dir = pos_body_oth - pos_body_cur; // 3 FLOP
	float distSqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z + EPS2; // 6 FLOP

	float partForce = mass_oth / sqrt(distSqr*distSqr*distSqr); // 4 FLOP
	return dir * partForce; // 3 FLOP
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel for computing velocities
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
device_computeForces(glm::vec3 *positions, float* masses, glm::vec3 *velocities, const int N, float EPS2) {
	
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < N) {
		glm::vec3 velo;
		for (unsigned int k = 0; k < N; ++k)
		{
			velo += bodyBodyInteraction(positions[tid], positions[k], masses[k], EPS2); // 3 FLOP
			// in total 19 FLOP per body body Interaction 
		}

		velocities[tid] += velo;
	}

	
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel for integration step using calculated velocities before
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__
void
device_integrateForces(glm::vec3 *positions, glm::vec3 *velocities, const int N, float dtG) {

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	glm::vec3 velo = glm::vec3(0, 0, 0);
	if (tid < N) {
		positions[tid] += dtG * velocities[tid];
	}
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Cuda_Computing::computeForces(float dt) {

	float dtG = dt*G;
	// run kernel computing velocities
	device_computeForces <<< num_blocks, num_threads_per_block >> > (Device::positions, Device::masses, Device::velocities, size, EPS2);

	// run kernel integrating velocities
	device_integrateForces <<< num_blocks, num_threads_per_block >> > (Device::positions, Device::velocities, size, dtG);


	// copy result back to host
	checkErrorsCuda(cudaMemcpy((void *)positions, (void *)Device::positions,
		size * sizeof(glm::vec3),
		cudaMemcpyDeviceToHost)
	);
	
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// returns positions as flat array 
////////////////////////////////////////////////////////////////////////////////////////////////////
const float *
Cuda_Computing::getPositions() const {
	return glm::value_ptr(positions[0]);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// returns the number of bodies 
////////////////////////////////////////////////////////////////////////////////////////////////////
size_t 
Cuda_Computing::getSize() const {
	return size;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
//! Entry point to device == KERNEL
////////////////////////////////////////////////////////////////////////////////////////////////////
