


#include "cuda_computing.cuh"

namespace Device {
	// array of coords
	float3 *positions;
	// array of masses
	float *masses;
	// array of velocities
	float3 *velocities;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// kernel for filling the vertexPointer for opengl/cuda inop
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		MakeVerticesKernel(float4 *vertexPointer, float3 *positions, const unsigned int N) {
		unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

		if (tidx < N) {
			vertexPointer[tidx].x = positions[tidx].x;
			vertexPointer[tidx].y = positions[tidx].y;
			vertexPointer[tidx].z = positions[tidx].z;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// device physics calculations
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__device__
		float3
		bodyBodyInteraction(float3 pos_body_cur, float3 pos_body_oth, float mass_oth, float epss, float3 velo) {
		float3 dir;
		//3 FLOP
		dir.x = pos_body_oth.x - pos_body_cur.x;
		dir.y = pos_body_oth.y - pos_body_cur.y;
		dir.z = pos_body_oth.z - pos_body_cur.z;
		// 6 FLOP
		float distSqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z + epss;
		// 4 FLOP
		float partForce = mass_oth / sqrtf(distSqr*distSqr*distSqr);
		// 6 FLOP
		velo.x += dir.x * partForce;
		velo.y += dir.y * partForce;
		velo.z += dir.z * partForce;
		return velo;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// kernel for computing velocities
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		ComputeVelocities(float3 *positions, float* masses, float3 *velocities, const unsigned int N, float epss) {
		unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

		if (tidx < N) {
			float3 velo = make_float3(0, 0, 0);
			for (unsigned int k = 0; k < N; ++k)
			{
				// in total 19 FLOP per body body Interaction
				velo = bodyBodyInteraction(positions[tidx], positions[k], masses[k], epss, velo);
			}

			velocities[tidx].x += velo.x;
			velocities[tidx].y += velo.y;
			velocities[tidx].z += velo.z;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// kernel for integration step using calculated velocities before
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		IntegrateVelocities(float3 *positions, float3 *velocities, const unsigned int N, float dtG) {
		unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

		if (tidx < N) {
			positions[tidx].x += dtG * velocities[tidx].x;
			positions[tidx].y += dtG * velocities[tidx].y;
			positions[tidx].z += dtG * velocities[tidx].z;
		}
	}

}

////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, copies all the stuff to this class
////////////////////////////////////////////////////////////////////////////////////////////////////
Cuda_Computing::Cuda_Computing(std::vector<Body> &bodies) : N(bodies.size()) {
	this->positions = new float3[N];
	this->masses = new float[N];
	this->velocities = new float3[N];

	for (unsigned int i = 0; i < N; ++i)
	{
		positions[i].x = bodies[i].position.x;
		positions[i].y = bodies[i].position.y;
		positions[i].z = bodies[i].position.z;

		masses[i] = bodies[i].mass;

		velocities[i].x = bodies[i].velocity.x;
		velocities[i].y = bodies[i].velocity.y;
		velocities[i].z = bodies[i].velocity.z;
	}

	std::cout << "Cuda_Computing::Cuda_Computing() - Copying of " << N << " bodies done." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor, deletes our dynamic arrays & frees memory on cuda device
////////////////////////////////////////////////////////////////////////////////////////////////////
Cuda_Computing::~Cuda_Computing() {
	//free arrays on cuda device
	 errorCheckCuda(cudaFree(Device::positions));
	 errorCheckCuda(cudaFree(Device::masses));
	 errorCheckCuda(cudaFree(Device::velocities));
	
	// free dynamic arrays
	delete[] positions;
	delete[] masses;
	delete[] velocities;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// initializes device, detects hardware, number of threads per block
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
Cuda_Computing::initDevice() {
	//check execution environment
	int deviceCount = 0;
	int device_handle = 0;
	 errorCheckCuda(cudaGetDeviceCount(&deviceCount));

	if (0 == deviceCount) {
		std::cerr << "initDevice() : No CUDA device found." << std::endl;
		return false;
	}

	if (deviceCount > 1) {
		std::cerr << "initDevice() : Multiple CUDA devices found. Using first one." << std::endl;
	}

	// set the device
	 errorCheckCuda(cudaSetDevice(device_handle));

	cudaDeviceProp device_props;
	 errorCheckCuda(cudaGetDeviceProperties(&device_props, device_handle));
	//std::cout << "Max CC: " << device_props.major << "   Min CC: " << device_props.minor << std::endl;

	// determine thread layout
	threadsPerBlock = 128;
	numBlocks = N / threadsPerBlock;
	if(0 != N % threadsPerBlock) numBlocks++;

	std::cout << "block size = " << numBlocks << " :: "
		<< "threads per Block = " << threadsPerBlock << std::endl;

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
	 errorCheckCuda(cudaMalloc((void **)&Device::positions,
		N * sizeof(float3))
	);
	 errorCheckCuda(cudaMalloc((void **)&Device::masses,
		N * sizeof(float))
	);
	 errorCheckCuda(cudaMalloc((void **)&Device::velocities,
		N * sizeof(float3))
	);

	// copy device memory
	 errorCheckCuda(cudaMemcpy((void *)Device::positions, (void *)positions,
		N * sizeof(float3),
		cudaMemcpyHostToDevice)
	);
	 errorCheckCuda(cudaMemcpy((void *)Device::masses, (void *)masses,
		N * sizeof(float),
		cudaMemcpyHostToDevice)
	);
	 errorCheckCuda(cudaMemcpy((void *)Device::velocities, (void *)velocities,
		N * sizeof(float3),
		cudaMemcpyHostToDevice)
	);

	return true;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// creating vertexBuffer for opengl/cuda used for inop between the two
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
Cuda_Computing::initDeviceVertexBuffer() {

	return false;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Cuda_Computing::computeForces(float dt) {
	// run kernel computing velocities
	Device::ComputeVelocities << < numBlocks, threadsPerBlock >> > (Device::positions, Device::masses, Device::velocities, N, EPS2);
	//used only for error checking
	/*errorCheckCuda(cudaPeekAtLastError());
	errorCheckCuda(cudaDeviceSynchronize());*/

	// run kernel integrating velocities
	Device::IntegrateVelocities << < numBlocks, threadsPerBlock >> > (Device::positions, Device::velocities, N, dt*G);
	//used only for error checking
	/*errorCheckCuda(cudaPeekAtLastError());
	errorCheckCuda(cudaDeviceSynchronize());*/
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// copying from device to host (with openGL inop not necessary)
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Cuda_Computing::copyPositionsFromDevice() {
	// copy result back to host
	cudaMemcpy((void *)positions, (void *)Device::positions,
		N * sizeof(float3),
		cudaMemcpyDeviceToHost);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// returns positions as flat array 
////////////////////////////////////////////////////////////////////////////////////////////////////
const float *
Cuda_Computing::getPositions() const {
	return nullptr;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// returns the number of bodies 
////////////////////////////////////////////////////////////////////////////////////////////////////
size_t 
Cuda_Computing::getSize() const {
	return N;
}
