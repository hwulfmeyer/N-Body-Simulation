


#include "cuda_computing.cuh"

namespace Device {
	// CUDA global constants
	__device__ __constant__
		float EPSILON2;
	__device__ __constant__
		float DTGRAVITY;

	// array of masses
	float *masses;
	// array of velocities
	float3 *velocities;
	// array of positions
	float3 *positions;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// device physics calculations
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__device__
		float3
		bodyBodyInteraction(float3 myPos, float3 othPos, float mass_oth, float3 velo) {
		float3 dir;
		//3 FLOP
		dir.x = othPos.x - myPos.x;
		dir.y = othPos.y - myPos.y;
		dir.z = othPos.z - myPos.z;
		// 6 FLOP
		float distSqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z + EPSILON2;
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
		ComputeVelocities(float3 *positions, float* masses, float3 *velocities, unsigned int N) {
		unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
		if (tidx < N) {
			float3 myPos = positions[tidx];
			float3 myVelo = velocities[tidx];
			for (unsigned int k = 0; k < N; ++k)
			{
				// in total 19 FLOP per body body Interaction
				myVelo = bodyBodyInteraction(myPos, positions[k], masses[k], myVelo);
			}
			
			myPos.x += myVelo.x * DTGRAVITY;
			myPos.y += myVelo.y * DTGRAVITY;
			myPos.z += myVelo.z * DTGRAVITY;
	
			positions[tidx] = myPos;
			velocities[tidx] = myVelo;
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

	std::cerr << "Cuda_Computing::Cuda_Computing() - Copying of " << N << " bodies done." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor, deletes our dynamic arrays & frees memory on cuda device
////////////////////////////////////////////////////////////////////////////////////////////////////
Cuda_Computing::~Cuda_Computing() {
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
	//std::cerr << "Max CC: " << device_props.major << "   Min CC: " << device_props.minor << std::endl;

	// determine thread layout
	blockSize = dim3(256, 1, 1);
	int numBlocks = N / blockSize.x;
	if (0 != N % blockSize.x) numBlocks++;
	gridSize = dim3(numBlocks, 1, 1);

	std::cerr << "num blocks = " << numBlocks << " :: "
		<< "threads per Block = " << blockSize.x << std::endl;

	float dtG = G*DT;

	errorCheckCuda(cudaMemcpyToSymbol(Device::EPSILON2, &EPS2, sizeof(float), 0, cudaMemcpyHostToDevice));
	errorCheckCuda(cudaMemcpyToSymbol(Device::DTGRAVITY, &dtG, sizeof(float), 0, cudaMemcpyHostToDevice));
	return true;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// creating vertexBuffer for openGL/cuda inop
////////////////////////////////////////////////////////////////////////////////////////////////////
bool
Cuda_Computing::initVertexBuffer() {
	// allocate & register the vertexbuffer
	cudaGraphicsResource *cuda_vbo_resources[3];
	GLuint vao;
	GLuint vbo_pos;
	GLuint vbo_mass;
	GLuint vbo_velos;

	// create a vertex array of our device pointer for opengl/cuda inop
	glGenVertexArrays(3, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo_pos);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
	glBufferData(GL_ARRAY_BUFFER, N * sizeof(float3), positions, GL_DYNAMIC_COPY); 	// buffer data with our positions
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	glGenBuffers(1, &vbo_mass);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_mass);
	glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), masses, GL_DYNAMIC_COPY);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(1);

	glGenBuffers(1, &vbo_velos);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_velos);
	glBufferData(GL_ARRAY_BUFFER, N * sizeof(float3), velocities, GL_DYNAMIC_COPY);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(2);

	glBindVertexArray(vao);

	//cudaGLRegisterBufferObject(vbo); ///deprecated
	errorCheckCuda(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resources[0], vbo_pos, cudaGraphicsMapFlagsNone));
	errorCheckCuda(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resources[1], vbo_mass, cudaGraphicsMapFlagsNone));
	errorCheckCuda(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resources[2], vbo_velos, cudaGraphicsMapFlagsNone));

	// Map the buffer to CUDA
	//cudaGLMapBufferObject(&vptr, vbo); ///deprecated
	errorCheckCuda(cudaGraphicsMapResources(3, cuda_vbo_resources));
	size_t numBytes;
	errorCheckCuda(cudaGraphicsResourceGetMappedPointer((void**)&Device::positions, &numBytes, cuda_vbo_resources[0]));
	errorCheckCuda(cudaGraphicsResourceGetMappedPointer((void**)&Device::masses, &numBytes, cuda_vbo_resources[1]));
	errorCheckCuda(cudaGraphicsResourceGetMappedPointer((void**)&Device::velocities, &numBytes, cuda_vbo_resources[2]));

	// Unmap the buffer
	//cudaGLUnmapBufferObject(vbo); /// deprecated
	errorCheckCuda(cudaGraphicsUnmapResources(3, cuda_vbo_resources));
	return true;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel entry point
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Cuda_Computing::computeNewPositions() {
	// run kernel computing velocities
	Device::ComputeVelocities << < gridSize, blockSize >> > (Device::positions, Device::masses, Device::velocities, N);
	//used only for error checking
	//errorCheckCuda(cudaPeekAtLastError());
	errorCheckCuda(cudaDeviceSynchronize());
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// returns the number of bodies 
////////////////////////////////////////////////////////////////////////////////////////////////////
size_t
Cuda_Computing::getSize() const {
	return N;
}
