


#include "cuda_computing.cuh"

#define THREADS_PER_BLOCK 128
#define BODIES_PER_THREAD 4 //only mutiples of 2
#define EB_SHAREDMEMORY

namespace Device {
	// CUDA global constants
	__device__ __constant__
		float EPSILON2;
	__device__ __constant__
		float DTGRAVITY;
	__device__ __constant__
		int NBODIES;
	__device__ __constant__
		int NTHREADS;

	// array of masses
	float *masses;
	// array of velocities
	float3 *velocities;
	// array of positions
	float3 *positions;

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// physics calculations between bodies
	// NOTES: try not using EPSILON for calculations
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
		// 4 FLOP <- wrong
		float partForce = mass_oth / sqrtf(distSqr*distSqr*distSqr);
		// 6 FLOP
		velo.x += dir.x * partForce;
		velo.y += dir.y * partForce;
		velo.z += dir.z * partForce;
		// in total 19 FLOP per body body Interaction
		return velo;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// physics calculations between bodies [v.SHARED MEMORY]
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__device__
		float3
		smBodyBodyInteractionV1(float3 myPos, float4 othPos, float3 velo) {
		float3 dir;
		//3 FLOP
		dir.x = othPos.x - myPos.x;
		dir.y = othPos.y - myPos.y;
		dir.z = othPos.z - myPos.z;
		// 6 FLOP
		float distSqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z + EPSILON2;
		// 4 FLOP
		float partForce = othPos.w / sqrtf(distSqr*distSqr*distSqr);
		// 6 FLOP
		velo.x += dir.x * partForce;
		velo.y += dir.y * partForce;
		velo.z += dir.z * partForce;
		return velo;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// physics calculations between bodies [v.SHARED MEMORY + RSQRTF()]
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__device__
		float3
		smBodyBodyInteractionV2(float3 myPos, float4 othPos, float3 velo) {
		float3 dir;
		//3 FLOP
		dir.x = othPos.x - myPos.x;
		dir.y = othPos.y - myPos.y;
		dir.z = othPos.z - myPos.z;
		// 6 FLOP
		float distSqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z + EPSILON2;
		// 4 FLOP
		float partForce = rsqrtf(distSqr*distSqr*distSqr);
		partForce *= othPos.w;
		// 6 FLOP
		velo.x += dir.x * partForce;
		velo.y += dir.y * partForce;
		velo.z += dir.z * partForce;
		return velo;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// physics calculations between bodies [v.SHARED MEMORY + RSQRTF() + float4pad usage]
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__device__
		float3
		smBodyBodyInteractionV2pad(float3 myPos, float3 othPos, float othMass, float3 velo) {
		float3 dir;
		//3 FLOP
		dir.x = othPos.x - myPos.x;
		dir.y = othPos.y - myPos.y;
		dir.z = othPos.z - myPos.z;
		// 6 FLOP
		float distSqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z + EPSILON2;
		// 4 FLOP
		float partForce = rsqrtf(distSqr*distSqr*distSqr);
		partForce *= othMass;
		// 6 FLOP
		velo.x += dir.x * partForce;
		velo.y += dir.y * partForce;
		velo.z += dir.z * partForce;
		return velo;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// naive kernel computing velocities [v1.NAIVE]
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		computeVelocities(float3 *positions, float* masses, float3 *velocities) {
		unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

		float3 myPos = positions[tidx];
		float3 myVelo = velocities[tidx];

		for (unsigned int k = 0; k < NBODIES; ++k)
		{
			myVelo = bodyBodyInteraction(myPos, positions[k], masses[k], myVelo);
		}

		myPos.x += myVelo.x * DTGRAVITY;
		myPos.y += myVelo.y * DTGRAVITY;
		myPos.z += myVelo.z * DTGRAVITY;

		positions[tidx] = myPos;
		velocities[tidx] = myVelo;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// kernel [v2.SHARED] 
	// in order for it to work correctly NBODIES % NUM_THREADS_PER_BLOCK = 0 must be true
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		smComputeVelocitiesV1(float3 *positions, float* masses, float3 *velocities) {
		unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
		__shared__ float4 smPos[THREADS_PER_BLOCK];

		float3 myPos = positions[tidx];
		float3 myVelo = velocities[tidx];
		// each loop step copies NUM_THREADS_PER_BLOCK values into shared memory
		// hence we have to do it gridDim.x = NBODIES/NUM_THREADS_PER_BLOCK times to get to each body
		for (int curTileIdx = 0; curTileIdx < gridDim.x; curTileIdx++) {
			int idx = curTileIdx * blockDim.x + threadIdx.x;
			smPos[threadIdx.x].x = positions[idx].x;
			smPos[threadIdx.x].y = positions[idx].y;
			smPos[threadIdx.x].z = positions[idx].z;
			smPos[threadIdx.x].w = masses[idx];
			__syncthreads();
			//compute interactions in our current sharedMemory

			for (unsigned int i = 0; i < NTHREADS; i++)
			{
				myVelo = smBodyBodyInteractionV1(myPos, smPos[i], myVelo);
			}
			__syncthreads();
		}

		myPos.x += myVelo.x * DTGRAVITY;
		myPos.y += myVelo.y * DTGRAVITY;
		myPos.z += myVelo.z * DTGRAVITY;

		__syncthreads();
		positions[tidx] = myPos;
		velocities[tidx] = myVelo;

	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// kernel [v4.SHARED + LOOP UNROLL + rsqrtf() + padding]
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		smComputeVelocitiesV3(float3 *positions, float* masses, float3 *velocities) {
		unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
		__shared__ float4 smPos[THREADS_PER_BLOCK];

		float3 myPos = positions[tidx];
		float3 myVelo = velocities[tidx];
		// each loop step copies NUM_THREADS_PER_BLOCK values into shared memory
		// hence we have to do it gridDim.x = NBODIES/NUM_THREADS_PER_BLOCK times to get to each body
		for (int curTileIdx = 0; curTileIdx < gridDim.x; curTileIdx++) {
			int idx = curTileIdx * blockDim.x + threadIdx.x;
			smPos[threadIdx.x].x = positions[idx].x;
			smPos[threadIdx.x].y = positions[idx].y;
			smPos[threadIdx.x].z = positions[idx].z;
			smPos[threadIdx.x].w = masses[idx];
			__syncthreads();
			//compute interactions in our current sharedMemory

			for (unsigned int i = 0; i < THREADS_PER_BLOCK; i++)
			{
				myVelo = smBodyBodyInteractionV2(myPos, smPos[i], myVelo);
			}
			__syncthreads();
		}

		myPos.x += myVelo.x * DTGRAVITY;
		myPos.y += myVelo.y * DTGRAVITY;
		myPos.z += myVelo.z * DTGRAVITY;

		__syncthreads();
		positions[tidx] = myPos;
		velocities[tidx] = myVelo;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// kernel [v1_1.NAIVE + TWO AT ONCE]
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		ComputeVelocitiesTao(float3 *positions, float* masses, float3 *velocities) {
		unsigned int tidA = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
		unsigned int tidB = tidA + 1;

		float3 myPosA = positions[tidA];
		float3 myPosB = positions[tidB];
		float3 myVeloA = velocities[tidA];
		float3 myVeloB = velocities[tidB];
		for (unsigned int k = 0; k < NBODIES; ++k)
		{
			myVeloA = bodyBodyInteraction(myPosA, positions[k], masses[k], myVeloA);
			myVeloB = bodyBodyInteraction(myPosB, positions[k], masses[k], myVeloB);
		}
		myPosA.x += myVeloA.x * DTGRAVITY;
		myPosA.y += myVeloA.y * DTGRAVITY;
		myPosA.z += myVeloA.z * DTGRAVITY;
		myPosB.x += myVeloB.x * DTGRAVITY;
		myPosB.y += myVeloB.y * DTGRAVITY;
		myPosB.z += myVeloB.z * DTGRAVITY;

		positions[tidA] = myPosA;
		positions[tidB] = myPosB;
		velocities[tidA] = myVeloA;
		velocities[tidB] = myVeloB;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// kernel [v1_2.NAIVE + X AT ONCE]
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		ComputeVelocitiesXao(float3 *positions, float* masses, float3 *velocities) {
		unsigned int tids[BODIES_PER_THREAD];

		tids[0] = blockIdx.x * blockDim.x * BODIES_PER_THREAD + threadIdx.x * BODIES_PER_THREAD;

#pragma unroll BODIES_PER_THREAD
		for (unsigned int i = 1; i < BODIES_PER_THREAD; ++i) {
			tids[i] = tids[i - 1] + 1;
		}

		float3 myPos[BODIES_PER_THREAD];
		float3 myVelo[BODIES_PER_THREAD];

#pragma unroll BODIES_PER_THREAD
		for (int i = 0; i < BODIES_PER_THREAD; ++i) {
			myPos[i] = positions[tids[i]];
		}

#pragma unroll BODIES_PER_THREAD
		for (int i = 0; i < BODIES_PER_THREAD; ++i) {
			myVelo[i] = velocities[tids[i]];
		}

		for (unsigned int k = 0; k < NBODIES; ++k)
		{

#pragma unroll BODIES_PER_THREAD
			for (int i = 0; i < BODIES_PER_THREAD; ++i) {
				myVelo[i] = bodyBodyInteraction(myPos[i], positions[k], masses[k], myVelo[i]);
			}
		}

#pragma unroll BODIES_PER_THREAD
		for (int i = 0; i < BODIES_PER_THREAD; ++i) {
			myPos[i].x += myVelo[i].x * DTGRAVITY;
			myPos[i].y += myVelo[i].y * DTGRAVITY;
			myPos[i].z += myVelo[i].z * DTGRAVITY;
		}
		__syncthreads();
#pragma unroll BODIES_PER_THREAD
		for (int i = 0; i < BODIES_PER_THREAD; ++i) {
			positions[tids[i]] = myPos[i];
		}

#pragma unroll BODIES_PER_THREAD
		for (int i = 0; i < BODIES_PER_THREAD; ++i) {
			velocities[tids[i]] = myVelo[i];
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// kernel [v4_1.SHARED + LOOP UNROLL + rsqrtf() + TWO AT ONCE]
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		smComputeVelocitiesV3tao(float3 *positions, float* masses, float3 *velocities) {
		unsigned int tidA = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
		unsigned int tidB = tidA + 1;
		__shared__ float4 smPos[THREADS_PER_BLOCK];

		float3 myPosA = positions[tidA];
		float3 myPosB = positions[tidB];
		float3 myVeloA = velocities[tidA];
		float3 myVeloB = velocities[tidB];

		for (int curTileIdx = 0; curTileIdx < gridDim.x * 2; ++curTileIdx) {
			int idx = curTileIdx * blockDim.x + threadIdx.x;
			smPos[threadIdx.x].x = positions[idx].x;
			smPos[threadIdx.x].y = positions[idx].y;
			smPos[threadIdx.x].z = positions[idx].z;
			smPos[threadIdx.x].w = masses[idx];

			__syncthreads();
			//compute interactions in our current sharedMemory

			for (unsigned int i = 0; i < THREADS_PER_BLOCK; i++)
			{
				myVeloA = smBodyBodyInteractionV2(myPosA, smPos[i], myVeloA);
				myVeloB = smBodyBodyInteractionV2(myPosB, smPos[i], myVeloB);
			}
			__syncthreads();
		}

		myPosA.x += myVeloA.x * DTGRAVITY;
		myPosA.y += myVeloA.y * DTGRAVITY;
		myPosA.z += myVeloA.z * DTGRAVITY;
		myPosB.x += myVeloB.x * DTGRAVITY;
		myPosB.y += myVeloB.y * DTGRAVITY;
		myPosB.z += myVeloB.z * DTGRAVITY;

		__syncthreads();
		positions[tidA] = myPosA;
		positions[tidB] = myPosB;
		velocities[tidA] = myVeloA;
		velocities[tidB] = myVeloB;
	}


	////////////////////////////////////////////////////////////////////////////////////////////////////
	// kernel [v4_1.SHARED + LOOP UNROLL + rsqrtf() + X AT ONCE]
	////////////////////////////////////////////////////////////////////////////////////////////////////
	__global__
		void
		smComputeVelocitiesV3xao(float3 *positions, float* masses, float3 *velocities) {
		unsigned int tids[BODIES_PER_THREAD];
		__shared__ float4 smPos[THREADS_PER_BLOCK];

		tids[0] = blockIdx.x * blockDim.x * BODIES_PER_THREAD + threadIdx.x * BODIES_PER_THREAD;

#pragma unroll BODIES_PER_THREAD
		for (unsigned int i = 1; i < BODIES_PER_THREAD; ++i) {
			tids[i] = tids[i - 1] + 1;
		}

		float3 myPos[BODIES_PER_THREAD];
		float3 myVelo[BODIES_PER_THREAD];

#pragma unroll BODIES_PER_THREAD
		for (unsigned int i = 0; i < BODIES_PER_THREAD; ++i) {
			myPos[i] = positions[tids[i]];
		}

#pragma unroll BODIES_PER_THREAD
		for (unsigned int i = 0; i < BODIES_PER_THREAD; ++i) {
			myVelo[i] = velocities[tids[i]];
		}

		for (unsigned int curTileIdx = 0; curTileIdx < gridDim.x * BODIES_PER_THREAD; ++curTileIdx) {
			int idx = curTileIdx * blockDim.x + threadIdx.x;
			smPos[threadIdx.x].x = positions[idx].x;
			smPos[threadIdx.x].y = positions[idx].y;
			smPos[threadIdx.x].z = positions[idx].z;
			smPos[threadIdx.x].w = masses[idx];

			__syncthreads();
			//compute interactions in our current sharedMemory

#pragma unroll THREADS_PER_BLOCK
			for (unsigned int i = 0; i < THREADS_PER_BLOCK; i++)
			{
#pragma unroll BODIES_PER_THREAD
				for (unsigned int k = 0; k < BODIES_PER_THREAD; ++k) {
					myVelo[k] = smBodyBodyInteractionV2(myPos[k], smPos[i], myVelo[k]);
				}
			}
			__syncthreads();
		}

#pragma unroll BODIES_PER_THREAD
		for (unsigned int i = 0; i < BODIES_PER_THREAD; ++i) {
			myPos[i].x += myVelo[i].x * DTGRAVITY;
			myPos[i].y += myVelo[i].y * DTGRAVITY;
			myPos[i].z += myVelo[i].z * DTGRAVITY;
		}

		__syncthreads();
#pragma unroll BODIES_PER_THREAD
		for (unsigned int i = 0; i < BODIES_PER_THREAD; ++i) {
			positions[tids[i]] = myPos[i];
		}
		__syncthreads();
#pragma unroll BODIES_PER_THREAD
		for (unsigned int i = 0; i < BODIES_PER_THREAD; ++i) {
			velocities[tids[i]] = myVelo[i];
		}
	}

}


////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, copies all the bodies into this class
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
	// num of threads on 1 block, thread layout per block
	blockSize = dim3(THREADS_PER_BLOCK, 1, 1);
	int numBlocks = N / THREADS_PER_BLOCK;
	if (0 != N % blockSize.x) numBlocks++;
	// number of blocks, block layout on grid
	gridSize = dim3(numBlocks, 1, 1);

	//determine thread layout when doing 2 body calculations per thread
	int numBlocksTAO = N / (THREADS_PER_BLOCK * 2);
	if (0 != N % (blockSize.x * 2)) numBlocksTAO++;
	// number of blocks, block layout on grid
	gridSizeTAO = dim3(numBlocksTAO, 1, 1);

	//determine thread layout when doing NUM_BODIES_PER_THREAD body calculations per thread
	int numBlocksXAO = N / (THREADS_PER_BLOCK * BODIES_PER_THREAD);
	if (0 != N % (blockSize.x * BODIES_PER_THREAD)) numBlocksXAO++;
	// number of blocks, block layout on grid
	gridSizeXAO = dim3(numBlocksXAO, 1, 1);

	std::cerr << "num blocks = " << gridSize.x << " :: "
		<< "threads per Block = " << blockSize.x << " :: "
		<< "num blocks tao = " << gridSizeTAO.x << " :: "
		<< "num blocks xao = " << gridSizeXAO.x << std::endl;

	float dtG = G*DT;
	int nTh = THREADS_PER_BLOCK;

	errorCheckCuda(cudaMemcpyToSymbol(Device::EPSILON2, &EPS2, sizeof(float), 0, cudaMemcpyHostToDevice));
	errorCheckCuda(cudaMemcpyToSymbol(Device::DTGRAVITY, &dtG, sizeof(float), 0, cudaMemcpyHostToDevice));
	errorCheckCuda(cudaMemcpyToSymbol(Device::NBODIES, &N, sizeof(int), 0, cudaMemcpyHostToDevice));
	errorCheckCuda(cudaMemcpyToSymbol(Device::NTHREADS, &blockSize.x, sizeof(int), 0, cudaMemcpyHostToDevice));
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
float
Cuda_Computing::computeNewPositions() {
	float time;
	cudaEvent_t start, stop;
	errorCheckCuda(cudaEventCreate(&start));
	errorCheckCuda(cudaEventCreate(&stop));
	errorCheckCuda(cudaEventRecord(start, 0));

	//Device::computeVelocities << < gridSize, blockSize
	//	>> > (Device::positions, Device::masses, Device::velocities);

	//Device::ComputeVelocitiesTao << < gridSizeTAO, blockSize
	//	>> > (Device::positions, Device::masses, Device::velocities);

	//Device::ComputeVelocitiesXao << < gridSizeXAO, blockSize
	//	>> > (Device::positions, Device::masses, Device::velocities);

	//Device::smComputeVelocitiesV1 << < gridSize, blockSize
	//	>> > (Device::positions, Device::masses, Device::velocities);

	//Device::smComputeVelocitiesV3 << < gridSize, blockSize
	//	>> > (Device::positions, Device::masses, Device::velocities);

	//Device::smComputeVelocitiesV3tao << < gridSizeTAO, blockSize
	//	>> > (Device::positions, Device::masses, Device::velocities);

	Device::smComputeVelocitiesV3xao << < gridSizeXAO, blockSize
		>> > (Device::positions, Device::masses, Device::velocities);

	//errorCheckCuda(cudaPeekAtLastError());
	errorCheckCuda(cudaDeviceSynchronize());
	errorCheckCuda(cudaEventRecord(stop, 0));
	errorCheckCuda(cudaEventSynchronize(stop));
	errorCheckCuda(cudaEventElapsedTime(&time, start, stop));

	return time;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// returns the number of bodies 
////////////////////////////////////////////////////////////////////////////////////////////////////
size_t
Cuda_Computing::getSize() const {
	return N;
}
