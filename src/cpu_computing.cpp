

#include "cpu_computing.h"


////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, copies all the stuff to this class
////////////////////////////////////////////////////////////////////////////////////////////////////
Cpu_Computing::Cpu_Computing(std::vector<Body> &bodies) : size(bodies.size()){
	this->masses = new float[size];
	this->velocities = new glm::vec3[size];

	for (unsigned int i = 0; i < size; ++i)
	{
		this->positions.push_back(bodies[i].position);

		masses[i] = bodies[i].mass;

		velocities[i] = bodies[i].velocity;
	}

	std::cout << "Cpu_Computing::Cpu_Computing() - Copying of " << size << " bodies done." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// destructor, deletes our dynamic arrays
////////////////////////////////////////////////////////////////////////////////////////////////////
Cpu_Computing::~Cpu_Computing() {
	delete[] masses;
	delete[] velocities;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// determines number of threads according to hardware
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Cpu_Computing::setThreads() {
	// determine available hardware capabilities
	unsigned int nthreads = std::thread::hardware_concurrency();
	if (nthreads == 0) {
		std::cerr << "Couldn't detect hardware capabilities, number of threads set to ONE." << std::endl;
		nthreads = 1;
	}
	setThreads(nthreads);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// set number of threads to an arbitrary value
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Cpu_Computing::setThreads(unsigned int nthreads) {
	this->numThreads = nthreads;

	std::cout << "Cpu_Computing::Cpu_Computing() - Using " << numThreads << " threads." << std::endl;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// spawns threads and waits for all of them to finish
////////////////////////////////////////////////////////////////////////////////////////////////////
void 
Cpu_Computing::compute(float dt) {

	// spawn threads
	std::vector<std::thread> threads;
	for (unsigned int i = 0; i < numThreads; ++i) {
			threads.push_back(std::thread(&Cpu_Computing::computeTile, this, i, std::ref(numThreads), dt));
		
	}

	// wait for all threads to finish
	for (unsigned int i = 0; i < threads.size(); ++i) {
		threads[i].join();
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// computes one tile for n-body
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Cpu_Computing::computeTile(const int tid, const int &num_threads, float dt){
	// calc start & end for i
	const unsigned int i_start = float(size) / num_threads * tid;
	const unsigned int i_end = float(size) / num_threads * (tid + 1);

	// cache for later writing
	glm::vec3 *vel_cache = new glm::vec3[i_end - i_start];

	glm::vec3 dir;

	for (unsigned int i = i_start; i < i_end; ++i)
	{
		for (unsigned int k = 0; k < size; ++k)
		{
			dir.x = positions[k][0] - positions[i][0];
			dir.y = positions[k][1] - positions[i][1];
			dir.z = positions[k][2] - positions[i][2];
			float distSqr = dir.x*dir.x+ dir.y*dir.y+ dir.z*dir.z + EPS2;

			float partForce = masses[k] / sqrt(distSqr*distSqr*distSqr);
			vel_cache[i - i_start] = dir * partForce;
		}
	}
	
	// writing tile to velocities and then to positions
	mutexTiles.lock();
	for (unsigned int i = i_start; i < i_end; ++i)
	{
		velocities[i] += vel_cache[i - i_start];
		positions[i][0] += dt * velocities[i].x * G;
		positions[i][1] += dt * velocities[i].y * G;
		positions[i][2] += dt * velocities[i].z * G;
	}
	mutexTiles.unlock();

	delete[] vel_cache;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// returns positions to be used by openGL
////////////////////////////////////////////////////////////////////////////////////////////////////
const float*
Cpu_Computing::getPositions() const {
	// "converting" vec3 vector to float* 
	return &(positions[0].x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// returns the number of bodies 
////////////////////////////////////////////////////////////////////////////////////////////////////
size_t
Cpu_Computing::getSize() const {
	return size;
}

