

#include "cpu_computing.h"

Cpu_Computing::Cpu_Computing(std::vector<Body> &bodies) : size(bodies.size()){
	this->positions = new float[3 * size];
	this->masses = new float[size];
	this->velocities = new float[3 * size];

	for (unsigned int i = 0; i < size; ++i)
	{
		positions[3 * i] = bodies[i].position.x;
		positions[3 * i + 1] = bodies[i].position.y;
		positions[3 * i + 2] = bodies[i].position.z;

		masses[i] = bodies[i].mass;

		velocities[3 * i] = bodies[i].velocity.x;
		velocities[3 * i + 1] = bodies[i].velocity.y;
		velocities[3 * i + 2] = bodies[i].velocity.z;
	}

	std::cout << "Cpu_Computing::Cpu_Computing() - Copying of " << size << " bodies done." << std::endl;
}

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

void
Cpu_Computing::setThreads(unsigned int nthreads) {
	this->numThreads = nthreads;

	std::cout << "Cpu_Computing::Cpu_Computing() - Using " << numThreads << " threads." << std::endl;
}


void 
Cpu_Computing::compute(float dt) {

	// spawn threads
	std::vector<std::thread> threads;
	for (unsigned int i = 0; i < numThreads; i++) {
			threads.push_back(std::thread(&Cpu_Computing::computeTile, this, i, std::ref(numThreads), dt));
		
	}

	// wait for all threads to finish
	for (unsigned int i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}


void
Cpu_Computing::computeTile(const int tid, const int &num_threads, float dt){
	// calc start & end for i
	unsigned int i_start = float(size) / num_threads * tid;
	unsigned int i_end = float(size) / num_threads * (tid + 1);

	// array cache for later writing
	float *vel_cache = new float[3 * (i_end - i_start)];

	float *dir = new float[3];

	for (unsigned int i = i_start; i < i_end; ++i)
	{
		for (unsigned int k = 0; k < size; ++k)
		{
			dir[0] = positions[3 * k] - positions[3 * i];
			dir[1] = positions[3 * k + 1] - positions[3 * i + 1];
			dir[2] = positions[3 * k + 2] - positions[3 * i + 2];
			float dist = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];

			float force = masses[k] / sqrt(pow(dist + EPS*EPS, 3));
			vel_cache[3 * (i - i_start)] = dir[0] * force;
			vel_cache[3 * (i - i_start) + 1] = dir[1] * force;
			vel_cache[3 * (i - i_start) + 2] = dir[2] * force;
		}
	}

	// writing tile to velocities and then to positions
	mutexTiles.lock();
	for (unsigned int i = i_start; i < i_end; ++i)
	{
		velocities[3 * i] += vel_cache[3 * (i - i_start)];
		velocities[3 * i + 1] += vel_cache[3 * (i - i_start) + 1];
		velocities[3 * i + 2] += vel_cache[3 * (i - i_start) + 2];

		positions[3 * i] += dt * velocities[3 * i] * G;
		positions[3 * i + 1] += dt * velocities[3 * i + 1] * G;
		positions[3 * i + 2] += dt * velocities[3 * i + 2] * G;
	}
	mutexTiles.unlock();

	delete[] vel_cache;
	delete[] dir;
}

float*
Cpu_Computing::getPositions() {
	return positions;
}

size_t
Cpu_Computing::getSize() const {
	return size;
}

