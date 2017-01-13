


#define _USE_MATH_DEFINES
#define CUDAPARALLEL

#include <iostream>
#include <string>
#include <math.h>
#include "cuda_computing.cuh"
#include "cpu_computing.h"

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>



void
testSystem(std::vector<Body> &bodies);

void
starSystem1(std::vector<Body> &bodies);

void
starSystem2(std::vector<Body> &bodies);

void
starSystem3(std::vector<Body> &bodies);

void
starSystem4(std::vector<Body> &bodies);

void
starSystem4flat(std::vector<Body> &bodies);

int
main()
{
	unsigned const int winWidth = 1280;
	unsigned const int winHeight = 768;

	// get size
	//std::cout << "Max Size: " << SIZE_MAX << std::endl;

	// zoom factor
	float zoomFactor = 0.05f;
	// time between frames
	float dt = 0;
	// time keeper for updating
	float timercopying = 0;
	// translation
	float xTranslation = 10;
	float yTranslation = 10;
	float avgFPS = 0;
	int frameRuns = 0;
	float curFPS = 0;


	/// SFML/openGL stuff
	sf::Window window(sf::VideoMode(winWidth, winHeight), "N-Body Simulation");
	glewExperimental = GL_TRUE;
	glewInit();

	std::vector<Body> bodies;
	// System
	starSystem4flat(bodies);
	//computing for cpu
#ifdef CPUPARALLEL
	Cpu_Computing cpu_computer(bodies);
	cpu_computer.setThreads();

	const float *positions = cpu_computer.getPositions();
	const size_t sizeBodies = cpu_computer.getSize();
#endif
#ifdef CUDAPARALLEL
	Cuda_Computing cuda_computer(bodies);
	cuda_computer.initDevice();
	cuda_computer.initDeviceMemory();
	//cuda_computer.initDeviceVertexBuffer();

	const size_t sizeBodies = cuda_computer.getSize();
#endif

	// allocate & register the vertexbuffer
	GLuint vbo;
	cudaGraphicsResource *cuda_vbo_resource;
	// device pointer for opengl/cuda inop
	void* vptr;
	int N = 200;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//vertex contain 3 float coords (x,y,z) and 4 color bytes(RGBA) => total 16 bytes per vertex
	glBufferData(GL_ARRAY_BUFFER, N * 16, NULL, GL_DYNAMIC_COPY);

	//cudaGLRegisterBufferObject(vbo); ///deprecated
	errorCheckCuda(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

	// Map the buffer to CUDA
	//cudaGLMapBufferObject(&vptr, vbo); ///deprecated
	errorCheckCuda(cudaGraphicsMapResources(1, &cuda_vbo_resource));
	size_t numBytes;
	errorCheckCuda(cudaGraphicsResourceGetMappedPointer(&vptr, &numBytes, cuda_vbo_resource));

	// execute kernel creating the data
	//Device::MakeVerticesKernel << < numBlocks, threadsPerBlock >> > (Device::vertexPointer, Device::positions, N);

	// Unmap the buffer
	//cudaGLUnmapBufferObject(vbo); /// deprecated
	errorCheckCuda(cudaGraphicsUnmapResources(1, &cuda_vbo_resource));


	glViewport(0, 0, winWidth, winHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, winWidth, winHeight, 0, -1 * 1e8, 1e8);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPointSize(2);

	// clock for time keeping
	sf::Clock elapsedTime;
	sf::Event event;

	elapsedTime.restart();

	// window loop
	while (window.isOpen())
	{

		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		//input stuff
		if (window.hasFocus()) {
			if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
				zoomFactor += dt * zoomFactor;
			if (sf::Mouse::isButtonPressed(sf::Mouse::Right))
				zoomFactor -= dt * zoomFactor;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
				yTranslation += dt * 200;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
				yTranslation -= dt * 200;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
				xTranslation += dt * 200;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
				xTranslation -= dt * 200;
			if(sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
				window.close();

		}

#if 1	// turn drawing on/off
		// clear the screen buffer
		glClearColor(0.1, 0.1, 0.1, 0.1);
		glClear(GL_COLOR_BUFFER_BIT);

		glPushMatrix();

		glTranslatef(xTranslation, yTranslation, 0);
		glScalef(zoomFactor, zoomFactor, zoomFactor);

		glEnableClientState(GL_VERTEX_ARRAY);

		glVertexPointer(3, GL_FLOAT, 16, 0);
		//glColorPointer(4, GL_UNSIGNED_BYTE, 16, 12);
		glDrawArrays(GL_POINTS, 0, sizeBodies);

		glDisableClientState(GL_VERTEX_ARRAY);

		glPopMatrix();

		glFlush();
		window.display();
#endif

		/// nbody calculations
#ifdef CPUPARALLEL
		cpu_computer.compute_forces(3e-5f);
#endif
#ifdef CUDAPARALLEL
		//compute forces on cuda
		cuda_computer.computeForces(3e-5f);

		//cuda_computer.copyPositionsFromDevice();
#endif

		// time measurement
		dt = elapsedTime.restart().asSeconds();
		curFPS = 1.f / dt;
		window.setTitle(std::to_string(int(curFPS)) + " FPS");
		// calculating average fps
		++frameRuns;
		avgFPS += (curFPS - avgFPS) / frameRuns;
	}

	std::cout << "Average FPS: " << avgFPS << std::endl;
	return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// different body systems for testing
////////////////////////////////////////////////////////////////////////////////////////////////////
void
testSystem(std::vector<Body> &bodies)
{
	unsigned const int numOneSideParticles = 13;

	Body starBody(1e10, glm::vec3(500, 2000, 1000), glm::vec3(0, 0, 0));
	bodies.push_back(starBody);
	float distance = 300;
	// fill vector body with bodies
	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				Body curBody1(
					0,
					starBody.position + glm::vec3(0, 2000, 0) + glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(0, 0, 0)
				);
				bodies.push_back(curBody1);
				Body curBody2(
					0,
					starBody.position - glm::vec3(0, 2000, 0) - glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(0, 0, 0)
				);
				bodies.push_back(curBody2);
			}
		}
	}
}


void
starSystem1(std::vector<Body> &bodies)
{
	unsigned const int numOneSideParticles = 10;

	Body starBody(9e18f, glm::vec3(500, 2000, 1000), glm::vec3(0, 0, 0));
	bodies.push_back(starBody);
	float distance = 300;
	// fill vector body with bodies
	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				Body curBody1(
					2e10,
					starBody.position + glm::vec3(0, 2000, 0) + glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(2e14, 0, 0)
				);
				bodies.push_back(curBody1);
				Body curBody2(
					2e10,
					starBody.position - glm::vec3(0, 2000, 0) - glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(-2e14, 0, 0)
				);
				bodies.push_back(curBody2);
			}
		}
	}
}

void
starSystem2(std::vector<Body>& bodies)
{
	unsigned const int numOneSideParticles = 10;

	Body starBody(1.2e19f, glm::vec3(500, 2000, 1000), glm::vec3(0, 0, 0));
	bodies.push_back(starBody);
	float distance = 300;
	// fill vector body with bodies
	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				Body curBody1(
					2e10,
					starBody.position + glm::vec3(0, 2000, 0) + glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(2e14, 0, 0)
				);
				bodies.push_back(curBody1);
				Body curBody2(
					2e10,
					starBody.position - glm::vec3(0, 2000, 0) - glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(-2e14, 0, 0)
				);
				bodies.push_back(curBody2);
				Body curBody3(
					2e10,
					starBody.position + glm::vec3(2000, numOneSideParticles*-distance, 0) + glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(0, -2e14, 0)
				);
				bodies.push_back(curBody3);
				Body curBody4(
					2e10,
					starBody.position - glm::vec3(2000, numOneSideParticles*-distance, 0) - glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(0, 2e14, 0)
				);
				bodies.push_back(curBody4);
			}
		}
	}
}

void
starSystem3(std::vector<Body>& bodies)
{
	unsigned const int numOneSideParticles = 13;
	float speed = 2e15f;
	Body starBody(6e19f, glm::vec3(500, 2000, 1000), glm::vec3(0, 0, 0));
	bodies.push_back(starBody);
	float distance = 300;
	// fill vector body with bodies
	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				Body curBody1(
					2e10,
					starBody.position + glm::vec3(0, 2000, 0) + glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(speed, 0, 0)
				);
				bodies.push_back(curBody1);
				Body curBody2(
					2e10,
					starBody.position - glm::vec3(0, 2000, 0) - glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(-speed, 0, 0));
				bodies.push_back(curBody2);
				Body curBody3(
					2e10,
					starBody.position + glm::vec3(2000, numOneSideParticles*-distance, 0) + glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(0, -speed, 0)
				);
				bodies.push_back(curBody3);
				Body curBody4(
					2e10,
					starBody.position - glm::vec3(2000, numOneSideParticles*-distance, 0) - glm::vec3(x * distance, y * distance, z * distance),
					glm::vec3(0, speed, 0)
				);
				bodies.push_back(curBody4);
			}
		}
	}


}

void
starSystem4(std::vector<Body>& bodies)
{

	float radius = 2e5f;
	float angle = 0.0f;
	float angle_stepsize = 0.075f;
	int bodies_per_angle = 25;

	// go through all angles from 0 to 2 * PI radians
	while (angle < 2 * M_PI)
	{
		// calculate x, y from a vector with known length and angle
		float x = radius * cos(angle);
		float y = radius * sin(angle);

		for (int i = 1; i <= bodies_per_angle; ++i) {
			Body body(2e16f, glm::vec3(x*i / 70, y*i / 70, 0), glm::vec3(0, 0, 0));
			bodies.push_back(body);
		}
		angle += angle_stepsize;
	}
}

void starSystem4flat(std::vector<Body>& bodies)
{
	unsigned const int numParticles = 2503;
	// fill vector body with bodies
	for (int x = 0; x < numParticles; ++x) {
		Body curBody1(
			2e10f,
			glm::vec3(x,0,0),
			glm::vec3(0, 0, 0)
		);
		bodies.push_back(curBody1);
	}
}

