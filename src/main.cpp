


#define _USE_MATH_DEFINES
#define CUDAPARALLEL

#include <iostream>
#include <string>
#include <math.h>
#include "body.h"
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include "cpu_computing.h"
#include "cuda_computing.cuh"


float
updateAverageFrames(float curFrames);

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

int
main()
{
	unsigned const int winWidth = 1280;
	unsigned const int winHeight = 768;

	// get size
	//std::cout << "Max Size: " << SIZE_MAX << std::endl;
	std::vector<Body> bodies;
	// System
	starSystem3(bodies);
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

	const float *positions = cuda_computer.getPositions();
	const size_t sizeBodies = cuda_computer.getSize();
#endif

	// zoom factor
	float zoomFactor = 0.05f;
	// time between frames
	float dt = 0;
	// translation
	float xTranslation = 10;
	float yTranslation = 10;
	float avgFPS = 0;
	int frameRuns = 0;
	sf::Event event;
	
	/// SFML stuff
	// clock for time keeping
	sf::Clock elapsedTime;

	sf::Window window(sf::VideoMode(winWidth, winHeight), "N-Body Simulation");

	glViewport(0, 0, winWidth, winHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, winWidth, winHeight, 0, -1 * 1e8, 1e8);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPointSize(2);

	elapsedTime.restart();
	// window loop
	while (window.isOpen())
	{

		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}
		/*
		//zooming
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
			
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
			glRotatef(100 * dt * zoomFactor, 0, 1, 0);
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::E))
			glRotatef(-100 * dt * zoomFactor, 0, 1, 0);
			
		}


		// clear the screen buffer
		glClearColor(0.1, 0.1, 0.1, 0.1);
		glClear(GL_COLOR_BUFFER_BIT);

		glPushMatrix();

		glTranslatef(xTranslation, yTranslation, 0);
		glScalef(zoomFactor, zoomFactor, zoomFactor);

		glEnableClientState(GL_VERTEX_ARRAY);

		glVertexPointer(3, GL_FLOAT, 0, positions);
		glDrawArrays(GL_POINTS, 0, sizeBodies);

		glDisableClientState(GL_VERTEX_ARRAY);

		glPopMatrix();

		glFlush();
		window.display();
		*/

		// gravitational updating etc.
#ifdef CPUPARALLEL
		cpu_computer.compute_forces(3e-5f);
#endif
#ifdef CUDAPARALLEL
		//compute forces on cuda
		cuda_computer.computeForces(3e-5f);
#endif

		// time measurement
		dt = elapsedTime.restart().asSeconds();
		float curFPS = 1.f / dt;
		window.setTitle(std::to_string(int(curFPS)) + " FPS");

		//calculating average fps
		++frameRuns;
		avgFPS += (curFPS - avgFPS) / frameRuns;
	}

	std::cout << "Average FPS: " << avgFPS << std::endl;
	getchar();

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

	Body starBody(9e18, glm::vec3(500, 2000, 1000), glm::vec3(0, 0, 0));
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

	Body starBody(1.2e19, glm::vec3(500, 2000, 1000), glm::vec3(0, 0, 0));
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
	unsigned const int numOneSideParticles = 16;
	float speed = 2e15;
	Body starBody(6e19, glm::vec3(500, 2000, 1000), glm::vec3(0, 0, 0));
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

void starSystem4(std::vector<Body>& bodies)
{

	int radius = 2e5;
	float angle = 0.0;
	float angle_stepsize = 0.075;
	int bodies_per_angle = 25;

	// go through all angles from 0 to 2 * PI radians
	while (angle < 2 * M_PI)
	{
		// calculate x, y from a vector with known length and angle
		int x = radius * cos(angle);
		int y = radius * sin(angle);

		for (int i = 1; i <= bodies_per_angle; ++i) {
			Body body(2e16, glm::vec3(x*i/70, y*i/70, 0), glm::vec3(0, 0, 0));
			bodies.push_back(body);
		}
		angle += angle_stepsize;
	}
}

