

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include "body.h"
#include "cpu_computing.h"
#include <iostream>
#include <string>
#include <math.h>

#define CPUPARALLEL

void
starGalaxy1(std::vector<Body> &bodies);

void
starGalaxy2(std::vector<Body> &bodies);

int
main()
{
	unsigned const int winWidth = 1280;
	unsigned const int winHeight = 768;

	// get size
	//std::cout << "Max Size: " << SIZE_MAX << std::endl;
	std::vector<Body> bodies;
	starGalaxy2(bodies);
	//computing for cpu
#ifdef CPUPARALLEL
	Cpu_Computing cpu_computer(bodies);
	cpu_computer.setThreads(4);

	const float *positions = cpu_computer.getPositions();
	const size_t sizeBodies = cpu_computer.getSize();
#endif
	// array of colors
	unsigned char *vertexColors = new unsigned char[3 * sizeBodies];
	// zoom factor
	float zoomFactor = 0.1;
	// time between frames
	float dt = 0;
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
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		//zooming
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
			zoomFactor += dt * zoomFactor;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
			zoomFactor -= dt * zoomFactor;

		// clear the screen buffer
		glClearColor(0.1, 0.1, 0.1, 0.1);
		glClear(GL_COLOR_BUFFER_BIT);

		glPushMatrix();

		glTranslatef(winWidth / 3, winHeight / 4, 0);
		glScalef(zoomFactor, zoomFactor, 0);

		glEnableClientState(GL_VERTEX_ARRAY);

		glVertexPointer(3, GL_FLOAT, 0, positions);
		glDrawArrays(GL_POINTS, 0, sizeBodies);

		glDisableClientState(GL_VERTEX_ARRAY);

		glPopMatrix();

		glFlush();
		window.display();

		
		// gravitational updating etc.
#ifdef CPUPARALLEL
		cpu_computer.compute(1e-3f);
#endif

		// time measurement
		dt = elapsedTime.restart().asSeconds();
		window.setTitle(std::to_string(int(1.f / dt)) + " FPS");
	}

	return 0;
}

void
starGalaxy1(std::vector<Body> &bodies)
{
	unsigned const int numOneSideParticles = 7;

	// fill vector body with bodies
	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				float distance = 130;
				Body curBody(2e10, glm::vec3(x * distance, y * distance, z * distance) + glm::vec3(100, 100, 0), glm::vec3(2e14, 0, 0));
				bodies.push_back(curBody);
			}
		}
	}

	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				float distance = 130;
				Body curBody(2e10, glm::vec3(x * distance + numOneSideParticles * 130, y * distance + 1920 + 1920 - numOneSideParticles * 130, -z * distance) + glm::vec3(100, 100, 0), glm::vec3(-2e14, 0, 0));
				bodies.push_back(curBody);
			}
		}
	}

	Body starBody(3e18, glm::vec3(numOneSideParticles * 130, 1920, 0), glm::vec3(0, 0, 0));
	bodies.push_back(starBody);
}

void starGalaxy2(std::vector<Body>& bodies)
{
	unsigned const int numOneSideParticles = 6;

	// fill vector body with bodies
	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				float distance = 130;
				Body curBody(2e10, glm::vec3(x * distance, y * distance, z * distance) + glm::vec3(100, 100, 0), glm::vec3(2e14, 0, 0));
				bodies.push_back(curBody);
			}
		}
	}

	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				float distance = 130;
				Body curBody(2e10, glm::vec3(x * distance + numOneSideParticles * 130, y * distance + 1920 + 1920 - numOneSideParticles * 130, z * distance) + glm::vec3(100, 100, 0), glm::vec3(-2e14, 0, 0));
				bodies.push_back(curBody);
			}
		}
	}

	Body starBody(3e18, glm::vec3(numOneSideParticles * 130, 1920, 0), glm::vec3(0, 0, 0));
	bodies.push_back(starBody);
}
