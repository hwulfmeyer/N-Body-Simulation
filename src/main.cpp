

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include "body.h"
#include "cpu_computing.h"
#include <iostream>
#include <string>
#include <math.h>

int
main()
{
	unsigned const int winWidth = 1280;
	unsigned const int winHeight = 768;
	unsigned const int numOneSideParticles = 7;

	// vector for our bodies
	std::vector<Body> bodies;

	// fill vector body with bodies
	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				float distance = 130;
				Body curBody(2e10, vec3(x * distance, y * distance, z * distance) + vec3(100, 100, 0), vec3(2e14, 0, 0));
				bodies.push_back(curBody);
			}
		}
	}

	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				float distance = 130;
				Body curBody(2e10, vec3(x * distance + numOneSideParticles * 130, y * distance + 1920 + 1920 - numOneSideParticles * 130, -z * distance) + vec3(100, 100, 0), vec3(-2e14, 0, 0));
				bodies.push_back(curBody);
			}
		}
	}

	Body starBody(3e18, vec3(numOneSideParticles * 130, 1920, 0), vec3(0, 0, 0));
	bodies.push_back(starBody);

	// get size
	//std::cout << "Max Size: " << SIZE_MAX << std::endl;

	//computing for cpu
	Cpu_Computing cpu_computer(bodies);
	cpu_computer.setThreads(2);

	// array of colors
	unsigned char *vertexColors = new unsigned char[3 * bodies.size()];
	// zoom factor
	float zoomFactor = 0.1;
	// time between frames
	float dt = 0;
	// clock for time keeping
	sf::Clock elapsedTime;
	//for pausing
	bool isPaused = false;

	sf::Window window(sf::VideoMode(winWidth, winHeight), "N-Body Simulation");

	glViewport(0, 0, winWidth, winHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, winWidth, winHeight, 0, -1e8, 1e8);

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
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))
			isPaused = true;
		else
			isPaused = false;

		// clear the screen
		glClearColor(0, 0, 0, 0);
		// clear the buffer
		glClear(GL_COLOR_BUFFER_BIT);

		//calculating color
		for (unsigned int i = 0; i < cpu_computer.getSize(); ++i)
		{
			int colorVal = cpu_computer.getPositions()[3 * i + 2]>0?255: cpu_computer.getPositions()[3 * i + 2]<0?127:0;
			vertexColors[3 * i] = colorVal;
			vertexColors[3 * i + 1] = 230;
			vertexColors[3 * i + 2] = colorVal;
		}


		glPushMatrix();


		glTranslatef(winWidth / 3, winHeight / 4, 0);
		glScalef(zoomFactor, zoomFactor, 0);


		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glVertexPointer(3, GL_FLOAT, 0, cpu_computer.positions);
		glColorPointer(3, GL_UNSIGNED_BYTE, 0, vertexColors);
		glDrawArrays(GL_POINTS, 0, cpu_computer.getSize());

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);


		glPopMatrix();

		glFlush();
		window.display();


		// gravitational updating etc.
		if (!isPaused) {
#if 1
			cpu_computer.compute(1e-3f);
#endif
		}

		// time measurement
		dt = elapsedTime.restart().asSeconds();
		window.setTitle(std::to_string(int(1.f / dt)) + " FPS");
	}

	return 0;
}
