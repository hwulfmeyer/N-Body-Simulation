////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      wulfihm, https://github.com/wulfihm/
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include "body.h"
#include <iostream>
#include <string>
#include <math.h>

unsigned static const int winWidth = 1280;
unsigned static const int winHeight = 768;
unsigned static const int numOneSideParticles = 7;

int
main()
{
	// vector for our bodies
	std::vector<Body> bodies;
	// zoom factor
	float zoomFactor = 0.1;
	// time between frames
	float dt = 0;
	// clock for time keeping
	sf::Clock elapsedTime;
	//for pausing
	bool isPaused = true;

	sf::Window window(sf::VideoMode(winWidth, winHeight), "N-Body Simulation");

	glViewport(0, 0, winWidth, winHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, winWidth, winHeight, 0, -1e8, 1e8);

	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPointSize(2);

	// fill vector body with bodies
	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				float distance = 130;
				Body curBody(2e10, vec3(x * distance, y * distance, z * distance) + vec3(100, 100, 0), vec3(2e14, 0, 0), true);
				bodies.push_back(curBody);
			}
		}
	}

	for (int x = 0; x < numOneSideParticles; ++x) {
		for (int y = 0; y < numOneSideParticles; ++y) {
			for (int z = 0; z < numOneSideParticles; ++z) {
				float distance = 130;
				Body curBody(2e10, vec3(x * distance + numOneSideParticles * 130, y * distance + 1920 + 1920-numOneSideParticles*130, -z * distance) + vec3(100, 100, 0), vec3(-2e14, 0, 0), true);
				bodies.push_back(curBody);
			}
		}
	}



	Body starBody(3e18, vec3(numOneSideParticles * 130, 1920, 0), vec3(0, 0, 0), true);
	bodies.push_back(starBody);



	// get size
	std::cout << "Num Particles: " << bodies.size() << std::endl;
	//std::cout << "Max Size: " << SIZE_MAX << std::endl;

	// array of coords
	float *vertexCoords = new float[2 * bodies.size()];
	// array of colors
	unsigned char *vertexColors = new unsigned char[3 * bodies.size()];
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


		//copying the bodies into the vertex arrays
		for (int i = 0; i < bodies.size(); ++i)
		{
			vertexCoords[2 * i] = bodies[i].position.x;
			vertexCoords[2 * i + 1] = bodies[i].position.y;

			vertexColors[3 * i] = 230;
			vertexColors[3 * i + 1] = 230;
			vertexColors[3 * i + 2] = 255;
		}


		glPushMatrix();


		glTranslatef(winWidth / 3, winHeight / 4, 0);
		glScalef(zoomFactor, zoomFactor, 0);


		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glVertexPointer(2, GL_FLOAT, 0, vertexCoords);
		glColorPointer(3, GL_UNSIGNED_BYTE, 0, vertexColors);
		glDrawArrays(GL_POINTS, 0, bodies.size());

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);


		glPopMatrix();

		glFlush();
		window.display();


		// gravitational updating etc.
		if (!isPaused) {
			for (int i = 0; i < bodies.size(); ++i)
			{
				for (int k = 0; k < bodies.size(); ++k)
				{
					bodies[i].bodyInteraction(bodies[k]);
				}
				bodies[i].updatePosition(1e-3f);
			}
		}

		// time measurement
		dt = elapsedTime.restart().asSeconds();
		window.setTitle(std::to_string(int(1.f / dt)) + " FPS");
	}

	return 0;
}