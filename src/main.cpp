



#define CUDAPARALLEL

#include <iostream>
#include <string>
#include "cuda_computing.cuh"
#include "cpu_computing.h"

#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>


void
opengl_display(int kernel, int num);

void
benchmark(int kernel, int num);

int
main(int argc, char** argv)
{
	if (argc == 4) {
		std::string cmd = argv[1];
		int kernel = std::stoi(argv[2]);
		int num = std::stoi(argv[3]);

		if (cmd == "bench") {
			std::cout << "Running benchmark! This may take a while! Sit back and relax!" << std::endl;
			benchmark(kernel, num);
		}
		else if (cmd == "disp") {
			opengl_display(kernel, num);
		}
	}
	else {
		std::cout << "Not enough input parameters defined!" << std::endl;
		std::cout << "Benchmarking parameters: 'bench k n'" << std::endl;
		std::cout << "\t k = which kernel to use, if k = 99 it benchmarks all" << std::endl;
		std::cout << "\t n = number of bodies, best use a number that is a power of 2" << std::endl;
		std::cout << "\t Benchmarks the program with n number of bodies." << std::endl;
		std::cout << "Displaying parameters: 'disp k n'" << std::endl;
		std::cout << "\t k = which kernel to use" << std::endl;
		std::cout << "\t n = number of bodies, best use a number that is a power of 2" << std::endl;
		std::cout << "\t Displaying the n bodies in OpenGL." << std::endl;
	}

	return 0;
}


void opengl_display(int kernel, int num)
{
	unsigned int winWidth = 1280;
	unsigned int winHeight = 768;
	// zoom factor
	float zoomFactor = 0.05f;
	// time between frames
	float dt = 0;
	// translation
	float xTranslation = 10;
	float yTranslation = 10;
	//fpscounter etc.
	float avgFPS = 0;
	int frameRuns = 0;
	float curFPS = 0;
	float kernelTime = 0;
	bool isComputing = true;
	// clock for time keeping
	sf::Clock elapsedTime;
	sf::Event event;

	/// SFML/openGL stuff
	sf::Window window(sf::VideoMode(winWidth, winHeight), "N-Body Simulation");
	glewExperimental = GL_TRUE;
	glewInit();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glViewport(0, 0, winWidth, winHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, winWidth, winHeight, 0, -1 * 1e8, 1e8);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glPointSize(2);


	//setup cuda etc.
	std::vector<Body> bodies;
	Body::starSystemFlat(bodies, num); /// system

#ifdef CUDAPARALLEL
	Cuda_Computing cuda_computer(bodies);
	cuda_computer.initDevice();
	cuda_computer.initVertexBuffer();
	const size_t sizeBodies = cuda_computer.getSize();
#else
	Cpu_Computing cpu_computer(bodies);
	cpu_computer.setThreads();

	const float *positions = cpu_computer.getPositions();
	const size_t sizeBodies = cpu_computer.getSize();
#endif

	elapsedTime.restart();
	// window loop
	while (window.isOpen())
	{

		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			if (event.type == sf::Event::Resized) {
				// get the size of the window
				sf::Vector2u size = window.getSize();
				winWidth = size.x;
				winHeight = size.y;
				glViewport(0, 0, winWidth, winHeight);
			}
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
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
				window.close();
			if (!isComputing && sf::Keyboard::isKeyPressed(sf::Keyboard::B))
				isComputing = true;
		}

#if 1	// turn drawing on/off
		// clear the screen buffer
		glClearColor(0.1, 0.1, 0.1, 0.1);
		glClear(GL_COLOR_BUFFER_BIT);

		glPushMatrix();

		glTranslatef(xTranslation, yTranslation, 0);
		glScalef(zoomFactor, zoomFactor, zoomFactor);

		glEnableClientState(GL_VERTEX_ARRAY);

#ifdef CUDAPARALLEL
		glVertexPointer(3, GL_FLOAT, sizeof(float3), 0);
#else
		glVertexPointer(3, GL_FLOAT, 0, positions);
#endif

		glDrawArrays(GL_POINTS, 0, sizeBodies);

		glDisableClientState(GL_VERTEX_ARRAY);

		glPopMatrix();

		glFlush();
		window.display();
#endif

		if (isComputing) {
			/// nbody calculations
#ifdef CUDAPARALLEL
			kernelTime += cuda_computer.computeNewPositions(kernel);
#else
			kernelTime += cpu_computer.compute_forces(DT);
#endif
		}

		// time measurement
		dt = elapsedTime.restart().asSeconds();
		curFPS = 1.f / dt;
		window.setTitle(std::to_string(int(curFPS)) + " FPS");

		if (isComputing) {
			// calculating average fps
			++frameRuns;
			avgFPS += (curFPS - avgFPS) / frameRuns;
		}
	}

	std::cerr << "Average FPS: " << avgFPS << "  ::  gFLOPs from fps: " << avgFPS * sizeBodies * sizeBodies * 19 / 1e9f << std::endl;
	std::cerr << "Average Kernel time: " << kernelTime / frameRuns << "  ::  gFLOPs from time: " << 1000 / (kernelTime / frameRuns) * sizeBodies * sizeBodies * 19 / 1e9f << std::endl;
}

void benchmark(int kernel, int num)
{

	sf::Window window(sf::VideoMode(30, 30), "N-Body Simulation");
	glewExperimental = GL_TRUE;
	glewInit();

	int frameRuns = 40;
	float kernelTime = 0;

	std::vector<Body> bodies;
	Body::starSystemFlat(bodies, num); /// system

	Cuda_Computing cuda_computer(bodies);
	cuda_computer.initDevice();
	cuda_computer.initVertexBuffer();
	const size_t sizeBodies = cuda_computer.getSize();
	
	if (kernel == 99) {
		for (int k = 0; k <= 6; ++k) {
			for (int i = 0; i < frameRuns; ++i) {
				kernelTime += cuda_computer.computeNewPositions(k);
			}
			std::cerr << "Average Kernel time: " << kernelTime / frameRuns << "  ::  gFLOPs: " << 1000 / (kernelTime / frameRuns) * sizeBodies * sizeBodies * 19 / 1e9f << std::endl;
			kernelTime = 0;
		}
	}
	else {
		for (int i = 0; i < frameRuns; ++i) {
			kernelTime += cuda_computer.computeNewPositions(kernel);
		}
		std::cerr << "Average Kernel time: " << kernelTime / frameRuns << "  ::  gFLOPs: " << 1000 / (kernelTime / frameRuns) * sizeBodies * sizeBodies * 19 / 1e9f << std::endl;
		kernelTime = 0;
	}


}

