


#include "body.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
// user constructor
////////////////////////////////////////////////////////////////////////////////////////////////////
Body::Body(float m, glm::vec3 p, glm::vec3 v) :
	mass(m), 
	position(p), 
	velocity(v)
{ }


////////////////////////////////////////////////////////////////////////////////////////////////////
// adds the force from body k as velocity to the body
////////////////////////////////////////////////////////////////////////////////////////////////////
void 
Body::bodyInteraction(Body &othBody) {
	glm::vec3 dir = othBody.position - this->position;
	float dist = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;  // L^2-Norm without root ie. dot product
	this->velocity += dir * othBody.mass / sqrt(pow(dist + EPS*EPS,3));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// updates the position of the body in reference to the timespan dt and the gravitational constant
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Body::updatePosition(float dt) {
	this->position += dt * this->velocity * G;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// different body systems for testing
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Body::testSystem(std::vector<Body> &bodies, int num)
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
Body::starSystem1(std::vector<Body> &bodies, int num)
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
Body::starSystem2(std::vector<Body>& bodies, int num)
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
Body::starSystem3(std::vector<Body>& bodies, int num)
{
	unsigned const int numOneSideParticles = 20;
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
Body::starSystem4(std::vector<Body>& bodies, int num)
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

void
Body::starSystemFlat(std::vector<Body>& bodies, int num)
{
	unsigned const int numParticles = num; //65536;
											 // fill vector body with bodies
	for (int i = 0, x = 0, y = 0; i < numParticles; ++i, ++x) {
		if (x > 768) {
			++y;
			x = 0;
		}
		glm::vec3 pos = glm::vec3(x * 30, y * 100, 0) + glm::vec3(400, 400, 0);

		Body curBody1(
			2e17f,
			pos,
			glm::vec3(0, 0, 0)
		);
		bodies.push_back(curBody1);
	}

	fillUp(bodies);
}

void
Body::fillUp(std::vector<Body>& bodies)
{
	int num = bodies.size();
	int rest = 128 - num % 128;

	std::cout << rest << std::endl;

	if (rest != 0) {
		for (int i = 0; i < rest; ++i) {

			glm::vec3 pos = glm::vec3(FLT_MAX, 0, 0);

			Body curBody(
				0,
				pos,
				glm::vec3(0, 0, 0)
			);
			bodies.push_back(curBody);
		}
	}

}