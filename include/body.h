
#ifndef _BODY_H_
#define _BODY_H_

#define _USE_MATH_DEFINES

#include <vector>
#include <math.h>
#include <iostream>
#include "glm/glm.hpp"

static const float G = 6.67408e-11f;	// gravitational constant
static const float EPS = 3e2f;
static const float EPS2 = EPS*EPS;
static const float DT = 3e-6f;

class Body {

public:

	Body(float m, glm::vec3 p, glm::vec3 v);
	~Body() { };

	void bodyInteraction(Body &p);
	void updatePosition(float dt);

	float mass;
	glm::vec3 position;
	glm::vec3 velocity;

public:
	static void testSystem(std::vector<Body> &bodies, int num);
	static void starSystem1(std::vector<Body> &bodies, int num);
	static void starSystem2(std::vector<Body> &bodies, int num);
	static void starSystem3(std::vector<Body> &bodies, int num);
	static void starSystem4(std::vector<Body> &bodies, int num);
	static void starSystemFlat(std::vector<Body> &bodies, int num);
	static void fillUp(std::vector<Body> &bodies);

private:
	// default constructor
	Body();

};

#endif