

#ifndef _BODY_H_
#define _BODY_H_

#include "glm/glm.hpp"

static const float G = 6.67408e-11f;	// gravitational constant
static const float EPS = 3e1f;
static const float EPS2 = EPS*EPS;
static const float DT = 3e-5f;

class Body {

public:

	Body(float m, glm::vec3 p, glm::vec3 v);
	~Body() { };

	void bodyInteraction(Body &p);
	void updatePosition(float dt);

	float mass;
	glm::vec3 position;
	glm::vec3 velocity;

private:
	// default constructor
	Body();

};

#endif