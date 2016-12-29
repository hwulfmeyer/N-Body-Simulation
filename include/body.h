

#ifndef _BODY_H_
#define _BODY_H_

#include "glm/glm.hpp"
using namespace glm;

static const float G = 6.67408e-11;	// gravitational constant
static float EPS = 4e1;

class Body {

public:

	Body(float m, vec3 p, vec3 v);
	~Body() { };

	void bodyInteraction(Body &p);
	void updatePosition(float dt);

	float mass;
	vec3 position;
	vec3 velocity;

private:
	// default constructor
	Body();

};

#endif