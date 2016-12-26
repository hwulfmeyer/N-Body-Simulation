////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      wulfihm, https://github.com/wulfihm/
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _BODY_H_
#define _BODY_H_

#include "glm/glm.hpp"
using namespace glm;

class Body {

public:

	Body(float m, vec3 p, vec3 v, bool im);
	~Body() { };

	void bodyInteraction(Body &p);
	void updatePosition(float dt);

	float mass;
	vec3 position;
	vec3 velocity;
	bool isMoveable;

private:
	// default constructor
	Body();

};

#endif