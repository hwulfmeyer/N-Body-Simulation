# N-Body Simulation

For parallel computing on CPU & CUDA.

### Simulation Setup

The classical N-body problem simulates the evolution of a system of N bodies, where the force exerted on each body arises due to its interaction with all the other bodies in the system. It is using newtons gravitational laws.


Infomaterial:
* http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html
* http://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf
* http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
* http://www.cs.cmu.edu/~scandal/alg/nbody.html
* http://physics.princeton.edu/~fpretori/Nbody/intro.htm
* http://www.cs.princeton.edu/courses/archive/spr15/cos126/assignments/nbody.html


### The Algorithm

The naive implementation computates each force for each object from all objects. It's called all-pairs-agorithm or brute force method and has a complexity of O(N^2), however it gives the most correct results.

See https://en.wikipedia.org/wiki/N-body_problem#General_formulation


Pseudocode calculating netforce for each body:

    G = gravitational constant;
    for each Body i {    
      for each Body k where k!=i {
        vec3 direction = k.position - i.position;
        double dist = norm(direction);  // L^2-Norm
        i.force += (G * k.mass * i.mass / dist^3) * direction;
      }
    }

Updating position & velocity for each body:
    
    dt = timespan;
    for each Body i {    
      i.velocity += i.force / i.mass;
      i.position += dt * i.velocity;
    }

With softening factor & simplification:

    G = gravitational constant;
    epsilon = softening factor;     // epsilon^2 > 0
    for each Body i {
      i.velocity = (0,0,0);
      for each Body k {
        vec3 direction = k.position - i.position;
        double dist = norm2(direction);  // dot product (l2 norm squared)
        i.velocity += k.mass*direction  / (dist + epsilon^2)^(3/2);
      }
      i.velocity = G * i.velocity;
    }
    dt = timespan;
    for each Body i {
      i.position += dt * i.velocity;
    }


### Implementation

Drawing is done with OpenGL.

Libraries (https://www.opengl.org/wiki/Related_toolkits_and_APIs)

UI: http://anttweakbar.sourceforge.net/doc/

## Project Timeline
1. Program on single threaded CPU       [DONE]
2. porting to multithreaded CPU         [DONE]
3. naive algorithm on CUDA              [DONE]
4. utilize OpenGL & Cuda inoperability  [DONE]
5. CUDA performance optimizations       [TODO]
6. create better & 3D particles/world   [TODO]
7. integrate AntTweakBar                [TODO]

### Algorithm Optimization/Approximation methods

https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation




