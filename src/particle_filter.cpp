/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <map>

#include "helper_functions.h"
#include "multiv_gauss.h"
#include "map.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::map;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;
  double std_x, std_y, std_theta;  
  
  num_particles = 100;  // TODO: Set the number of particles
  
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];
  
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  for(int i = 0; i < num_particles; i++)
  {
    double sample_x, sample_y, sample_theta;  
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);
    
    Particle current;
    current.id = i;
    current.x = sample_x;
    current.y = sample_y;
    current.theta = sample_theta;
    current.weight = 1.0;
    
    particles.push_back(current);
  }
  is_initialized = true;
  return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  int i = 0;
  for(Particle current: particles)
  {
    double noise_x, noise_y, noise_theta;  
    noise_x = dist_x(gen);
    noise_y = dist_y(gen);
    noise_theta = dist_theta(gen);
    if (fabs(yaw_rate) >= 0.00001) {
      double new_theta = current.theta + yaw_rate * delta_t;
      particles[i].x = noise_x + current.x + ((velocity / yaw_rate) *  (sin(new_theta) - sin(current.theta)));
      particles[i].y = noise_y + current.y + ((velocity / yaw_rate) *  (cos(current.theta) - cos(new_theta)));
      particles[i].theta = noise_theta + new_theta;
    }
    else{
      particles[i].x += noise_x + (velocity * delta_t * cos(current.theta));
      particles[i].y += noise_y + (velocity * delta_t * sin(current.theta));
      particles[i].theta += noise_theta;
    }
    i++;
  }
  return;

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  int i = 0;
  for(LandmarkObs obs: observations)
  {
    double min_distance = -1;
    int id_current = -1;
    
    for(LandmarkObs pre: predicted)
    {
      double distance = dist(obs.x, obs.y, pre.x, pre.y);
      
      if(min_distance == -1 || distance < min_distance)
      {
        min_distance = distance;
        id_current = pre.id;
      }
      
    }
    observations[i].id = id_current;  
    i++;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  
  int i = 0;
  for(Particle current: particles)
  {
    vector<LandmarkObs> predicted_landmarks;
    vector<LandmarkObs> converted_observations;
    double new_w = 1;
    
    for(Map::single_landmark_s lm: map_landmarks.landmark_list)
    {
      if(dist(current.x, current.y, lm.x_f, lm.y_f) <= sensor_range)
        predicted_landmarks.push_back(LandmarkObs{ lm.id_i, lm.x_f, lm.y_f });
      
    }
    
    for(LandmarkObs obs: observations)
    {
      LandmarkObs curr;
      curr.id = obs.id;
      curr.x = current.x + cos(current.theta) * obs.x - sin(current.theta) * obs.y;
      curr.y = current.y + sin(current.theta) * obs.x + cos(current.theta) * obs.y;
      converted_observations.push_back(curr);
    }
    
    dataAssociation(predicted_landmarks, converted_observations);
    
    for(LandmarkObs obs: converted_observations)
    {
      double lm_x, lm_y;
      for(LandmarkObs lm: predicted_landmarks)
      {
        if(obs.id == lm.id)
        {
          lm_x = lm.x;
          lm_y = lm.y;
        }        
      }
      new_w = new_w * ::multiv_prob(std_landmark[0], std_landmark[1], obs.x, obs.y, lm_x, lm_y);
    }
    
    particles[i].weight = new_w;
    
    i++;
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::random_device rd;
  std::mt19937 gen(rd());
  
  std::vector<double> weights;
  for(Particle current: particles) {
    weights.push_back(current.weight);
  }
  
  std::discrete_distribution<> d(weights.begin(), weights.end());
  
  std::map<int, int> m;
  for(int n=0; n<num_particles; ++n) {
    ++m[d(gen)];   
  }
  
  std::vector<Particle> new_particles;
  for(auto p : m) {
    
    for(int n=0; n<p.second; ++n)
    {
      new_particles.push_back(particles[p.first]);
    }
    
  }
  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}