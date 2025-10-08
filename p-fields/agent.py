# MAC0318 Intro to Robotics
# Please fill-in the fields below with your info
#
# Name: Kaiky Henrique Ribeiro Cintra
# NUSP: 13731160
#
# ---
#
# Assignment 7 - Potential Fields
#
# Task:
#  - Write a potential field based agent.
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/duckievillage
#   source bin/activate 
#   python3 assignments/p-fields/agent.py
#
# Submission instructions:
#  0. Add your name and USP number to the file header above.
#  1. Make sure that any last change hasn't broken your code. If the code crashes without running you'll get a 0.
#  2. Submit this file via e-disciplinas.

import sys
import pyglet
import numpy as np
import numpy.linalg
import math
from pyglet.window import key
from duckievillage import create_env
import cv2

def go_mr_duckie(env):
    '''Creates and plops a walking Duckie into the environment's world.'''
    import random
    possible_starts = [[0.88, 0.86], [3.85, 0.89], [3.90, 2.68], [0.80, 3.26]]
    p = possible_starts[random.randint(0, len(possible_starts)-1)]
    return env.add_walking_duckie(p)
mr_duckie = None

def mr_duckie_pos() -> np.ndarray:
    '''Returns the position of Mr. Duckie, the walking duck.'''
    return np.delete(mr_duckie.pos, 1)

def dist(p: np.ndarray, q: np.ndarray) -> float:
    '''Returns the distance between two points.'''
    return np.linalg.norm(p-q)

def line(a: np.ndarray, b: np.ndarray, x: float) -> float:
    '''Line equation function.'''
    return a + x*(b-a)

def dist_obj(p: np.ndarray, o: list) -> (float, np.ndarray):
    '''
    Measures the distance between a point p and an object (polygon) o. The polygonal object is
    represented as a list of vertices. Returns the distance and the nearest point in the frontier
    of o relative to p.
    '''
    a = o[-1]
    m, mo = math.inf, None
    for i in range(len(o)):
        b = o[i]
        t = np.dot(p-a, b-a)/np.dot(b-a, b-a)
        if t < 0:
            d = dist(p, a)
            if m > d: m, mo = d, a
        elif t > 1:
            d = dist(p, b)
            if m > d: m, mo = d, b
        else:
            q = line(a, b, t)
            d = dist(p, q)
            if m > d: m, mo = d, q
        a = b
    return m, np.array(mo)

class Agent:
    # Agent initialization
    def __init__(self, environment):
        ''' Initializes agent '''
        self.env = environment

        self.radius = 0.0318 # R
        self.baseline = environment.unwrapped.wheel_dist/2
        self.motor_gain = 0.68*0.0784739898632288
        self.motor_trim = 0.0007500911693361842

        self.velocity = 0.25
        self.rotation = 0

        self.K_att = 0
        self.K_rep = 0
        self.rho = 0
        self.alpha = 0
        self.epsilon = 0

        self.K_v = 0
        self.K_w = 0
        self.error_history = []

        key_handler = key.KeyStateHandler()
        environment.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline)/self.radius
        return V_l, V_r

    def F_att(self, p: np.ndarray, g: np.ndarray) -> float:
        '''Returns the attraction force applied at position p from goal g.'''
        k_att = 1
        direction = g-p
        return k_att*direction


    def F_rep_vector(self, p: np.ndarray, o: list) -> np.ndarray:
        '''Returns the repulsion force vector applied at position p from object o.'''
        k_rep = 1.5
        rho_0 = 0.3
        
        dist_to_obj, nearest_point = dist_obj(p, o)
        
        if dist_to_obj > rho_0:
            return np.array([0.0, 0.0])
        
        direction_vector = p - nearest_point 
        
        unit_vector = direction_vector / dist_to_obj
        
        magnitude = k_rep * (1.0/dist_to_obj - 1.0/rho_0) * (1.0/dist_to_obj**2)
        
        return magnitude * unit_vector


    def preprocess(self, p: np.ndarray, g: np.ndarray, P: list) -> np.ndarray:
        '''
        Takes the bot's current position p, a goal position g, and a list of polygons P. The
        function should then compute the TOTAL FORCE VECTOR and return it.
        '''
        F_att = self.F_att(p, g)

        F_rep_total = np.array([0.0, 0.0])
        for obstaculo in P:
            F_rep_total += self.F_rep_vector(p, obstaculo) 
        
        F_total = F_att + F_rep_total

        return F_total

    def piRange(self, angulo: float) -> float:
        return (angulo + np.pi) % (2 * np.pi) - np.pi

    def send_commands(self, dt: float):
        ''' Agent control loop '''
        pwm_left, pwm_right = 0, 0

        p = self.env.get_position()
        q = mr_duckie_pos()
        # target position
        a = self.env.cur_angle
        # TODO: compute velocity and rotation using PID controller
        total_force = self.preprocess(p, q, self.env.poly_map.polygons())
        print(f"total force: {total_force}")
        
        distance = np.linalg.norm(total_force)

        if distance > 0:
            angulo_desejado = math.atan2(-total_force[1], total_force[0])
            
            error_theta = self.piRange(angulo_desejado - a)
        else:
            error_theta = 0.0

        Kp_a = 2

        steer = Kp_a*error_theta

        pwm_left, pwm_right = self.get_pwm_control(self.velocity, steer)
        self.env.step(pwm_left, pwm_right)
        self.env.render()

def main():
    print("MAC0318 - Assignment 6")
    env = create_env(
        raw_motor_input = True,
        noisy = True,
        mu_l = 0.007123895,
        mu_r = -0.000523123,
        std_l = 1e-7,
        std_r = 1e-7,
        seed = 101,
        map_name = './maps/catch',
        draw_curve = False,
        draw_bbox = False,
        domain_rand = False,
        user_tile_start = (0, 0),
        distortion = False,
        top_down = False,
        cam_height = 10,
        #is_external_map = True,
        randomize_maps_on_reset = False,
        enable_polymap = True,
    )

    global mr_duckie
    # Mr. Duckie ready for duty.
    mr_duckie = go_mr_duckie(env)

    # env.poly_map.dilate(0.05, True)
    env.reset()
    env.render('human') # show visualization

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE: # exit simulation
            env.close()
            sys.exit(0)
        elif symbol == key.RETURN:  # Reset pose.
            env.reset_pos()

        env.render() # show image to user

    # Instantiate agent
    agent = Agent(env)
    # Call send_commands function from periodically (to simulate processing latency)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    # Now run simulation forever (or until ESC is pressed)
    pyglet.app.run()
    # When it's done, close environment and exit program
    env.close()

if __name__ == '__main__': main()
