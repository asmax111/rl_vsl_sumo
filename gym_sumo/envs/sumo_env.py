from gym import Env
from gym import error, spaces, utils
from gym.utils import seeding
import traci
import traci.constants as tc
from matplotlib.pyplot import imread
from gym import spaces
from string import Template
import numpy as np
import math
import time
from cv2 import imread,imshow,resize
import cv2
from collections import namedtuple

import os, sys, subprocess

sumoConfig = "C://PHD/Workspace/gym_sumo_vsl_maroc/gym_sumo/envs/sumo_configs/maroc.sumocfg"
sumoBinary = "C://Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	action_space= np.array([[19,19,19], [22,22,22], [25,25,25], [28,28,28], [31,31,31], [33,33,33]])
	def __init__(self,mode='gui',simulation_end=3600,control_horizon = 60):

		self.simulation_end = simulation_end
		self.mode = mode
		self.seed()
		self.traci = self.initSimulator(True,8870)
		# Render the simulation
		self.render(self)
		self.sumo_step = 0
		self.episode = 0
		self.flag = True
		self.state_detector = ['loop1r1l1','loop1r1l2','loop1r1l3','loop1r2l1','loop1r2l2','loop1r2l3',\
							   'loop2r1l1','loop2r1l2','loop2r1l3', 'loop2r2l1','loop2r2l2','loop2r2l3',\
							   'loop3r1l1','loop3r1l2','loop3r1l3', 'loop3r2l1','loop3r2l2','loop3r2l3',\
							   'loop4r1l1','loop4r1l2','loop4r1l3', 'loop4r2l1','loop4r2l2','loop4r2l3',\
							   'loop6r1l1','loop6r1l2','loop6r1l3', 'loop6r2l1','loop6r2l2','loop6r2l3', \
							   'loop7r1l1','loop7r1l2','loop7r1l3', 'loop7r2l1','loop7r2l2','loop7r2l3',\
							   'loop8r1l1','loop8r1l2','loop8r1l3', 'loop8r2l1','loop8r2l2','loop8r2l3', \
							   'loop9r1l1','loop9r1l2','loop9r1l3', 'loop9r2l1','loop9r2l2','loop9r2l3', \
							   'loop10r1l1','loop10r1l2','loop10r1l3', 'loop10r2l1','loop10r2l2','loop10r2l3']
		self.VSLlist = ['141131874','141130184']
		self.bottleneck_detector = ['loop5r1l1','loop5r1l2','loop5r1l3','loop5r2l1','loop5r2l2','loop5r2l3']
		self.inID = ['loop1r1l1','loop1r1l2','loop1r1l3','loop1r2l1','loop1r2l2','loop1r2l3']
		self.outID = ['loop10r1l1','loop10r1l2','loop10r1l3', 'loop10r2l1','loop10r2l2','loop10r2l3']
		self.sumo_running = True
		self.viewer = None	
		self.control_horizon = control_horizon  #seoncs	
	
	def set_vsl(self, v):
		number_of_edges = len(self.VSLlist)
		for j in range(number_of_edges):
			traci.edge.setMaxSpeed(self.VSLlist[j], v[j])

	def calc_bottlespeed(self):
		speed = []
		for detector in self.bottleneck_detector:
			dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
			print(f"dspeed: {dspeed}")
			if dspeed < 0:
				dspeed = 5                                              
                #The value of no-vehicle signal will affect the value of the reward
		return dspeed

	#####################  a new round simulation  #################### 
	def start_new_simulation(self):
		self.simulation_step = 0
		sumoCmd = [self.sumoBinary, "--step-length",
            "1800", "-c", sumoConfig, "--start"]
		traci.start(sumoCmd)

	def get_step_state(self):
		state_occu = []
		for detector in self.state_detector:
			occup = traci.inductionloop.getLastStepOccupancy(detector)
			if occup == -1:
				occup = 0
			state_occu.append(occup)
		return np.array(state_occu)
	
	def reset(self):
		"""
        Function to reset the simulation and return the observation
        """
		self.simulation_step = 0
		sumoCmd = [sumoBinary, "-c", sumoConfig, "--start"]
		traci.start(sumoCmd)
		#self.traci.simulationStep() 
		state_occu = []
		for detector in self.state_detector:
			occup = traci.inductionloop.getLastStepOccupancy(detector)
			if occup == -1:
				occup = 0
			state_occu.append(occup)		
		return np.array(state_occu)

	def render(self, mode='gui', close=False):

		if self.mode == "gui":
			img = imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),'sumo.png'), 1)
			if mode == 'rgb_array':
				return img
			elif mode == 'human':
				from gym.envs.classic_control import rendering
				if self.viewer is None:
					self.viewer = rendering.SimpleImageViewer()
				self.viewer.imshow(img)
		else:
			raise NotImplementedError("Only rendering in GUI mode is supported")

	def calc_outflow(self):
		state = []
		statef = []
		for detector in self.outID:
			veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
			state.append(veh_num)
		for detector in self.inID:
			veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
			statef.append(veh_num)
		return np.sum(np.array(state)) - np.sum(np.array(statef))

	def reward(self):
		bspeed = 0
		oflow = 0
		bspeed = self.calc_bottlespeed()
		oflow = self.calc_outflow()
		reward = oflow/80 * 0.1 + bspeed/(30*self.control_horizon)*0.9
		return reward

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self,v):
		state_overall = 0
		reward = 0
		oflow = 0
		bspeed = 0
		self.set_vsl(v)
		r = 0
		self.sumo_step +=1
		done = False
		for i in range(self.control_horizon):
			traci.simulationStep()
			self.simulation_step += 1
			state_overall = state_overall + self.get_step_state()
			oflow = oflow + self.calc_outflow()
			print(f"oflow: {oflow}")
			bspeed = self.calc_bottlespeed()
			print(f"botlneck speed: {bspeed}")
		reward = reward + oflow/80 * 0.1 + bspeed/(30*self.control_horizon)*0.9
		if bspeed >= 30:
			done = True
			print(f"done true!")
		return state_overall/self.control_horizon/100, reward, done, self.simulation_step

	def initSimulator(self,withGUI,portnum):
		# Path to the sumo binary
		if withGUI:
			sumoBinary = "C://Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
		else:
			sumoBinary = "C://Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"

		sumoConfig = "C://PHD/Workspace/gym_sumo_vsl_maroc/gym_sumo/envs/sumo_configs/maroc.sumocfg"

		# Call the sumo simulator
		#from subprocess import run, PIPE
		
		#sumoProcess = subprocess.run([sumoBinary, "-c", sumoConfig], stdout=PIPE)
		#traci.start([sumoBinary, "-c", sumoConfig], label= "sim")
		# Initialize the simulation
		#traci.init(portnum)
		# run a single step
		#traci.simulationStep()
		return traci

