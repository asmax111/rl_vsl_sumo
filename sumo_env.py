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


class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	def __init__(self,mode='gui',simulation_end=36000,control_horizon = 60, incidents = False):

		self.simulation_end = simulation_end
		self.mode = mode
		self._seed()
		self.traci = self.initSimulator(True,8870)
		self.sumo_step = 0
		self.episode = 0
		self.flag = True
		self.state_detector = ['rout1_011loop','rout1_021loop','rout1_031loop','rout1_111loop','rout1_121loop','rout1_131loop',\
							   'rout1_0111loop','rout1_0211loop','rout1_0311loop', 'rout1_1111loop','rout1_1211loop','rout1_1311loop',\
							   'rout1_0112loop','rout1_0212loop','rout1_0312loop', 'rout1_1112loop','rout1_1212loop','rout1_1312loop']
		self.VSLlist = ['486885259_0','486885259_1','486885259_3','154190805_1','154190805_2','154190805_3']
		self.bottleneck_detector = ['bottlneck1_loop','bottlneck2_loop','bottlneck3_loop','bottlneck11_loop','bottlneck12_loop','bottlneck13_loop']
		self.inID = ['rout1_01loop','rout1_02loop','rout1_03loop','rout1_11loop','rout1_12loop','rout1_13loop']
		self.outID = ['rout1_2222loop','rout1_2223loop','rout1_2224loop','rout1_2225loop','rout1_2226loop','rout1_2227loop']
		self.sumo_running = False
		self.viewer = None	
		self.control_horizon = control_horizon  #seoncs
		self.incidents = incidents
		if self.incidents == False:
			self.incident_time = 2000000
			self.incident_length = 10000000  # it will never happened
		else:
			self.incident_time = np.random.randint(low = 1, high = self.simulation_hour * 3600 - 1800)
			self.incident_length = np.random.randint(low = 1, high = 600)
	
	
	def get_step_state(self):
		state_occu = []
		for detector in self.state_detector:
			occup = traci.inductionloop.getLastStepOccupancy(detector)
			if occup == -1:
				occup = 0
			state_occu.append(occup)
		return np.array(state_occu)

	def set_vsl(self, v):
		number_of_lane = len(self.VSLlist)
		for j in range(number_of_lane):
			traci.setMaxSpeed(self.VSLlist[j], v[j])

	def calc_bottlespeed(self):
		speed = []
		for detector in self.bottleneck_detector:
			dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
			if dspeed < 0:
				dspeed = 5                                              
                #The value of no-vehicle signal will affect the value of the reward
		return np.mean(np.array(speed))

	def _reset(self):
		self.traci.simulationStep() 		# Take a simulation step to initialize car
		traci.close()


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

	def _reward(self):
		bspeed = 0
		oflow = 0
		bspeed = self.calc_bottlespeed()
		oflow = self.calc_outflow()
		reward = oflow/80 * 0.1 + bspeed/(30*self.control_horizon)*0.9
		return reward

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self,v):
		state_overall = 0
		reward = 0
		oflow = 0
		bspeed = 0
		self.set_vsl(v)
		r = 0
		self.sumo_step +=1
		for i in range(self.control_horizon):
			traci.simulationStep()
			self.simulation_step += 1
			if self.simulation_step == self.incident_time:
				vehid = traci.vehicle.getIDList()
				r_tempo = np.random.randint(0, len(vehid) - 1)
				self.inci_veh = vehid[r_tempo]
				self.inci_edge = traci.vehicle.getRoadID(self.inci_veh) # get incident edge
			if self.simulation_step > self.incident_time and self.simulation_step < self.incident_time + self.incident_length:
				traci.vehicle.setSpeed(self.inci_veh, 0)                       # set speed as zero, to simulate incidents
			state_overall = state_overall + self.get_step_state()
			oflow = oflow + self.calc_outflow()
			bspeed = bspeed + self.calc_bottlespeed()
		reward = reward + oflow/80 * 0.1 + bspeed/(30*self.control_horizon)*0.9
		return state_overall/self.control_horizon/100, reward, self.simulation_step, oflow, bspeed/self.control_horizon

	def initSimulator(self,withGUI,portnum):
		# Path to the sumo binary
		if withGUI:
			sumoBinary = "C://Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
		else:
			sumoBinary = "C://Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"

		sumoConfig = "/sumo_configs/maroc.sumocfg"

		# Call the sumo simulator
		sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(portnum), \
			"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
			"--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout, stderr=sys.stderr)

		# Initialize the simulation
		traci.init(portnum)
		return traci

