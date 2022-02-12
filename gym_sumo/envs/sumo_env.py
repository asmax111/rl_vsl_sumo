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
import traci.constants as tc

import os, sys, subprocess
action_space= np.array([19, 22, 25, 28, 31, 33])
observation_space = spaces.Tuple([spaces.Box(0,100, shape=(1,), dtype='float'),spaces.Box(0,3000, shape=(1,), dtype='float')])
sumoConfig = "C://PHD/Workspace/gym_sumo_vsl_maroc/gym_sumo/envs/sumo_configs/maroc.sumocfg"
sumoBinary = "C://Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	#Discrete speeds for VSL
	action_space= np.array([19, 22, 25, 28, 31, 33])
	observation_space = spaces.Tuple([spaces.Box(0,100, shape=(1,), dtype='float'),spaces.Box(0,3000, shape=(1,), dtype='float')])
	def __init__(self,mode='gui',simulation_end=3600):
		self.simulation_end = simulation_end
		self.mode = mode
		self.seed()
		self.traci = self.initSimulator(True,8870)
		# Render the simulation
		self.render(self)
		self.sumo_step = 0
		self.episode = 0
		self.flag = True
		#loop detector's names : see sumo additional config file
		self.state_detector1 = ['loop1r1l1','loop1r1l2','loop1r1l3',\
							   'loop2r1l1','loop2r1l2','loop2r1l3',\
							   'loop3r1l1','loop3r1l2','loop3r1l3',\
							   'loop4r1l1','loop4r1l2','loop4r1l3',\
							   'loop6r1l1','loop6r1l2','loop6r1l3',\
							   'loop7r1l1','loop7r1l2','loop7r1l3',\
							   'loop8r1l1','loop8r1l2','loop8r1l3',\
							   'loop9r1l1','loop9r1l2','loop9r1l3',\
							   'loop10r1l1','loop10r1l2','loop10r1l3']
		self.state_detector2 = ['loop1r2l1','loop1r2l2','loop1r2l3',\
							    'loop2r2l1','loop2r2l2','loop2r2l3',\
							    'loop3r2l1','loop3r2l2','loop3r2l3',\
							    'loop4r2l1','loop4r2l2','loop4r2l3',\
							    'loop6r2l1','loop6r2l2','loop6r2l3', \
							    'loop7r2l1','loop7r2l2','loop7r2l3',\
							    'loop8r2l1','loop8r2l2','loop8r2l3', \
							    'loop9r2l1','loop9r2l2','loop9r2l3', \
							    'loop10r2l1','loop10r2l2','loop10r2l3']
		self.VSLlist = ['141131874','141130184']
		self.bottleneck_detector1 = ['loop5r1l1','loop5r1l2','loop5r1l3']
		self.bottleneck_detector2 = ['loop5r2l1','loop5r2l2','loop5r2l3']
		self.inID = ['loop1r1l1','loop1r1l2','loop1r1l3','loop1r2l1','loop1r2l2','loop1r2l3']
		self.outID = ['loop10r1l1','loop10r1l2','loop10r1l3', 'loop10r2l1','loop10r2l2','loop10r2l3']
		self.sumo_running = True
		self.viewer = None	
	
	#Method to set VSL speed limit (VSL per edge)
	def set_vsl(self, v):
		#number_of_edges = len(self.VSLlist)
		#for j in range(number_of_edges):
		#	traci.edge.setMaxSpeed(self.VSLlist[j], v[j])
		traci.edge.setMaxSpeed("141130184", v)

	def calc_bottlespeed1(self):
		speed = []
		edgeSpeed = traci.edge.getLastStepMeanSpeed("141130184")
		for detector in self.bottleneck_detector1:
			dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
			if dspeed < 0:
				dspeed = edgeSpeed                                            
                #The value of no-vehicle signal will affect the value of the reward
			speed.append(dspeed)
		return np.mean(speed)
	def calc_bottlespeed2(self):
		speed = []
		for detector in self.bottleneck_detector2:
			dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
			if dspeed < 0:
				dspeed = 25                                            
                #The value of no-vehicle signal will affect the value of the reward
			speed.append(dspeed)
		return speed
	#####################  a new round simulation  #################### 
	def start_new_simulation(self):
		self.simulation_step = 0
		sumoCmd = [sumoBinary, "-c", sumoConfig, "--start"]
		traci.start(sumoCmd)


	def get_step_state1(self):
		state_occu = []
		state_flow = []
		state = []
		vehicleNumber= traci.edge.getLastStepVehicleNumber("141130184")
		edgeOccupancy= traci.edge.getLastStepOccupancy("141130184")
		#print(f"edge:141130184;  LastStepVehicleNumber: {vehicleNumber}; LastStepOccupancy: {edgeOccupancy}")
		for detector in self.state_detector1:
			occup = traci.inductionloop.getLastStepOccupancy(detector)
			flow =  traci.inductionloop.getLastStepVehicleNumber(detector)
			#print(f"detector 141130184:{detector}:  LastStepOccupancy: {occup}; LastStepVehicleNumber: {flow} ")
			if(occup > 0):
				state_occu.append(occup)
			else:
				state_occu.append(edgeOccupancy)
			if(flow>0):
				state_flow.append(flow)	
			else:
				state_flow.append(vehicleNumber)	
		state.append(np.mean(state_occu))
		state.append(np.mean(state_flow))
		print(f"state= {tuple(state)}")
		observation_space= tuple(state)
		return tuple(state)

	def get_step_state2(self):
		state_occu = []
		vehicleNumber= traci.edge.getLastStepVehicleNumber("141131874")
		print(f"edge:141131874;  LastStepVehicleNumber: {vehicleNumber}")
		for detector in self.state_detector1:
			occup = traci.inductionloop.getLastStepOccupancy(detector)
			if occup == -1:
				occup = 0
			print(f"detector 141131874:{detector}:  LastStepOccupancy: {occup}")
			state_occu.append(occup)
			if(occup > 0):
				state_occu.append(occup)
		return np.mean(np.array(state_occu))
	
	def reset(self):
		"""
        Function to reset the simulation and return the observation
        """
		self.start_new_simulation()
		return self.get_step_state1()
	
	def close(self):
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

	#def calc_outflow(self):
	#	state = []
	#	statef = []
	#	for detector in self.outID:
	#		veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
	#		state.append(veh_num)
	#	for detector in self.inID:
	#		veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
	#		statef.append(veh_num)
	#	return np.sum(np.array(state)) - np.sum(np.array(statef))

	def get_reward(self):
		bspeed = self.calc_bottlespeed1()
		#reward = np.mean(np.array(bspeed))
		reward = bspeed
		print(f"reward: {reward}")
		return reward

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self,v):
		state_overall = 0
		reward = 0
		self.set_vsl(v)
		self.sumo_step +=1
		done = False
		traci.simulationStep()
		self.simulation_step += 1
		state_overall =self.get_step_state1()	
		reward = self.get_reward()
		if reward >= 28:
			done = True
			print(f"done true!")
		return state_overall, reward, done, self.simulation_step

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

