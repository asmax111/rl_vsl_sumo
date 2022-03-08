from audioop import avg
from turtle import st
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
#from cv2 import imread,imshow,resize
#import cv2
from collections import namedtuple
import traci.constants as tc

import os, sys, subprocess
action_space= np.array([19, 22, 25, 28, 31, 33])
observation_space = spaces.Box(low=np.array([0,0]), high=np.array([1,1]))
#paths to sumo binary and sumo configs
sumoConfig = "/users/asmae/Desktop/PHD/rl_vsl_sumo/rl_vsl_sumo/gym_sumo/envs/sumo_configs/maroc.sumocfg"
sumoBinary = "/opt/homebrew/opt/sumo/share/sumo/bin/sumo-gui"
class SUMOEnv(Env):
	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
	#Discrete speeds for VSL
	action_space= np.array([19, 22, 25, 28, 31, 33])
	observation_space = spaces.Box(low=np.array([0,0]), high=np.array([1,1]))
	def __init__(self,mode='gui',simulation_end=7200):
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
	def set_vsl(self, v):
		speed = action_space[v]
		print(f"speed chosen = {speed}")
		traci.edge.setMaxSpeed("141131874", action_space[v])
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
		edgeSpeed = traci.edge.getLastStepMeanSpeed("141131874")
		for detector in self.bottleneck_detector2:
			dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
			if dspeed < 0:
				dspeed = edgeSpeed                                            
                #The value of no-vehicle signal will affect the value of the reward
			speed.append(dspeed)
		return np.mean(speed)
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
	# A smaller TTC indicates a more dangerous traffic condition
	def get_average_ttc(self):
		vehicles =  traci.vehicle.getIDList()
		ttc =[]
		for vehicle in vehicles:
			vehcileSpeed = traci.vehicle.getSpeed(vehicle)
			vehicleTtc= traci.vehicle.getParameter(vehicle, "device.ssm.minTTC")
			#print(f"Vehicle id :{vehicle}, speed: {vehcileSpeed}, TTC: {vehicleTtc}")
			if(vehicleTtc != ''):
				if(float(vehicleTtc) < 1000):
					ttc.append(float(vehicleTtc))
		#print(f"TTC: {ttc}")
		return np.mean(ttc)
	def get_step_state2(self):
		state  = []
		state_occu = []
		state_vn= []
		vehicleNumber= traci.edge.getLastStepVehicleNumber("141131874")
		edgeOccupancy= traci.edge.getLastStepOccupancy("141131874")
		#print(f"edge:141131874;  LastStepVehicleNumber: {vehicleNumber}")
		for detector in self.state_detector1:
			occup = traci.inductionloop.getLastStepOccupancy(detector)
			vn = traci.inductionloop.getLastStepVehicleNumber(detector)
			if occup == 0 and edgeOccupancy == 0:
				occup = 1
			elif occup == 0 and edgeOccupancy >0:
				occup = edgeOccupancy
			if vn == 0 and vehicleNumber == 0:
				vn = 1
			elif vn == 0 and vehicleNumber > 0:
				vn = vehicleNumber
			#print(f"detector 141131874:{detector}:  LastStepOccupancy: {occup}")
			#print(f"detector 141131874:{detector}:  getLastStepVehicleNumber: {vn}")
			state_occu.append(occup)
			state_vn.append(vn)
		meanvn = np.mean(np.array(state_vn))
		meanocc= np.mean(np.array(state_occu))
		state.append(meanvn)
		state.append(meanocc)
		#return np.array([meanvn] ,[meanvn])
		return state
	def get_state_cible(self):
		state  = []
		speed = []
		vehicleNumber = []
		flowState = 0
		meanSpeedState = 0
		jamFlow = 0.42 # 3000 vehicle per hour => 0.42 vehicle per 0.5 s
		freeFlowSpeed = 33 #free flow speed

		#vehicleNumber= traci.edge.getLastStepVehicleNumber("141131874")
		for detector in self.state_detector1:
			vn = traci.inductionloop.getLastStepVehicleNumber(detector)
			vehicleNumber.append(vn)
		flowState = np.mean(vehicleNumber)/jamFlow

		for detector in self.bottleneck_detector2:
			dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
			if dspeed > 0:
				speed.append(dspeed)
			else:
				speed.append(0)
		if(np.mean(speed) > 0):
			meanSpeedState = np.mean(speed)/freeFlowSpeed	
		state.append(flowState)
		state.append(meanSpeedState)
		return state	
	def reset(self):
		"""
        Function to reset the simulation and return the observation
        """
		self.start_new_simulation()
		return self.get_state_cible()	
	def close(self):
		traci.close()
	def render(self, mode='gui', close=False):
		if self.mode != "gui":
			raise NotImplementedError("Only rendering in GUI mode is supported")
	def get_reward(self):
		reward = 0
		bspeed = self.calc_bottlespeed2()
		avgTtc = self.get_average_ttc()
		if(bspeed != 0 or bspeed != 1):
			reward = avgTtc
		else:
			reward = avgTtc - avgTtc * 0.1 #penalty of 10%
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
		info={}
		traci.simulationStep()
		self.simulation_step += 1
		state_overall =self.get_state_cible()
		print(f"State : {state_overall}")	
		self.observation_space = state_overall
		reward = self.get_reward()
		if reward > 500:
			done = True
			print(f"done true!")
		return state_overall, reward, done, info
	def initSimulator(self,withGUI,portnum):
		# Path to the sumo binary
		if withGUI:
			sumoBinary = "/opt/homebrew/opt/sumo/share/sumo/bin/sumo-gui"
		else:
			sumoBinary = "/opt/homebrew/opt/sumo/share/sumo/bin/sumo-gui"

		sumoConfig = "/users/asmae/Desktop/PHD/rl_vsl_sumo/rl_vsl_sumo/gym_sumo/envs/sumo_configs/maroc.sumocfg"
		return traci

