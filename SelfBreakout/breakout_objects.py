# Objects

import numpy as np

class Object():

	def __init__(self, pos, attribute):
		self.pos = pos
		self.width = 0
		self.height = 0
		self.attribute = attribute

	def getBB(self):
		return [self.pos[0], self.pos[1], self.pos[0] + self.height, self.pos[1] + self.width]

	def getMidpoint(self):
		return [self.pos[0] + (self.height / 2), self.pos[1]  + (self.width/2)]

	def getAttribute(self):
		return self.attribute

	def getState(self):
		return self.getMidpoint() + [self.attribute]

	def interact(self, other):
		pass

class animateObject(Object):

	def __init__(self, pos, attribute, vel):
		super(animateObject, self).__init__(pos, attribute)
		self.vel = vel
		self.apply_move = True

	def move(self):
		# print (self.pos, self.vel)
		if self.apply_move:
			self.pos += self.vel
		else:
			self.apply_move = True
			self.pos += self.vel

def intersection(a, b):
	midax, miday = (a.next_pos[1] * 2 + a.width)/ 2, (a.next_pos[0] * 2 + a.height)/ 2
	midbx, midby = (b.pos[1] * 2 + b.width)/ 2, (b.pos[0] * 2 + b.height)/ 2
	# print (midax, miday, midbx, midby)
	return (abs(midax - midbx) * 2 < (a.width + b.width)) and (abs(miday - midby) * 2 < (a.height + b.height))

class Ball(animateObject):
	def __init__(self, pos, attribute, vel):
		super(Ball, self).__init__(pos, attribute, vel)
		self.width = 2
		self.height = 2
		self.name = "Ball"
		self.losses = 0
		self.paddlehits = 0
		# self.nohit_delay = 0

	def interact(self, other):
		'''
		interaction computed before movement
		'''
		self.next_pos = self.pos + self.vel
		# print(self.apply_move, self.vel)
		if intersection(self, other) and self.apply_move:
			if other.name == "Paddle":
				rel_x = self.pos[1] - other.pos[1]
				if rel_x == -2:
					self.vel = np.array([-1, -1])
				elif rel_x == -1:
					self.vel = np.array([-1, -1])
				elif rel_x == 0:
					self.vel = np.array([-1, -1])
				elif rel_x == 1:
					self.vel = np.array([-2, -1])
				elif rel_x == 2:
					self.vel = np.array([-2, -1])
				elif rel_x == 3:
					self.vel = np.array([-2, 1])
				elif rel_x == 4:
					self.vel = np.array([-2, 1])
				elif rel_x == 5:
					self.vel = np.array([-1, 1])
				elif rel_x == 6:
					self.vel = np.array([-1, 1])
				elif rel_x == 7:
					self.vel = np.array([-1, 1])
				self.apply_move = False
				self.paddlehits += 1
			elif other.name.find("SideWall") != -1:
				self.vel = np.array([self.vel[0], -self.vel[1]])
				self.apply_move = False
			elif other.name.find("TopWall") != -1:
				self.vel = np.array([-self.vel[0], self.vel[1]])
				self.apply_move = False
			elif other.name.find("BottomWall") != -1:
				self.pos = np.array([46, np.random.randint(20, 36)])
				self.vel = np.array([1, np.random.choice([-1,1])])
				# self.pos = np.array([46, 24])
				# self.vel = np.array([1, 1])
				self.apply_move = False
				self.losses += 1
			elif other.name.find("Block") != -1:
				if other.attribute == 1:
					rel_x = self.pos[1] - other.pos[1]
					rel_y = self.pos[0] - other.pos[0]
					print(rel_x, rel_y, self.vel, other.name, intersection(self, other))
					other.attribute = 0
					next_vel = self.vel
					if rel_y == -2 or rel_y == 3 or rel_y == 2:
						next_vel[0] = - self.vel[0]
					# else:
					# 	if rel_x == -2 or rel_x == 4 or (rel_x == 3 and rel_y != -2):
					# 		next_vel[1] = - self.vel[1]
					self.vel = np.array(next_vel)
					self.apply_move = False
					# self.nohit_delay = 2

class Paddle(animateObject):
	def __init__(self, pos, attribute, vel):
		super(Paddle, self).__init__(pos, attribute, vel)
		self.width = 7
		self.height = 2
		self.name = "Paddle"
		self.nowall = False

	def interact(self, other):
		if other.name == "Action":
			# print ("action", other.attribute)
			if other.attribute == 0 or other.attribute == 1:
				self.vel = np.array([0,0])
			elif other.attribute == 2:
				if self.pos[1] == 12:
					if self.nowall:
						self.pos = np.array([0,64])
					self.vel = np.array([0,0])
				else:
					self.vel = np.array([0,-2])
				# self.vel = np.array([0,-2])
			elif other.attribute == 3:
				if self.pos[1] >= 64:
					if self.nowall:
						self.pos = np.array([0,12])
					self.vel = np.array([0,0])
				else:
					self.vel = np.array([0,2])
				# self.vel = np.array([0,2])

class Wall(Object):
	def __init__(self, pos, attribute, side):
		super(Wall, self).__init__(pos, attribute)
		if side == "Top":
			self.width = 84
			self.height = 4
		elif side == "RightSide":
			self.width = 4
			self.height = 84
		elif side == "LeftSide":
			self.width = 4
			self.height = 84
		elif side == "Bottom":
			self.width = 84
			self.height = 4
		self.name = side + "Wall"

class Block(Object):
	def __init__(self, pos, attribute, index):
		super(Block, self).__init__(pos, attribute)
		self.width = 3
		self.height = 2
		self.name = "Block" + str(index)

	# def interact(self, other):
	# 	if other.name == "Ball":
	# 		if intersection(other, self):
	# 			print(self.name, self.pos, other.pos)
	# 			self.attribute = 0

class Action(Object):
	def __init__(self, pos, attribute):
		super(Action, self).__init__(pos, attribute)
		self.width = 0
		self.height = 0
		self.name = "Action"

	def take_action(self, action):
		self.attribute = action

	def interact (self, other):
		pass