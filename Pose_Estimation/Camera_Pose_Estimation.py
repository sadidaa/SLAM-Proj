import numpy as np 
import cv2
import tensorflow as tf

im_size = (480,640)
sigma_p = 0 # Some white noise variance thing
index_matrix = np.dstack(np.meshgrid(np.arange(480),np.arange(640),indexing = 'ij'))

class Keyframe:
	def __init__(self, pose, depth, uncertainty, image):
		self.T = pose # 3x4 transformation matrix
		self.D = depth
		self.U = uncertainty
		self.I = image 

def get_camera_image():
	cam = cv2.VideoCapture(0)
	ret,frame = cam.read()
	#Convert frame to 2D numpy array?

def get_camera_matrix(): #Change to read from camera calib file
	return np.zeros((3,3))

cam_matrix = get_camera_matrix()
cam_matrix_inv = np.linalg.inv(cam_matrix)

def get_cnn_depth(): #To get CNN predicted depth from an image


def find_uncertainty(u,D,D_prev,T):
	u.append(1) #Convert to homogeneous
	V = D_prev * np.matmul(cam_matrix_inv,u) # World point
	u_prop = np.matmul(cam_matrix,T)
	u_prop = np.matmul(u_prop,V)
	u_prop = u_prop/u_prop[2]
	u_prop.pop()
	U = D_prev[u[0]][u[1]] - D[u_prop[0]][u_prop[1]]
	return U**2

def get_uncertainty(frame, prev_keyframe):
	T = np.matmul(np.linalg.inv(frame.T),prev_keyframe.T) #Check if this is right
	find_uncertainty_v = np.vectorize(find_uncertainty)
	U = find_uncertainty_v(index_matrix,frame,prev_keyframe,T) #Check
	return U

def get_initial_uncertainty():


def get_initial_pose():


def get_highgrad_element(img):


def calc_photo_residual(i,frame,cur_keyframe,T):
	i.append(1) # Make i homogeneous
	V = cur_keyframe.D[i[0]][i[1]] * np.matmul(cam_matrix_inv,i) # 3D point
	V.append(1) # Make V homogeneous
	u_prop = np.matmul(T,V) # 3D point in the real world shifted
	u_prop = np.matmul(cam_matrix,u_prop) # 3D point in camera frame
	u_prop = u_prop/u_prop[2] # Projection onto image plane
	u_prop.pop()
	r = (cur_keyframe.I[i[0]][i[1]] - frame.I[u_prop[0]][u_prop[1]]) # Works for single channel image
	return r

def calc_photo_residual_d(u,D,T,frame,cur_keyframe): #For finding the derivative only
	u.append(1)
	V = D*np.matmul(cam_matrix_inv,i)
	V.append(1)
	u_prop = np.matmul(T,V)
	u_prop = np.matmul(cam_matrix,u_prop)
	u_prop = u_prop/u_prop[2]
	u_prop.pop()
	r = cur_keyframe.I[u[0]][u[1]] - frame.I[u_prop[0]][u_prop[1]]
	return r 

def delr_delD(u,frame,cur_keyframe,T):
	D = tf.constant(cur_keyframe.D[u[0]][u[1]])
	r = calc_photo_residual_d(u,D,T,frame,cur_keyframe)
	delr = 0
	with tf.Session() as sess:
		delr = tf.gradients(r,D)
	return delr

def calc_photo_residual_uncertainty(u,frame,cur_keyframe,T):
	deriv = delr_delD()
	sigma = (sigma_p**2 + (deriv**2)*cur_keyframe.U[u[0]][u[1]])**0.5
	return sigma

def huber_norm(x):


def calc_cost_func(u,frame,cur_keyframe,T): # No need for this?
	sum = 0
	for i in u:
		sum = sum + huber_norm(w[i]*calc_photo_residual(i,frame,cur_keyframe,T)/calc_photo_residual_uncertainty(i,frame,cur_keyframe,T))
	return sum

def calc_cost(u,frame,cur_keyframe,T):
	r = []
	for i in u:
		r.append(huber_norm(calc_photo_residual(i,frame,cur_keyframe,T)/calc_photo_residual_uncertainty(i,frame,cur_keyframe,T)))
	return r

def calc_cost_jacobian(u,frame,cur_keyframe,T_s):
	T = np.reshape(T_s,(3,4))
	r = []
	for i in u:
		r.append(huber_norm(calc_photo_residual(i,frame,cur_keyframe,T)/calc_photo_residual_uncertainty(i,frame,cur_keyframe,T)))
	return r

def get_jacobian(dof,u,frame,cur_keyframe,T):
	T_s = T.flatten()
	T_c = tf.constant(T_s)
	r_s = calc_cost_jacobian(u,frame,keyframe,T_c)
	with tf.Session() as sess:
		_,J = tf.run(tf.test.compute_gradient(r_s,(dof,1),T_c,(12,1))) # Returns two jacobians...
	return J

def get_W(dof,stack_r):
	W = np.zeros((dof,dof))
	for i in range(dof):
		W[i][i] = (dof + 1)/(dof + stack_r[i]**2)
	return W

def gauss_newton(dof,u,frame,cur_keyframe):
	ini_T = np.zeros((3,4))
	T = ini_T 
	while(1):
		stack_r = calc_cost(u,frame,cur_keyframe,T)
		J = get_jacobian()
		Jt = J.transpose()
		W = get_W(dof,stack_r) # dof x dof
		hess = np.linalg.inv(np.matmul(np.matmul(Jt,W),J)) # 12x12
		delT = np.matmul(hess,Jt)
		delT = np.matmul(delT,W)
		delT = -np.matmul(delT,stack_r) 
		T = np.dot(delT,T) # Or do subtraction?
		if exit_crit(delT):
			break

def minimize_cost_func(u,frame, cur_keyframe):
	dof = len(u)
	T = gauss_newton(dof,u,frame,cur_keyframe)
	return T

def check_keyframe(T):
	W = np.zeros((12,12)) #Weight Matrix
	threshold = 0
	R = T[:3][:3]
	t = T[3][:3]
	R = R.flatten()
	E = np.concatenate(R,t) # 12 dimensional 	
	temp = matmul(W,E)
	temp = matmul(E.transpose(),temp)
	if temp>=threshold:
		return 1
	else
		return 0

def actual_fuse(u,frame,prev_keyframe):
	u.append(1)
	v_temp = prev_keyframe.D[u[0]][u[1]]*np.matmul(cam_matrix_inv,u)
	v_temp = np.matmul(np.matmul(np.linalg.inv(frame.T),prev_keyframe.T),v_temp)
	v_temp = np.matmul(cam_matrix,v_temp)
	v = v_temp/v_temp[2]
	v.pop()
	u_p = (prev_keyframe.D[v[0]][v[1]]*prev_keyframe.U[v[0]][v[1]]/frame.D[u[0]][u[v]]) + sigma_p
	frame.D[u[0]][u[1]] = (u_p*frame.D[u[0]][u[1]] + frame.U[u[0]][u[1]]*prev_keyframe.D[v[0]][v[1]])/(u_p + frame.U[u[0]][u[1]])
	frame.U[u[0]][u[1]] = u_p*frame.U[u[0]][u[1]]/(u_p + frame.U[u[0]][u[1]])
	return frame.D[u[0]][u[1]],frame.U[u[0]][u[1]]

def fuse_depth_map(frame,prev_keyframe):
	actual_fuse_v = vectorize(actual_fuse)
	frame.D,frame.U = actual_fuse(index_matrix,frame,prev_keyframe)
	return frame.D,frame.U

def refine_depth_map():


def put_delay():


def exit_program():


def main():
	ret,frame = get_camera_image()
	K = [] # Will be a list of keyframe objects
	ini_depth = get_cnn_depth(frame)
	ini_uncertainty = get_initial_uncertainty()
	ini_pose = get_initial_pose()
	K.append(Keyframe(ini_pose,ini_depth,ini_uncertainty,frame)) # First Keyframe appended
	cur_keyframe = K[0]
	cur_index = 0
	while(True):
		while(True):
			ret,frame = get_image()
			if not ret:
				exit_program()
			u = get_highgrad_element(frame) # u consists of a list of points. Where a point is a list of length 2
			T = minimize_cost_func(u,frame,cur_keyframe)
			if check_keyframe(T):
				depth = get_cnn_depth(frame)	
				K.append(Keyframe(T,depth,uncertainty,frame))
				cur_index += 1
				uncertainty = get_uncertainty(K[cur_index],K[cur_index - 1])
				K[cur_index].D,K[cur_index].U = fuse_depth_map(K[cur_index],K[cur_index - 1])
				cur_keyframe = K[cur_index]
				put_delay()
				break
			else:
				refine_depth_map()
				put_delay()

if__name__ == "__main__":
	main()