import math
import random
import numpy as np
import matplotlib.pyplot as plt


total_ep=100
initial_velocity=5
initial_accn=0
initial_disp=0
accn_data={
    0:0,
    5:2,
    10: 8,
    20:-2,
    40:5,
    45: 9,
    60:-3,
    85:0
} #acceleration will be introduced at different time steps, the dict format is (time step of introduction:magnitude)

true_values=[]
for i in range(0,total_ep):
	initial_disp=initial_disp+initial_velocity+0.5*initial_accn
	
	try:
		initial_accn=accn_data[i]
	except KeyError:
		pass

	initial_velocity=initial_velocity+initial_accn

	true_values.append((initial_accn,initial_disp,initial_velocity))


error=[1.8,700,10]#introducing noise in accn,disp. and velocity respectively
measurements=[] #holds noisy data

for j in true_values:
	a,d,v=j
	intro_error=[np.random.randint(-1*error[0], error[0]),\
	np.random.randint(-1*error[1], error[1]),\
	np.random.randint(-1*error[2],error[2])] #disturb our data extracted from kinematics by intoducing error
	err_accn=a+intro_error[0]
	err_disp=d+intro_error[1] if d+intro_error[1]>0 else 0
	err_velocity=v+intro_error[2]
	measurements.append((err_accn,err_disp,err_velocity))#measurement array now holds noisy data


# plt.subplot(3,1,1)
# plt.plot([i for i in range(0,total_ep)],[y[0] for y in true_values],'b--',label='True Values')
# plt.plot([i for i in range(0,total_ep)],[y[0] for y in measurements],'r--',label='Measurements')
# plt.title('Acceleration')
# plt.legend()
# plt.subplot(3,1,2)
# plt.plot([i for i in range(0,total_ep)],[y[1] for y in true_values],'b--',label='True Values')
# plt.plot([i for i in range(0,total_ep)],[y[1] for y in measurements],'r--',label='Measurements')
# plt.title('Displacement')
# plt.legend()
# plt.subplot(3,1,3)
# plt.plot([i for i in range(0,total_ep)],[y[2] for y in true_values],'b--',label='True Values')
# plt.plot([i for i in range(0,total_ep)],[y[2] for y in measurements],'r--',label='Measurements')
# plt.title('Velocity')
# plt.legend()
#plt.show()

#CONTROL INPUT IS ASSUMED TO BE ZERO



#Kalman Algorithm

X = np.asarray([10,20]) #first estimate belief computation 
Q = np.asarray([[0.004,0.002],[0.002,0.001]]) #Estimate error covariance/process error
A = np.asarray([[1,1],[0,1]]) #Transition matrix. 
R = np.asarray([[0.4,0.01],[0.04,0.01]]) #Measurement error covariance
H = np.asarray([[1,0],[0,1]]) # Matrix from update equation
P = np.asarray([[0,0],[0,0]]) #Covariance of estimate

estimation = []

for k_loop in range(total_ep):
    
    #z_k is the measurement at every step
    z_k = np.asarray([measurements[k_loop][1], measurements[k_loop][2]])
    
    X = A.dot(X) #predict estimate
    P = (A.dot(P)).dot(A.T) + Q #predict error covariance
    
    K = (P.dot(H.T)).dot(np.linalg.inv((H.dot(P).dot(H.T)) + R)) #update Kalman Gain
    X = X + K.dot((z_k - H.dot(X))) #update estimate
    
    P = (np.identity(2) - K.dot(H)).dot(P) #update error covariance
    
    estimation.append((X[0], X[1]))



plt.subplot(3,1,1)
plt.plot([i for i in range(0,total_ep)],[y[1] for y in true_values],'b--',label='True Values')
plt.plot([i for i in range(0,total_ep)],[y[1] for y in measurements],'r--',label='Measurements')
plt.plot([i for i in range(0,total_ep)],[y[0] for y in estimation],'g--',label='Estimated')
plt.title('Displacement')
plt.legend()

plt.subplot(3,1,2)
plt.plot([i for i in range(0,total_ep)],[y[2] for y in true_values],'b--',label='True Values')
plt.plot([i for i in range(0,total_ep)],[y[2] for y in measurements],'r--',label='Measurements')
plt.plot([i for i in range(0,total_ep)],[y[1] for y in estimation],'g--',label='Estimated')
plt.title('Velocity')
plt.legend()
plt.show()







    



    