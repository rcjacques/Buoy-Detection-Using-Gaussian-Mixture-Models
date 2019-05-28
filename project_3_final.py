'''
@Authors: Rene Jacques, Zachary Zimits
'''

import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from EM import EM
import copy
import scipy.stats as stats

step = 1
red = (0,0,255)

def breakup(data):
	'''Calculates histogram for input data'''
	data.sort()
	sec = len(data)//3
	hist,bins = np.histogram(data,256,[0,256],density=True)
	return hist

def train(color):
	'''Training function'''
	reds = []
	blues = []
	greens = []

	for i in range(200):
		file_color = color.lower()
		file_name = color+'_Buoys\\'+file_color+str(i)+'.png'
		try: 
			frame = cv2.imread(file_name)

			frame_blue = frame[:,:,0]
			y,x = np.where(frame_blue!=0)
			mean_blue = np.mean(frame_blue[y,x])
			blues.append(mean_blue)

			frame_green = frame[:,:,1]
			y,x = np.where(frame_green!=0)
			mean_green = np.mean(frame_green[y,x])
			greens.append(mean_green)

			frame_red = frame[:,:,2]
			y,x = np.where(frame_red!=0)
			mean_red = np.mean(frame_red[y,x])

			reds.append(mean_red)

			cv2.imshow('data',frame_red)
			cv2.waitKey(1)
		except:
			pass

	#red
	em = EM(reds,100,2)
	red_model = [em[0],em[1]]

	#blue
	em = EM(blues,100,2)
	blue_model = [em[0],em[1]]

	#green
	em = EM(greens,100,2)
	green_model = [em[0],em[1]]

	red_hist = breakup(reds)
	blue_hist = breakup(blues)
	green_hist = breakup(greens)

	model = [blue_model,green_model,red_model]

	return model

def genProbGraph(frame,model,threshold,wr,wg,wb,print_probs=False):
	'''Generate probability image for input frame using input model and parameters'''
	prob_graph = np.zeros([frame.shape[0],frame.shape[1]],np.uint8)

	for i in range(0,frame.shape[0],step):
		for j in range(5,frame.shape[1]-5,step):
			x_red = np.mean(frame[i-5:i+5,j-5:j+5,2])
			x_green = np.mean(frame[i-5:i+5,j-5:j+5,1])
			x_blue = np.mean(frame[i-5:i+5,j-5:j+5,0])

			P = 0
			for n in range(3):
				mus = model[n][0]

				cov = model[n][1]

				for k in range(len(mus)):
					if n == 2:
						P += wr*(1/(np.sqrt(2*np.pi*cov[k])))*np.exp(-(x_red-mus[k])**2/(2*cov[k]**2))
					elif n == 1:
						P += wg*(1/(np.sqrt(2*np.pi*cov[k])))*np.exp(-(x_green-mus[k])**2/(2*cov[k]**2))
					elif n == 0:
						P += wb*(1/(np.sqrt(2*np.pi*cov[k])))*np.exp(-(x_blue-mus[k])**2/(2*cov[k]**2))

			if(P>threshold):
				if print_probs:
					print(P)
				prob_graph[i:i+step,j:j+step] = int(P*255)

	return prob_graph

def findCentroid(points):
	'''Calculate cluster centroid'''
	x_sum,y_sum = 0,0
	for p in points:
		x_sum += p[0]
		y_sum += p[1]

	return (x_sum//len(points),y_sum//len(points))

def findRadius(points,centroid):
	'''Calculate radius from centroid'''
	r = 0
	for p in points:
		dist = np.sqrt((p[0]-centroid[0])**2+(p[1]-centroid[1])**2)
		if dist > r:
			r = int(dist)

	return r

def findPoints(graph):
	'''Find all points in graph with a probability greater than 0'''
	coords = np.where(graph!=0)

	points = []
	for y in coords[0]:
		for x in coords[1]:
			points.append([x,y])

	return points

def findBouy(prob_graph,num):
	'''Iterative probability function: iterate until desired input number of pixels is obtained'''
	temp_image = np.zeros(prob_graph.shape,np.uint8)
	for i in range(100):
		i_dec = i / 100.0
		y,x = np.where(prob_graph/255.0>i_dec)
		
		if(y.shape[0]<num):
			temp_image[y,x] = 255
			break
	return temp_image

def identifyColor(img,model,full_params,local_params,init_threshold):
	'''Generate probability graphs, process those graphs and fit bounding circles to the corresponding models buoy'''
	img_copy = copy.copy(img)

	print('generate prob graph')
	full_prob_graph = genProbGraph(img_copy,model,full_params['threshold'],full_params['r_weight'],full_params['g_weight'],full_params['b_weight'])
	bouy_prob_graph = findBouy(full_prob_graph,init_threshold)

	full = copy.copy(full_prob_graph)
	small = copy.copy(bouy_prob_graph)

	print('finish prob graph')
	
	print('find points',bouy_prob_graph.shape)
	points = findPoints(bouy_prob_graph)
	print('finish find points',len(points))
	cluster = []
	for i in points:
		temp = []
		for j in points:
			d = dist(i,j)
			if d < 60:
				if j not in temp:
					temp.append(j)
		if len(temp) > len(cluster):
			cluster = temp

	print('found clusters',len(cluster))
	if len(cluster) < 20:
		points = cluster
	print('find centroid and radius')
	if len(points) == 0:
		print('Bouy Not in Frame')
		return (0,0),0,full,small

	centroid = findCentroid(points)
	cx = centroid[0]
	cy = centroid[1]
	r = findRadius(points,centroid)
	print('finish find points and radius',centroid,r)

	r = int(2*r)
	r = r if r > 30 else 30
 
	print('generate new prob graph')
	new_prob_graph = genProbGraph(img_copy[centroid[1]-r:centroid[1]+r,centroid[0]-r:centroid[0]+r],model,local_params['threshold'],local_params['r_weight'],local_params['g_weight'],local_params['b_weight'])
	new_prob_graph = findBouy(new_prob_graph,150)
	if new_prob_graph.shape[0] == 0:
		print('empty graph')
		return (0,0),0,full,small
	print('finish new prob graph')

	print('find new points')
	points = findPoints(new_prob_graph)
	print('finish find new points',len(points))

	if len(points) == 0:
		print('Points == 0')
		return (0,0),0,full,small

	center,radius = cv2.minEnclosingCircle(np.array(points))

	print('find new centroid and radius')
	cx2,cy2 = findCentroid(points)
	r_n = findRadius(points,[cx2,cy2])
	print('finish new centroid and radius',cx2,cy2,r_n)
	center = (cx+int(center[0])-r,cy+int(center[1])-r)

	print(full.shape,small.shape)
	return center,int(radius),full,small

def dist(p1,p2):
	'''Straight line distance calculation'''
	return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def main():
	'''Main function'''
	R_model = train('Red')
	G_model = train('Green')
	Y_model = train('Yellow')

	cap = cv2.VideoCapture('detectbuoy.avi')
	frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	count = 0

	while count < frame_count:
		ret,img = cap.read()
		if(count == 35):
			red_center,red_radius,full_red,small_red = identifyColor(img,R_model,{'threshold':0,'r_weight':2,'g_weight':1,'b_weight':1},{'threshold':0,'r_weight':2,'g_weight':2,'b_weight':1},10)

			green_center,green_radius,full_green,small_green = identifyColor(img,G_model,{'threshold':0,'r_weight':2,'g_weight':1,'b_weight':1},{'threshold':0,'r_weight':2,'g_weight':2,'b_weight':1},5)

			yellow_center,yellow_radius,full_yellow,small_yellow = identifyColor(img,Y_model,{'threshold':0,'r_weight':2,'g_weight':1,'b_weight':1},{'threshold':0,'r_weight':2,'g_weight':2,'b_weight':1},10)		

			if (dist(green_center,yellow_center) <= yellow_radius) or (dist(green_center,red_center) <= red_radius and red_radius > green_radius):
				pass
			else:
				cv2.circle(img,green_center,green_radius,(0,255,0),1)

			if dist(red_center,yellow_center) <= yellow_radius or dist(red_center,green_center) <= green_radius:
				pass
			else:
				cv2.circle(img,red_center,red_radius,(0,0,255),1)

			cv2.circle(img,yellow_center,yellow_radius,(0,255,255),1)
			
			cv2.imshow('Buoy Detection',img)

			print('Frame ',count)
			cv2.waitKey(0)
		count += 1
		
if __name__ == "__main__":
	main()