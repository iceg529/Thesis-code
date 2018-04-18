### part of this script is adapted from Baker et al. A Database and Evaluation Methodology for Optical Flow, to write in the flow format ##
import torchfile   
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('flowfilename', help='output file name')
parser.add_argument('torchfile', help='torch file')
parser.add_argument('width', help='scaled width')
parser.add_argument('height', help='scaled height')
args = parser.parse_args()

floFile = args.flowfilename #flow_sample2_pred
floatCheck = np.float32(202021.25)
width = np.int32(args.width) #128 512  1024
height = np.int32(args.height) #96 384  448
flowSample = torchfile.load(args.torchfile) #flow_sample2_pred
flowCh = np.empty([2, height*width])
flowCh[0] = flowSample[0].flatten()
flowCh[1] = flowSample[1].flatten()
flowArray = flowCh.flatten('F')

flowFinal = np.float32(flowArray)


def writeFloData():	
	with open(floFile, 'w') as f:
		floatCheck.tofile(f)
		width.tofile(f)
		height.tofile(f)
		flowFinal.tofile(f)
		print("The optic flow estimation is saved as %s" % (floFile))

def readFloData1():
	with open('0000000-gt.flo', 'rb') as f:
		magic = np.fromfile(f, np.float32, count=1)
		if 202021.25 != magic:
			print 'Magic number incorrect. Invalid .flo file'
		else:
			w = np.fromfile(f, np.int32, count=1)
			h = np.fromfile(f, np.int32, count=1)
			print 'Reading %d x %d flo file' % (w, h)
			data = np.fromfile(f, np.float32, count=2*w*h)
			print(data.shape)
			# Reshape data into 3D array (columns, rows, bands)
			data2D = np.resize(data, (2, h, w))
			print(data2D.shape)
	return data2D

writeFloData()


