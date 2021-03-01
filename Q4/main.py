import matplotlib.pyplot as plt
import numpy as np

def LBP(img) :
	"""function to peform LBP based image feature extraction

	Parameters
	----------
	img : 2D numpy array
		Input image to be processed

	Returns
	-------
	2D numpy array
		LBP processed image
	"""
	neigbhouring_index = [(-1,-1) , (-1,0) , (-1,1) , (0,1) , (1,1) , (1,0) , (1,-1) , (0 ,-1)]			# Eight clockwise neigbours
	LBP_mat = np.empty(img.shape , dtype= np.uint8)

	for i in range(img.shape[0]) :
		for j in range(img.shape[1]) :
			bin_num = ""
			for neigbhour in neigbhouring_index :
				if(0 <= i+neigbhour[0] < img.shape[0] and 0 <= j+neigbhour[1] < img.shape[1]) :
					neigb_val = img[i+neigbhour[0]][j+neigbhour[1]]
				else :
					neigb_val = 0																		# 0 padding
				min_max_ratio = min(neigb_val , img[i][j]) / max(neigb_val , img[i][j] , 1e-05)			# denominator clipping
				bin_num += str(round(min_max_ratio).__int__())
			LBP_mat[i][j] = int(bin_num , 2)															# interger number using eight neigbours
	
	return LBP_mat

if __name__ == '__main__' :
	img = plt.imread('iiitd1.png')
	LBP_img = LBP(img)

	fig = plt.figure()
	plt.imshow(LBP_img , cmap = 'gray')
	plt.title('LBP values map')

	plt.show()