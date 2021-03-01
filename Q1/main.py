import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_image_hist(img , title) :
	"""Function to plot the image and noramalized histogam

	Args:
		img (numpy 2D array): This is the input image matrix
		title (string): It is the title to be shown on plot window

	Returns:
		numpy 1D array: It contains the pdf for the input image(i.e. img)
	"""
	pixel_count = np.zeros(2**8)								# intializing the pixel count from zero

	for i in range(img.shape[0]) :								# iterating over the image matrix
		for j in range(img.shape[1]) :
			pixel_count[img[i][j]] += 1							# counting the pixel
	
	pixel_count = pixel_count / sum(pixel_count)				# for normalising histogram or converting count to probability

	fig , axs = plt.subplots(1,2,figsize = (12,5))
	axs[0].imshow(img , cmap = 'gray')							# ploting image
	axs[0].set_title(title + ' imgage')
	axs[1].stem(np.arange(2**8) , pixel_count , linefmt = ':')	# ploting histogram
	axs[1].set_title(title + ' histogram')

	return pixel_count											# returning the pdf

def otsu_thresholding(img) :
	"""Function applies the otsu algorithm for finding the threshold such that foreground and background are seprated
	using the sum of Total sum of squares

	Parameters
	----------
	img : 2D numpy array
		a gray scale image which would be segmentated
	"""
	# intialising the parameters
	hist = show_image_hist(img , 'input')						# diplaying histogram
	pixel_index = np.arange(hist.shape[0])						# pixels index 1D numpy array
	total_pixels = 1											# which is np.sum(hist)
	threshold = 0												# initial trial threshold; at the end : threshold with minumum possible variance
	mean_memo = pixel_index * hist								# it is weigthed pixel values i.e. i * p(i) where i is pixel value (or index)
	min_val = np.inf											# minimum possible sum of variance

	for i in range(1,hist.shape[0]) :									# iterating over historgram
		# mean  for the both side of trial threshold with denominator clipping
		mean_left  = np.sum(mean_memo[:i]) / max(0.00001 , np.sum(hist[:i]))
		mean_right = np.sum(mean_memo[i:]) / max(0.00001 , np.sum(hist[i:]))

		# Total sum of squares for the both side of trial threshold
		TSS_left  = np.power(pixel_index[:i] - mean_left  , 2).sum()
		TSS_right = np.power(pixel_index[i:] - mean_right , 2).sum()

		# calculating sum of Total sum of squares
		val = TSS_left + TSS_right

		if(min_val > val) :										# selecting the minimum sum of Total sum of squares
			min_val = val 
			threshold = i										# storing the corresponding threshold

	return threshold

def segmentate_bg(mask, img) :
	"""function to fill the colors in the image using the mask i.e. color if mask = 1 and blue if mask = 0

	Parameters
	----------
	mask : 2D numpy array
		the mask with 0 and 1 values
	img : 2D numpy array
		RGB colored image

	Returns
	-------
	img
		segmentated colored image
	bounding_box
		list of bounding box coordinate (2 coordinate)
	"""
	bounding_box = [[img.shape[0]*img.shape[1],img.shape[0]*img.shape[1]] , [-1,-1]]								# (min x , min y) , (max x , max y)
	for i in range(img.shape[0]) :								# iterating over the image matrix
		for j in range(img.shape[1]) :
			if(mask[i][j] == 0) :								# background detected
				img[i][j] = 0 # np.array([0,0,255])					# filling blue color
			else :												# finding bounding box cordinates
				bounding_box[0][0] = min(bounding_box[0][0] , i)
				bounding_box[0][1] = min(bounding_box[0][1] , j)
				bounding_box[1][0] = max(bounding_box[1][0] , i)
				bounding_box[1][1] = max(bounding_box[1][1] , j)

	return img , bounding_box									# returning image and bounding box

def avg_border_pixel(img , percentage = 0.1) :
	"""function to give average pixel of the 4 strips (border) from image with ratio to length and breath

	Parameters
	----------
	img : 2D numpy array
		input gray scale image
	percentage : float, optional
		ration of strip and total size (for length and breath), by default 0.1

	Returns
	-------
	int
		average border pixel
	"""
	l = int(img.shape[0] * percentage)							# ratio of length
	b = int(img.shape[1] * percentage)							# ratio of breath

	a  = np.mean(img[:l])										# uper strip
	a += np.mean(img[img.shape[0] - l:])						# down strip
	a += np.mean(img[:,:b])										# left strip
	a += np.mean(img[:,img.shape[1] - b:])						# right strip

	return a // 4												# average of all 4 strips

def avg_center_pixel(img , percentage = 0.1) :
	"""function to give average pixel of central rectangle from image with ratio to length and breath

	Parameters
	----------
	img : 2D numpy array
		input gray scale image
	percentage : float, optional
		ration of dimensions of central rectangle and image (for length and breath), by default 0.1

	Returns
	-------
	int
		average central pixel
	"""
	center = (img.shape[0]//2 , img.shape[1]//2)				# coordinates of the central pixel
	l = int(img.shape[0] * percentage / 2)							# ratio of length
	b = int(img.shape[1] * percentage / 2)							# ratio of breath

	return np.mean(img[center[0] - l : center[0] + l , center[0] - b : center[0] + b]) // 1

if __name__ == "__main__":
	img = plt.imread('iiitd1.png')					# reading the image

	# converting the colored(RBG) image to gray_scale with range [0, 256)
	grayscale_img = (img * 255).astype(dtype = np.int)
	# grayscale_img =  np.dot(img[...,:3],[0.3334,0.3334,0.3334]).astype("int")

	threshold = otsu_thresholding(grayscale_img) ;				# computing otsu threshold
	print('threshold :' , threshold)

	# ------------------------------------------------- assumption choice -------------------------------------------------
	choice = 0
	print('Assumptions\n1) Object will be present at the center of the image.\n2) Boundary pixels are likely to be the background.')
	while(choice not in [1,2]) :
		choice = int(input('Choose one of the assumption to define the background and foreground class (1-2) : '))

	if(choice == 1) :											# using assumption 1
		foreground_pixel = avg_center_pixel(grayscale_img)
		print('average central pixel is ' , foreground_pixel)
		if(foreground_pixel >= threshold) :						# setting background_pixel as it required to make the mask
			background_pixel = 0
		else :
			background_pixel = 255
	if(choice == 2) :											# using assumption 2
		background_pixel = avg_border_pixel(grayscale_img)
		print('average border pixel is ' , background_pixel)

	# genrating the mask
	if(background_pixel < threshold) :							# class 1 (i.e. less than threshold) is background
		grayscale_img[grayscale_img <  threshold] = 0
		grayscale_img[grayscale_img >= threshold] = 1
	else :														# class 1 (i.e. less than threshold) is foreground
		grayscale_img[grayscale_img <  threshold] = 1
		grayscale_img[grayscale_img >= threshold] = 0

	img , box = segmentate_bg(grayscale_img, np.copy(img))		# coloring the image on the bases of mask and genrating the bounding box

	# ------------------------------------------------- showing the image -------------------------------------------------
	fig,ax = plt.subplots(1)
	ax.imshow(img)												# Display the image
	rect = patches.Rectangle((box[0][1],box[0][0]),box[1][1]-box[0][1],box[1][0]-box[0][0],linewidth=3,edgecolor='r',facecolor='none')
	ax.add_patch(rect)											# Adding the rectangular patch to the Axes

	plt.show()	
