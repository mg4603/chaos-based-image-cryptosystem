"""user defined libs"""
import helper
import hyper_chaos_algo as hca

"""Standard Libs"""
import cv2
import numpy as np
import skvideo.io
import os
from os.path import isfile, join 

# cap = cv2.VideoCapture("./videos/video.mov", 0)
######################################################################################

 
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('./videos/output.mp4',fourcc, 20.0, (512,512), 0)

# outputfile = "./videos/output.mp4"
# outputfile = "./videos/output2.mp4"
# writer = skvideo.io.FFmpegWriter(outputfile, outputdict={
#   '-vcodec': 'libx264',  #use the h.264 codec
#   '-crf': '0',           #set the constant rate factor to 0, which is lossless
#   '-preset':'veryslow'   #the slower the better compression, in princple, try 
#                          #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
# })


# cap2 = cv2.VideoCapture("./videos/output.mp4",0)
# out2 = cv2.VideoWriter('./videos/output2.mp4', fourcc, 20.0, (512,512), 0)



############################################################################

try:
    if not os.path.exists('data'):
        os.makedirs('data')
    else:
        shutil.rmtree('data')
        os.makedirs('data')
    
except OSError:
    print("Error in creating directory data to store video frames")



x0, u_x, y0, u_y, z0, u_z  = 0.1, 2.6 , 0.1, 0.95, 0.5594449875, 0.53341366688
  
def video_cap():
    camera = PiCamera()
    camera.start_preview()
    camera.start_recording('./videos/video.h264')
    sleep(1)
    camera.stop_recording()
    camera.stop_preview()
    
def video_enc(path, x0, u_x, y0, u_y, z0, u_z):
    cap = cv2.VideoCapture(path, 0)
    currentframe = 0
    
    while True:
        ret, frame = cap.read()
        if ret == True:
            name = './data/frame'+str(currentframe)+'.png'
            b = cv2.resize(frame,(512,512),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            # helper.display_image(b)
            b, u_z, z0, u_x, x0, u_y, y0 = hca.encrypt(b, x0, u_x, y0, u_y, z0, u_z)
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
            # print ('Creating...' + name)
            # helper.display_image(b)
            # b = hca.decrypt(b, u_z, z0, u_x, x0, u_y, y0)
            # helper.display_image(b)
            # writer.writeFrame(b)
            # break
            cv2.imwrite(name, b)
            currentframe += 1
        else:
            print(currentframe)
            break

# currentframe = 0

# while True:
#     ret, frame = cap.read()
#     if ret == True:
#         name = './data/frame'+str(currentframe)+'.png'
#         b = cv2.resize(frame,(512,512),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
#         b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
#         # helper.display_image(b)
#         b, u_z, z0, u_x, x0, u_y, y0 = hca.encrypt(b, x0, u_x, y0, u_y, z0, u_z)
#         b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
#         # print ('Creating...' + name)
#         # helper.display_image(b)
#         # b = hca.decrypt(b, u_z, z0, u_x, x0, u_y, y0)
#         # helper.display_image(b)
#         # writer.writeFrame(b)
#         # break
#         cv2.imwrite(name, b)
#         currentframe += 1
#     else:
#     	print(currentframe)
#     	break
 
########################################################################################### 
# while True:
# 	ret, frame = cap2.read()
# 	print(ret)
# 	if ret == True:
# 		print(1)
# 		b = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 		b = hca.decrypt(b, u_z, z0, u_x, x0, u_y, y0)
# 		# helper.display_image(b)
# 		b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
# 		writer.writeFrame(b)
# 	else:
# 		break  

##########################################################################################

def video_writer(pathIn, pathOut, fps):
	frame_array = []
	files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
	#for sorting the file names properly
	files.sort(key = lambda x: int(x[5:-4]))
	for i in range(len(files)):
		filename=pathIn + files[i]
		
		#reading each files
		img = cv2.imread(filename)
		
		# cv2.imshow('1',img)
		height, width, layers = img.shape
		size = (width,height)

		#inserting the frames into an image array
		frame_array.append(img)
		# cv2.imshow('2',b)
		# cv2.waitKey()
		# break
	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

	for i in range(len(frame_array)):
		print(i)
		# writing to a image array
		out.write(frame_array[i])

	out.release()


def video_decrypt(pathIn, pathOut, fps, u_z, z0, u_x, x0, u_y, y0):

	frame_array = []
	files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
	#for sorting the file names properly
	files.sort(key = lambda x: int(x[5:-4]))
	for i in range(len(files)):
		filename=pathIn + files[i]
		
		#reading each files
		img = cv2.imread(filename)
		
		# cv2.imshow('1',img)
		height, width, layers = img.shape
		size = (width,height)

		b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# print(b.shape)
		b = hca.decrypt(b, u_z, z0, u_x, x0, u_y, y0)
		b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
		#inserting the frames into an image array
		frame_array.append(b)
		# cv2.imshow('2',b)
		# cv2.waitKey()
		# break
	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

	for i in range(len(frame_array)):
		print(i)
		# writing to a image array
		out.write(frame_array[i])

	out.release()

video_writer('./data/','./videos/output_enc.avi', 25)
# video_decrypt('./data/','./videos/output.avi', 25,u_z, z0, u_x, x0, u_y, y0)
video_enc("./videos/video.mov", x0, u_x, y0, u_y, z0, u_z)

##############################################################

# cap.release()
# cap2.release()

# writer.close()

# out.release()
# out2.release()
################################################################
# cv2.destroyAllWindows()

# I_enc, u_z, z0, u_x, x0, u_y, y0 = encrypt(mat,x0, u_x, y0, u_y, z0, u_z)
# helper.display_image(I_enc)

# I_dec = decrypt(I_enc, u_z, z0, u_x, x0, u_y, y0)
# helper.display_image(I_dec)