""" standard libs"""
from Crypto.Cipher import AES
from Crypto import Random
import cv2
import numpy as np


key = Random.new().read(AES.block_size)
iv = Random.new().read(AES.block_size)


def encrypt(input_data):
	cfb_cipher = AES.new(key, AES.MODE_CFB, iv)
	enc_data = cfb_cipher.encrypt(input_data)

	enc_file = open("./aes_data/encrypted.enc", "wb")
	enc_file.write(enc_data)
	enc_file.close()





def decrypt(enc_data):
	cfb_decipher = AES.new(key, AES.MODE_CFB, iv)
	plain_data = cfb_decipher.decrypt(enc_data)

	nparr = np.fromstring(plain_data, np.uint8)
	
	img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1


	# gray = cv2.cvtColor(np.float32(plain_data), cv2.COLOR_RGB2GRAY)
	cv2.imwrite(".aes_data/output.png", img_np)
	# output_file = open("./aes_data/output.png", "wb")
	# output_file.write(plain_data)
	# output_file.close()


input_file = open("sample1.gif", 'rb')
input_data = input_file.read()
input_file.close()

encrypt(input_data)

enc_file = open("./aes_data/encrypted.enc", 'rb')
enc_data = enc_file.read()
enc_file.close()

decrypt(enc_data)