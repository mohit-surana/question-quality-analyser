# experiment.py
# Using flipped matrices

EPOCHS = 1000

W = abs(np.random.randn(6, 4))*0.5
inv = linalg.lstsq
alpha = 0.001

for e in range(1, EPOCHS+1):
	if(e % 100 == 0):
		print('Epoch', e)
	for K, C, y_cog, y_know in list(zip(Ks, Cs, Y_cog, Y_know))[:7*len(Ks)//10]:
		temp, _, _, _ =  inv(C.reshape((1,6)), np.array([y_cog + 6 * y_know]).reshape(1,1))
		#Wnew, _, _, _ = inv(temp.T, K.reshape(1,4))
		W_old = W
		K_t = K.reshape((1, 4)).T
		target = temp

		y = W_old.dot(K_t)
		dy = y - target# - y

		dW = dy.dot(K_t.T)
		dK_t = W_old.T.dot(dy) # Not relevant for us

		W = W - alpha * dW

pickle.dump(W, open("models/ADA_W_inv.pkl", 'wb'))


def pred2label(x):
	val = round(x)
	'''
	if(val < (int(val) + 0.5)):
		val = int(val)
	else:
		val = int(val) + 1
	'''
	if(val < 0):
		val = 0
	if(val > 23):
		val = 23
	return int(val)


targets = [y_cog + 6 * y_know for y_cog, y_know in zip(Y_cog, Y_know)]
predictions = []

correct = 0
total = 0

if(TEST):
	for K, C, y in list(zip(Ks, Cs, targets))[7*len(Ks)//10:]:
		prediction = np.dot(np.dot(C.reshape(1, 6), W), K.reshape(1, 4).T)
		prediction = pred2label(prediction[0][0])
		print(prediction, y)
		if(prediction == y):
			correct += 1
		total += 1
	print('Accuracy:', (correct / total) )
