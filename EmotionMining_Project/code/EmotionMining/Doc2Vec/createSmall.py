filenames = ["Anger_10k" , "Fear_10k", 'Joy_10k', 'Sadness_10k' , 'Surprise_10k']
#newfiles = [ 'Anger_5k' , 'Fear_5k' , 'Joy_1k' , 'Sadness_1k' , 'Surprise_1k']
for i in filenames:	
	file = open (i+".txt", "rb")
	data = file.readlines ()
	file.close ()
	file = open (i.replace ("10", "5")+".txt" , "wb" )
	ctr = 0
	for j in data :
		file.write (j)
		ctr +=1
		if ctr >=5000:
			break
	file.close()