import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def createPlot(filename):
	lines = open(filename,"rb").readlines()
	title = lines[0]
	values = lines[1:]
	x,pre,rec,f1 = [],[],[],[]
	for val in values:
		val_splitted = val.split(",")
		x1,x2,x3,x4 = str(val_splitted[0]),val_splitted[1],val_splitted[2],val_splitted[3]
		x.append(x1)
		pre.append(x2)
		rec.append(x3)
		f1.append(x4)
	N = len(x)
	ind = np.arange(N)  # the x locations for the groups
	width = 0.25       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, pre, width, color='r')
	rects2 = ax.bar(ind + width, rec, width, color='y')
	rects3 = ax.bar(ind + width + width, f1, width, color='b')

	# add some text for labels, title and axes ticks
	ax.set_ylabel("Accuracy")
	ax.set_xlabel("Feature Extraction Techniques")
	ax.set_title(title)
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(tuple(map(str,x)))
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax.legend((rects1[0], rects2[0],rects3[0]), ('Precision', 'Recall','F1-Score'),loc='center left', bbox_to_anchor=(1, 0.5))
	#plt.show()
	plt.draw()
	plt.savefig(title.replace("\n","")+".png")

createPlot("te_precrecf1.txt")

