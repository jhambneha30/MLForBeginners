import matplotlib.pyplot as plt
D = {'1000':85, '1000,500':95.6,'1000,500,300':92.7,'5000':75.9}
#D = {'1000':79.88, '1000,500':80.66,'1000,500,300':78.68,'5000':73.2}
#plt.plot(list(lr.keys()),list(lr.values()) , marker = 'o')
plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.ylabel("Accuracies")
plt.xlabel("Parameters : Neurons in each layer")
plt.title('Training accuracies using NN')
plt.savefig("Training10k_Accuracies.png")

plt.show()