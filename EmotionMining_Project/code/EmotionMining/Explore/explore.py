def explore(filename):
	lines = open("../Dataset/we_feel_fine/Phrases/"+filename,"rb").readlines()
	print "<<==================================>>"
	print "File : ",filename
	print "Number of phrases:",len(lines)

	

if __name__ == "__main__":
	files = ["ANGER_Phrases.txt","FEAR_Phrases.txt","JOY_Phrases.txt","SADNESS_Phrases.txt","SURPRISE_Phrases.txt"]
	for f in files:
		explore(f)
