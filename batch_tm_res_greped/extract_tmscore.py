import matplotlib.pyplot as plt
import re
import numpy as np

f1 = open("1.txt")
f2 = open("2.txt")
fnames1 = [line.strip() for line in f1.readlines()]
fnames2 = [line.strip() for line in f2.readlines()]
names1 = [file[:5] for file in fnames1]
names2 = [file[:5] for file in fnames2]

i = 0
j = 0
scoresList = []
scores01List = []
for fn1 in fnames1:
    scoreTheFile = []
    score01TheFile = []
    for fn2 in fnames2:
        theFilename = fn1 + "_" + fn2 + ".tmalign_res"
        theFile = open(theFilename)
        theFileContent = theFile.readlines()
        scores = [float(re.findall(r"TM-score = ([\.\d]*)", line)[0]) for line in theFileContent[:2]]
        try:
            scoreTheFile.append(max(scores))
            print("%.3f" % max(scores) + "   ", end="")
        except ValueError:
            print(theFilename)
        if max(scores) > 0.5:
            score01TheFile.append(1)
        else:
            score01TheFile.append(0)
    scoresList.append(scoreTheFile)
    scores01List.append(score01TheFile)
    print()

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(scoresList, cmap='plasma')
im = ax.imshow(scores01List, cmap='Greys')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(len(fnames1)))
ax.set_yticks(np.arange(len(fnames2)))
ax.set_xticklabels(names1)
ax.set_yticklabels(names2)
fig.colorbar(im)

plt.savefig("2.png")

