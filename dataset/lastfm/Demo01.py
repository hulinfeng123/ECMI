path = "./ratings.txt"
path2 = "./ratings.txt"

f = open(path, "r")
f2 = open(path2, "w")

w = f.readlines()
for line in w:
    tmp = line.split("\t")
    f2.write(tmp[0] + " ")
    f2.write(tmp[1] + " ")
    f2.write(tmp[2])

f2.close()
f.close()
