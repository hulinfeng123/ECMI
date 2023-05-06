path = "./ratings.txt"
path2 = "./ratings_45.txt"

f = open(path, "r")
f2 = open(path2, "w")

lines = f.readlines()
for line in lines:
    line = line.split(" ")
    u = line[0]
    i = line[1]
    r = int(line[2])

    if r == 4 or r == 5:
        f2.write(u + " " + i + " 1/n")

f2.close()
f.close()
