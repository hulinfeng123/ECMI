from util.structure import new_sparseMatrix
from collections import defaultdict


class Item(object):
    def __init__(self, conf, relation=None):
        self.config = conf
        self.item = {}  # used to store the order of items
        self.relation = relation
        self.followees = defaultdict(dict)
        self.followers = defaultdict(dict)
        self.trustMatrix = self.__generateSet()

    def __generateSet(self):
        triple = []
        for line in self.relation:
            itemId1, itemId2, weight = line
            # add relations to dict
            self.followees[itemId1][itemId2] = weight
            self.followers[itemId2][itemId1] = weight
            # order the user
            if itemId1 not in self.item:
                self.item[itemId1] = len(self.item)
            if itemId2 not in self.item:
                self.item[itemId2] = len(self.item)
            triple.append([self.item[itemId1], self.item[itemId2], weight])
        return new_sparseMatrix.SparseMatrix(triple)

    def row(self, i):
        # return user u's followees
        return self.trustMatrix.row(self.item[i])

    def col(self, i):
        # return user u's followers
        return self.trustMatrix.col(self.item[i])

    def elem(self, i1, i2):
        return self.trustMatrix.elem(i1, i2)

    def weight(self, i1, i2):
        if i1 in self.followees and i2 in self.followees[i1]:
            return self.followees[i1][i2]
        else:
            return 0

    def trustSize(self):
        return self.trustMatrix.size

    def getFollowers(self, i):
        if i in self.followers:
            return self.followers[i]
        else:
            return {}

    def getFollowees(self, i):
        if i in self.followees:
            return self.followees[i]
        else:
            return {}

    def hasFollowee(self, i1, i2):
        if i1 in self.followees:
            if i2 in self.followees[i1]:
                return True
            else:
                return False
        return False

    def hasFollower(self, i1, i2):
        if i1 in self.followers:
            if i2 in self.followers[i1]:
                return True
            else:
                return False
        return False
