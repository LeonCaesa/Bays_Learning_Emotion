


#Bays Learning Main Function 
class LaplaceEstimate(object):
    """
    LaplaceEstimate
    """
 
    def __init__(self):
        self.d = {}  # Map[Word-Frequency]
        self.total = 0.0  # Get all frequency
        self.none = 1  #When a word is nonexist, its frequency is one.
 
    def exists(self, key):
        return key in self.d
 
    def getsum(self):
        return self.total
 
    def get(self, key):
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]
 
    def getprob(self, key):
        """
        Estimate the Precondition Possibility
        :param key: words
        :return: possibility
        """
        return float(self.get(key)[1]) / self.total
 
    def samples(self):
        """
        Get all the Samples
        :return:
        """
        return self.d.keys()
 
    def add(self, key, value):
        self.total += value
        if not self.exists(key):
            self.d[key] = 1
            self.total += 1
        self.d[key] += value
 
 
class Bayes(object):
    def __init__(self):
        self.d = {}  # [Tags, Possibility] map
        self.total = 0  # All Word Frequency
 
 
    def train(self, data):
        for d in data:  # d是[[词链表], 标签]
            c = d[1]  # c是分类
            if c not in self.d:
                self.d[c] = LaplaceEstimate()  # d[c] is the possisbility tool
            for word in d[0]:
                self.d[c].add(word, 1)  # Count the Frequency
        self.total = sum(map(lambda x: self.d[x].getsum(), self.d.keys()))
 
    def classify(self, x):
        tmp = {}

        for c in self.d:  # 分类
            tmp[c] = log(self.d[c].getsum()) - log(self.total)  # P(Y=ck)

            for word in x:
                tmp[c] += log(self.d[c].getprob(word))          # P(Xj=xj | Y=ck)

        ret, prob = 0, 0
        for c in self.d:
            now = 0

            try:
                for otherc in self.d:
                    now += exp(tmp[otherc] - tmp[c])            
                now = 1 / now
            except OverflowError:
                now = 0
            if now > prob:
                ret, prob = c, now
        return (ret, prob)
 
 
class Sentiment(object):
    def __init__(self):
        self.classifier = Bayes()
 
    def segment(self, sent):
        words = sent.split(' ')
        return words
 
    def train(self, neg_docs, pos_docs):
        data = []
        for sent in neg_docs:
            data.append([self.segment(sent), u'neg'])
        for sent in pos_docs:
            data.append([self.segment(sent), u'pos'])
        self.classifier.train(data)
 
    def classify(self, sent):
 
        return self.classifier.classify(self.segment(sent))

def Screen(Content,Keywordnum):
    keywords = analyse.extract_tags(Content, topK=Keywordnum)
    keywords =" ".join(keywords)
    s = Sentiment()
    s.train([neg], [pos])
    Screen=s.classify(keywords)
    return Screen
