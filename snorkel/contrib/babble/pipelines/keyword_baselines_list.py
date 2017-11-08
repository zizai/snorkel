from keyword_baselines import KeywordBaselines
from snorkel.contrib.babble import Explanation, link_explanation_candidates

class KeywordBaselinesList:

    def __init__(self, explanationList, conjunction='and'):
	self.explanationList = explanationList
	self.conjunction=conjunction

    def makeBaselineList(self):
	baselineList = list(self.explanationList) 
	for index, explanation in enumerate(baselineList):
	    baselineList[index]=KeywordBaselines(baselineList[index], self.conjunction).makeBaseline()
	return baselineList
