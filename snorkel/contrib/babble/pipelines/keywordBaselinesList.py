from keywordBaselines import KeywordBaselines
from snorkel.contrib.babble import Explanation, link_explanation_candidates

class KeywordBaselinesList:

    def __init__(self, explanationList, conjunction='and'):
	self.explanationList = explanationList
	self.conjunction=conjunction

    def modifyList(self):
	modifiedList = list(self.explanationList) 
	for index, explanation in enumerate(modifiedList):
	    modifiedList[index]=KeywordBaselines(modifiedList[index], self.conjunction).modify()
	return modifiedList
