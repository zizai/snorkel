from keywordBaselines import KeywordBaselines
from snorkel.contrib.babble import Explanation, link_explanation_candidates
import re

class KeywordBaselineLists(KeywordBaselines):

    def __init__(self, explanationList):
	self.explanationList = explanationList

    def modifyList(self,explanationList):
	modifiedList = list(explanationList) 
	for i, explanation in explanationList:
		keyword=explanation.findKeywords()
		modified[i]=keyword[0]
	return modifiedList

testExp = Explanation(
        name='LF_spouse_to_left',
        condition="the word 'wife' is within two words to the left of arg 1 or arg 2",
        candidate='03a1e1a0-93c3-41a8-a905-a535ce8f2b09::span:6822:6837~~03a1e1a0-93c3-41a8-a905-a535ce8f2b09::span:6855:6858',
        label=True,
        semantics=None)

testList=list(testExp)

testing=KeywordBaselineLists(testList)
