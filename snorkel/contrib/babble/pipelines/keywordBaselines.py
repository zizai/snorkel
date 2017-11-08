from snorkel.contrib.babble import Explanation, link_explanation_candidates
import re

### This file will contain a class definition to remove extra conditions
### from explanantions containing a string literal. These modified explanations
### will ultimately be used to construct baselines investigating whether
### the extra condition information improves performance.

class KeywordBaselines:

    def __init__(self, explanation, conjunction='and'):
	self.explanation=explanation
	self.name=explanation.name
	self.condition=explanation.condition
	self.candidate=explanation.candidate
	self.label=explanation.label
	self.semantics=explanation.semantics
 	self.conjunction=conjunction
	self.keywords=[]

    def setKeywords(self):
	self.keywords=re.findall(r'\'(.+?)\'',self.condition)
    
    def makeCondition(self):
	helper="'" + self.keywords[0] + "'"
	for i in range(1,len(self.keywords)):
	    helper=helper + " " + self.conjunction + " " + "'" + self.keywords[i] + "'"
	if len(self.keywords)==1 or self.conjunction=='or':
	    self.condition=helper + " is in the sentence"
	else:
	    self.condition=helper + " are in the sentence"
    
    def modify(self):
	self.setKeywords()
	if(self.keywords):
	    self.makeCondition()
	newExp = Explanation(name = self.name, condition = self.condition, candidate = self.candidate, label = self.label, semantics = self.semantics)
	return newExp
