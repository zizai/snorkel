from snorkel.contrib.babble import Explanation, link_explanation_candidates
import re
###from snorkel.contrib.babble.semparser import Explanation
### This file will contain a class definition to remove extra conditions
### from explanantions containing a string literal. These modified explanations
### will ultimately be used to construct baselines investigating whether
### the extra condition information improves performance.

class KeywordBaselines:


    def __init__(self, explanation):
	self.explanation=explanation
	self.name=explanation.name
	self.condition=explanation.condition
	self.candidate=explanation.candidate
	self.label=explanation.label
	self.semantics=explanation.semantics

    def findKeywords(self):
	keyword=re.findall(r'\'(.+?)\'',self.condition)
	return keyword

    def modify(self, keyword):
	newExp = Explanation(name = self.name, condition = keyword, candidate = self.candidate, label = self.label, semantics = self.semantics)
	return newExp

testExp = Explanation(
        name='LF_spouse_to_left',
        condition="the word 'wife' is within two words to the left of arg 1 or arg 2",
        candidate='03a1e1a0-93c3-41a8-a905-a535ce8f2b09::span:6822:6837~~03a1e1a0-93c3-41a8-a905-a535ce8f2b09::span:6855:6858',
        label=True,
        semantics=None)

testing = KeywordBaselines(testExp)
print(testing.explanation)
keyword=testing.findKeywords()
print(keyword[0])
print(testing.modify(keyword[0]))
