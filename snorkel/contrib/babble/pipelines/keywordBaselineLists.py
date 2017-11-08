from keywordBaselines import KeywordBaselines
from snorkel.contrib.babble import Explanation, link_explanation_candidates
import re

class KeywordBaselineLists(KeywordBaselines):

    def __init__(self, explanationList):
	self.explanationList = explanationList

    def modifyList(self):
	modifiedList = list(self.explanationList) 
	for explanation in modifiedList:
	    print(explanation)
	    keyword=KeywordBaselines(explanation).findKeywords()
	    print("Keyword: " + str(keyword))
	return modifiedList

basic = [
    Explanation(
        name='LF_spouse_to_left',
        condition="there is a spouse word within two words to the left of arg 1 or arg 2",
        candidate='03a1e1a0-93c3-41a8-a905-a535ce8f2b09::span:6822:6837~~03a1e1a0-93c3-41a8-a905-a535ce8f2b09::span:6855:6858',
        label=True,
        semantics=None),
    Explanation(
        name='LF_no_spouse_in_sentence',
        condition="there are no spouse words in the sentence",
        candidate='d0de6a86-66d5-40e0-b345-6c86d2047c07::span:1634:1638~~d0de6a86-66d5-40e0-b345-6c86d2047c07::span:1650:1659',
        label=False,
        semantics=None),
    Explanation(
        name='LF_married_after',
        condition="the word 'and' is between arg 1 and arg 2 and 'married' or 'marriage' is after arg 2",
        candidate='e522e66f-ad1f-4b8b-a532-4f030a8e7a75::span:4054:4059~~e522e66f-ad1f-4b8b-a532-4f030a8e7a75::span:4085:4091',
        label=True,
        semantics=None),
    Explanation(
        name='LF_family_between',
        condition="there is a family word between arg 1 and arg 2",
        candidate='768f241b-786d-475e-a55c-9683ecdeeb86::span:518:529~~768f241b-786d-475e-a55c-9683ecdeeb86::span:637:638',
        label=False,
        semantics=None),
    Explanation(
        name='LF_family_to_left',
        condition="there is a family word within three words to the left of arg 1 or arg 2",
        candidate='b86261b6-62c3-456d-8ed0-458f781776f7::span:42:53~~b86261b6-62c3-456d-8ed0-458f781776f7::span:72:91',
        label=False,
        semantics=None),
    Explanation(
        name='LF_other_between',
        condition="there is an other word between arg 1 and arg 2",
        candidate='3375a3c2-9b8a-423a-8334-32fe860be60e::span:3939:3948~~3375a3c2-9b8a-423a-8334-32fe860be60e::span:3967:3981',
        label=False,
        semantics=None),
]

testing=KeywordBaselineLists(basic)
testing.modifyList()
#print(testing.modifyList())
