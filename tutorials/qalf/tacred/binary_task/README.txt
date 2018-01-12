Note from Robin on 1/9/18 about how binary classification task datasets were 
created:

"I decided not to subsample negatives--for each relation, we use every candidate 
that has subject and object mentions of the correct type.  Note that this 
includes both "no_relation" examples and examples tagged with a different 
relation (e.g., for per:spouses, we include examples from per:siblings, which 
should be treated as negatives).  I found that for relations with many positive 
examples, the positive/negative imbalance isn't that bad."