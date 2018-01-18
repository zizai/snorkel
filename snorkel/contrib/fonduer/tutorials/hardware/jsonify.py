import codecs
import json

def jsonify(doc):
    """Parse out word and location information from a doc.""" 
    in_data = codecs.open(pdf_path + doc.name + "_in.json", encoding="utf-8", mode='w')
    json_in = {"tables": []}
    
    for table in doc.tables:
        table_in = {"words": []}
        table_words = set()
        for cell in table.cells:
            for phrase in cell.phrases:
                word_in = {"file": doc.name,
                           "table": table.position,
                           "text": phrase.text,
                           "top": min(phrase.top),
                           "left": min(phrase.left),
                           "bottom": max(phrase.bottom),
                           "right": max(phrase.right),
                           "row_start": phrase.row_start,
                           "row_end": phrase.row_end,
                           "col_start": phrase.col_start,
                           "col_end": phrase.col_end}
                word_tup = (doc.name,
                            table.position,
                            phrase.text,
                            min(phrase.top),
                            min(phrase.left),
                            max(phrase.bottom),
                            max(phrase.right),
                            phrase.row_start,
                            phrase.row_end,
                            phrase.col_start,
                            phrase.col_end)
                if word_tup not in table_words:
                    table_in["words"].append(word_in)
                    table_words.add(word_tup)

        json_in["tables"].append(table_in)

    json.dump(json_in, in_data, ensure_ascii=False)
    in_data.close()
    return doc.name

if __name__ == "__main__":
    import os
    import sys
    ATTRIBUTE = "stg_temp_max"
    
    os.environ['FONDUERDBNAME'] = ATTRIBUTE
    os.environ['SNORKELDB'] = 'postgres://lwhsiao:123@localhost:5433/' + os.environ['FONDUERDBNAME']
    print(os.environ["FONDUERDBNAME"])
    print(os.environ["SNORKELDB"])

    from snorkel.contrib.fonduer import SnorkelSession
    session = SnorkelSession()
    
    from snorkel.contrib.fonduer.models import Document, Phrase
    from snorkel.contrib.fonduer import HTMLPreprocessor, OmniParser
    from snorkel.contrib.fonduer.models import candidate_subclass

    Part_Attr = candidate_subclass("Part_Attr", ['part', 'attr'])
    
    docs_path = os.environ['FONDUERHOME'] + '/tutorials/hardware/data/train_digikey/html/'
    pdf_path = os.environ['FONDUERHOME'] + '/tutorials/hardware/data/train_digikey/pdf/'
    print(pdf_path)
    print("JSONifying {} Documents...".format(session.query(Document).count()))
    count = 0
    docs = session.query(Document).order_by(Document.name).all()
    for doc in docs:
	count += 1
        print("{}: {}".format(count, jsonify(doc)))

