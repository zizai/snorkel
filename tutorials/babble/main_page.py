from flask import current_app, Blueprint, Flask, flash, redirect, render_template, request, session
import os
import sys
sys.path.append('/Users/Stephaniewang/Documents/snorkel')
from snorkel import SnorkelSession
from snorkel.contrib.babble.pipelines import merge_configs
from snorkel.models import candidate_subclass
from snorkel.contrib.babble import ExplanationIO
from tutorials.babble.spouse import SpousePipeline
from snorkel.viewer import SentenceNgramViewer
from snorkel.contrib.babble import Explanation
from snorkel.contrib.babble import Babbler
from flask_wtf import FlaskForm
from wtforms import StringField, validators, TextAreaField
from wtforms.fields import RadioField, SubmitField, BooleanField
from flask_bootstrap import Bootstrap
from snorkel.contrib.babble import BabbleStream
from snorkel.lf_helpers import *
from tutorials.babble.spouse.spouse_examples import get_explanations, get_user_lists
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

#----------# NOTES #----------#

class ExplanationForm(FlaskForm):
	explanation = TextAreaField('Explanation', [validators.DataRequired()])

class LabelingFunctionForm(FlaskForm):
	pass

# https://stackoverflow.com/questions/46653424/flask-wtforms-fieldlist-with-booleanfield
def parse_list_form_builder(parses, translator):
	class ParseListForm(FlaskForm):
		pass
	for (idx, parse) in enumerate(parses):
		label = "Parse {}:\n{}\n".format(idx, translator(parse.semantics))
		setattr(ParseListForm, 'parse', BooleanField(label=label))
	return ParseListForm()

def candidate_html(c):
	chunks = get_text_splits(c)
	div_tmpl = u'''<div style="border: 1px #858585; box-shadow:0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    background-color:#FDFDFD; padding:5pt 10pt 5pt 10pt; width: 80%; margin: auto; margin-top: 2%">	
    {}</div>'''
	arg_tmpl = u'<b style="background-color:#ffd77c;padding:1pt 5pt 1pt 5pt;">{}</b>'
	sent_tmpl = u'<p style="font-size:12pt;">{}</p>'
	text = u""
	for s in chunks[0:]:
	    if s in [u"{{A}}", u"{{B}}"]:
	        span = c[0].get_span() if s == u"{{A}}" else c[1].get_span()
	        text += arg_tmpl.format(span)
	    else:
	        text += s.replace(u"\n", u"<BR/>")
	html = div_tmpl.format(sent_tmpl.format(text.strip()))
	return html

def sentence_html(c):
	chunks = get_text_splits(c)
	arg_tmpl = u'<b style="padding:1pt 5pt 1pt 5pt;">{}</b>'
	sent_tmpl = u'<p style="font-size:10pt;">{}</p>'
	text = u""
	for s in chunks[0:]:
	    if s in [u"{{A}}", u"{{B}}"]:
	        span = c[0].get_span() if s == u"{{A}}" else c[1].get_span()
	        text += arg_tmpl.format(span)
	    else:
	        text += s.replace(u"\n", u"<BR/>")
	html = sent_tmpl.format(text.strip())
	return html

def apply_filtered_analysis(filtered_parses, translator):
	sent_tmpl = u'<dl class="row">{}</dl>'
	dt_tmpl = u'<dt class = "col-sm-3">{}</dt>'
	dd_tmpl = u'<dd class= "col-sm-9">{}</dd>'
	text = u""

	if not any(filtered_parses.values()):
		return None
	for filter_name, parses in filtered_parses.items():
	    # if parses:
	        # text += ("Filter {} removed {} parse(s):".format(filter_name, len(parses)))
	    for i, filtered_parse in enumerate(parses):
	        # text += ("\n#{} Filtered parse:".format(i))
	        text += dt_tmpl.format("Explanation (source)")
	        text += dd_tmpl.format(filtered_parse.parse.explanation)
	        text += dt_tmpl.format("Parse (pseudocode)")
	        text += dd_tmpl.format(translator(filtered_parse.parse.semantics))

	        if filter_name == 'DuplicateSemanticsFilter':
	            text += dt_tmpl.format("Reason: Collision with parse from this explanation")
	            text += dd_tmpl.format(filtered_parse.reason.explanation)
	            
	        elif filter_name == 'ConsistencyFilter':
				candidate = filtered_parse.reason
				consistency_filter = 'Reason: Inconsistent with candidate'
				text += dt_tmpl.format(consistency_filter)
				text += dd_tmpl.format(unicode(filtered_parse.reason.get_parent().text).encode('utf-8'))
	            
	        elif filter_name == 'UniformSignatureFilter':
	            text += dt_tmpl.format("Reason:")
	            text += dd_tmpl.format(filtered_parse.reason)
	            
	        elif filter_name == 'DuplicateSignatureFilter':
	            text += dt_tmpl.format("Reason: Collision with parse from this explanation:")
	            text += dd_tmpl.format(filtered_parse.reason.explanation)
	            text += "<br/>"
	html = sent_tmpl.format(text.strip())
	return html

def get_metrics(bs):
	# get the metrics to display
	lf_stats = bs.get_lf_stats().to_html(columns=['Coverage', 'Overlaps', 'TP', 'FP', 'FN', 'TN', 'Empirical Acc.'])
	coverage = bs.get_global_coverage().numer/float(bs.get_global_coverage().denom) * 100
	num_labels = len(bs.get_lfs())
	num_examples = len(bs.get_explanations())
	metrics = {'lf_stats': lf_stats, 'coverage': coverage, 'num_labels': num_labels, 'num_examples': num_examples}
	return metrics

def finish_pipeline(bs):
		pipe = current_app.config['pipe']
		pipe.lfs = bs.get_lfs()
		pipe.label()
		pipe.supervise()
		pipe.featurize()
		pipe.classify()

main_page = Blueprint('main_page', __name__, template_folder='templates')

@main_page.route("/", methods=['GET', 'POST'])
def index():
	bs = current_app.config['babble_stream_object']
	form1 = ExplanationForm(request.form)
	form2 = LabelingFunctionForm(request.form)

	metrics = get_metrics(bs)

	# CASE 1: SKIP THE CURRENT SAMPLE
	if request.method == 'POST' and "skip" in request.form:
		current_app.config['candidate'] = bs.next() # GET THE NEXT CANDIDATE
		candidate = candidate_html(current_app.config['candidate'])
		return render_template('index.html', candidate=candidate, form=form1, metrics=metrics)

	# CASE 2: SUBMIT AN EXPLANATION
	if request.method == 'POST' and form1.validate_on_submit() and ("pos" in request.form or "neg" in request.form):
		candidate = candidate_html(current_app.config['candidate'])
		if "pos" in request.form: label = True
		elif "neg" in request.form: label = False
		condition = form1.explanation.data 
		explanation = Explanation(condition, label, candidate=current_app.config['candidate'])

		###### for debugging purposes ######
		print current_app.config['candidate']
		print label
		print condition
		print explanation
		###################################

		parse_results = bs.apply(explanation)
		print len(parse_results)
		if len(parse_results[0]) > 0:
			parse_list, filtered_parses, conf_matrix_list, stats_list = parse_results
			parse_list = [bs.semparser.grammar.translate(parse.semantics) for parse in parse_list]
			correct_incorrect_sentences = []
			for idx in range(len(conf_matrix_list)):
				tf_sentence_dict = {}
				tf_sentence_dict["correct"] = [sentence_html(sentence) for sentence in conf_matrix_list[idx].correct]
				tf_sentence_dict["incorrect"] = [sentence_html(sentence) for sentence in conf_matrix_list[idx].incorrect]
				tf_sentence_dict["abstain"] = [sentence_html(sentence) for sentence in conf_matrix_list[idx].abstained]
				correct_incorrect_sentences.append(tf_sentence_dict)

			filtered_analysis = apply_filtered_analysis(filtered_parses, bs.semparser.grammar.translate)

			displayed_stats = zip(parse_list, correct_incorrect_sentences, stats_list)
			return render_template('index.html', candidate=candidate, form=form1, stats=displayed_stats, filtered_analysis=filtered_analysis, form2=form2, metrics=metrics)
		else:
			# generated no new parses
			return render_template('index.html', candidate=candidate, form=form1, no_parse_error = "True", metrics=metrics)

	# CASE 3: COMMIT THE LFS 
	elif request.method == 'POST' and "commit" in request.form:
		bs.commit()
		# get the next candidate
		current_app.config['candidate'] = bs.next()
		candidate = candidate_html(current_app.config['candidate'])

		metrics = get_metrics(bs)
		flash('You successfully committed your explanation', 'success')
		return render_template('index.html', candidate=candidate, form=form1, metrics=metrics)

	# CASE 4: FINISH THE PIPELINE -- LABEL DATASET AND TRAIN MODEL
	elif request.method == 'POST' and "finish" in request.form:
		finish_pipeline(bs)
		
	candidate = candidate_html(current_app.config['candidate'])
	return render_template('index.html', candidate=candidate, form=form1, metrics=metrics)

