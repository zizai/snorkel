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


class ExplanationForm(FlaskForm):
	label = RadioField('Label', choices=[('True', 'True, they are spouses.'),('False', 'False, are not spouses.')])
	explanation = TextAreaField('Explanation', [validators.InputRequired()])
	skip = SubmitField(label="Skip")
	submit_explaination = SubmitField('Submit')
	commit_lfs = SubmitField('Commit')


# https://stackoverflow.com/questions/46653424/flask-wtforms-fieldlist-with-booleanfield
def parse_list_form_builder(parses):
	class ParseListForm(ExplanationForm):
		pass
	for (i, parseName) in enumerate(parses):
		setattr(ParseListForm, 'parse', BooleanField(label=parseName))
	return ParseListForm()

def candidate_html(c):
	chunks = get_text_splits(c)
	div_tmpl = u'''<div style="border: 1px #858585; box-shadow:0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
    background-color:#FDFDFD; padding:5pt 10pt 5pt 10pt; width: 80%; margin: auto; margin-top: 2%">{}</div>'''
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

main_page = Blueprint('main_page', __name__, template_folder='templates')

@main_page.route("/", methods=['GET', 'POST'])
def index():
	bs = current_app.config['babble_stream_object']
	c = bs.next()

	candidate = candidate_html(c)

	form = ExplanationForm(request.form)

	# CASE 1: SKIP THE CURRENT SAMPLE
	if form.skip.data:
		c = bs.next()
		print bs
		candidate = candidate_html(c)
		return render_template('index.html', candidate=candidate, form=form)

	elif request.method == 'POST' and form.commit_lfs.data:
		print bs
		bs.commit([0])

	# CASE 2: SUBMIT AN EXPLAINATION
	elif request.method == 'POST' and form.validate():
		label = form.label.data
		if label == 'True': label = True
		elif label == 'False': label = False
		condition = form.explanation.data 
		explanation = Explanation(condition, label, candidate=c)
		print label
		print condition
		print explanation
		parse_results = bs.apply(explanation)
		if len(parse_results) == 3:
			parse_list, conf_matrix_list, stats_list = parse_results
			form = parse_list_form_builder(parse_list)
			return render_template('index.html', candidate=candidate, form=form, conf_matrix=conf_matrix_list)
		else:
			# generated no new parses
			return render_template('index.html', candidate=candidate, form=form, no_parses_msg = "True")
		return redirect('/error') # shouldn't be here

	return render_template('index.html', candidate=candidate, form=form)