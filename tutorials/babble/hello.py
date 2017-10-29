from flask import Flask, flash, redirect, render_template, request, session, abort
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
from wtforms.fields import RadioField, SubmitField
from snorkel.contrib.babble import BabbleStream
from snorkel.lf_helpers import *
from tutorials.babble.spouse.spouse_examples import get_explanations, get_user_lists

# dependencies:
# flask
# Flask-WTF
# wtforms

app = Flask(__name__)

app.secret_key = 'myverylongsecretkey'

class ExplanationForm(FlaskForm):
	label = RadioField('Label', choices=[('True', 'True, they are spouses.'),('False', 'False, are not spouses.')])
	explanation = TextAreaField('Explanation', [validators.InputRequired()])
	skip = SubmitField(label="Skip")

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

@app.route("/", methods=['GET', 'POST'])
def index():
	config = {
	'b': 0.5,
	'babbler_candidate_split': 0,
	'babbler_label_split': 0,
	'beam_width': 10,
	'candidate_entities': ['person1', 'person2'],
	'candidate_name': 'Spouse',
	'db_name': 'babble_spouse_demo',
	'debug': False,
	'deps_thresh': 0.01,
	'disc_eval_batch_size': 256,
	'disc_init_params': {'n_threads': 16, 'seed': 123},
	'disc_model_class': 'lstm',
	'disc_model_search_space': 10,
	'disc_params_default': {'batch_size': 128,
	'dim': 50,
	'dropout': 0.5,
	'lr': 0.01,
	'max_sentence_length': 100,
	'n_epochs': 20,
	'print_freq': 1,
	'rebalance': 0.25},
	'disc_params_range': {'dim': [50, 100],
	'dropout': [0.1, 0.25, 0.5],
	'lr': [0.01, 0.001, 0.0001],
	'rebalance': [0.1, 0.25]},
	'display_accuracies': True,
	'display_correlation': False,
	'display_learned_accuracies': False,
	'display_marginals': True,
	'do_filter_consistency': True,
	'do_filter_duplicate_semantics': True,
	'do_filter_duplicate_signatures': True,
	'do_filter_uniform_signatures': True,
	'domain': 'spouse',
	'end_at': 10,
	'gen_f_beta': 1,
	'gen_init_params': {'class_prior': False,
	'lf_class_propensity': False,
	'lf_prior': False,
	'lf_propensity': True,
	'seed': 123},
	'gen_model_search_space': 10,
	'gen_params_default': {'decay': 0.99,
	'epochs': 50,
	'init_class_prior': -1.15,
	'reg_param': 0.5,
	'step_size': 0.0001},
	'gen_params_range': {'LF_acc_prior_weight_default': [0.5, 1.0, 1.5],
	'decay': [0.9, 0.95, 0.99],
	'reg_param': [0.0, 0.01, 0.1, 0.25, 0.5],
	'step_size': [0.01, 0.001, 0.0001, 1e-05]},
	'learn_deps': False,
	'log_dir': 'logs',
	'max_docs': None,
	'max_train': None,
	'no_plots': False,
	'parallelism': 1,
	'postgres': False,
	'reports_dir': 'reports',
	'seed': 0,
	'splits': [0, 1, 2],
	'start_at': 0,
	'supervision': 'generative',
	'top_k': -1,
	'verbose': True
	}
	# Get DB connection string and add to globals
	# default_db_name = 'babble_' + config['domain'] + ('_debug' if config.get('debug', False) else '')
	# DB_NAME = config.get('db_name', default_db_name)
	# if 'postgres' in config and config['postgres']:
	#     DB_TYPE = 'postgres'
	# else:
	#     DB_TYPE = 'sqlite'
	#     DB_NAME += '.db'
	# DB_ADDR = "localhost:{0}".format(config['db_port']) if 'db_port' in config else ""
	# os.environ['SNORKELDB'] = '{0}://{1}/{2}'.format(DB_TYPE, DB_ADDR, DB_NAME)
	# print("$SNORKELDB = {0}".format(os.environ['SNORKELDB']))

	session = SnorkelSession()

	# config = merge_configs(config)

	# if config['debug']:
	#     print("NOTE: --debug=True: modifying parameters...")
	#     config['max_docs'] = 100
	#     config['gen_model_search_space'] = 2
	#     config['disc_model_search_space'] = 2
	#     config['gen_params_default']['epochs'] = 25
	#     config['disc_params_default']['n_epochs'] = 5

	Spouse = candidate_subclass('Spouse', ['person1', 'person2'])
	candidate_class = Spouse
	pipe = SpousePipeline(session, Spouse, config)
	# pipe.parse()
	# pipe.extract()
	# pipe.load_gold()
	
	candidates = session.query(Spouse).filter(Spouse.split == 0).all()
	spouse_explanations = get_explanations(candidates)
	spouse_user_lists = get_user_lists()

	bs = BabbleStream(session, strategy='linear', candidate_class=Spouse)
	# bs.preload(explanations=spouse_explanations, user_lists=spouse_user_lists)
	# bs.get_label_matrix()

	c = bs.next()
	# print c
	# sv = SentenceNgramViewer([c], session, n_per_page=1, height=150)
	# candidate = sv.get_selected()[0].sentence.text

	candidate = candidate_html(c)

	form = ExplanationForm(request.form)
	if form.validate and form.skip.data:
		c = bs.next()
		candidate = candidate_html(c)
		return render_template('index.html', candidate=candidate, form=form)

	elif request.method == 'POST' and form.validate():
		label = form.label.data
		if label == 'True':
			label = True
		elif label == 'False':
			label = False
		condition = form.explanation.data 
		print label
		print condition
		explanation = Explanation(condition, label, candidate=c, name='')
		print explanation
		parse_results = bs.apply(explanation)
		print len(parse_results)
		if len(parse_results) == 2:
			conf_matrix_list, stats_list = parse_results
			print conf_matrix_list
			print stats_list
			return render_template('index.html', candidate=candidate, form=form, conf_matrix=conf_matrix_list)
		else:
			# generated no new parses
			return render_template('index.html', candidate=candidate, form=form, no_parses_msg = "True")
		return redirect('/error') # shouldn't here

	return render_template(
		'index.html', candidate=candidate, form=form)

@app.route("/success", methods=['GET', 'POST'])
def success():
	return render_template('success.html')
 
if __name__ == "__main__":
    app.run()
