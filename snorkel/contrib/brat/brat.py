import os
import re
import sys
import codecs
import shutil
import signal
import tarfile
import subprocess

from .utils import download
from IPython.display import IFrame, display, HTML
from ...models import Span, Candidate, Document, Sentence


class BratAnnotator(object):
    """
    Snorkel Interface fo
    Brat Rapid Annotation Tool
    http://brat.nlplab.org/

    This implements a minimal interface for annotating simple relation pairs and their entities.

    """
    def __init__(self, session, candidate_class, tmpl_path='tmpl.config', encoding="utf-8",
                 address='localhost', port=8001):
        """
        Begin BRAT session by:
        - checking that all app files are downloaded
        - creating/validate a local file system mirror of documents
        - launch local server

        :param session:
        :param candidate_class:
        :param address:
        :param port:
        """
        self.session = session
        self.candidate_class = candidate_class
        self.address = address
        self.port = port
        self.encoding = encoding

        self.path = os.path.dirname(os.path.realpath(__file__))
        self.brat_root = 'brat-v1.3_Crunchy_Frog'
        self.data_root = "{}/{}/data".format(self.path, self.brat_root)

        # load brat annotation config template
        mod_path = "{}/{}".format(os.path.abspath(os.path.dirname(__file__)), tmpl_path)
        self.config_tmpl = "".join(open(mod_path, "rU").readlines())

        self._download()
        self.process_group = None
        self._start_server()

    def init_collection(self, doc_root, split=None, cid_query=None, overwrite=False):
        """
        Initialize document collection on disk

        :param doc_root:
        :param split:
        :param cid_query:
        :param overwrite:
        :return:
        """
        assert split != None or cid_query != None

        collection_path = "{}/{}".format(self.data_root, doc_root)
        if os.path.exists(collection_path) and not overwrite:
            msg = "Error! Collection at '{}' already exists.".format(doc_root)
            msg += "Please set overwrite=True to erase all existing annotations.\n"
            sys.stderr.write(msg)
            return

        # remove existing annotations
        if os.path.exists(collection_path):
            shutil.rmtree(collection_path, ignore_errors=True)
            print("Removed existing collection at '{}'".format(doc_root))

        # create subquery based on candidate split
        if split != None:
            cid_query = self.session.query(Candidate.id).filter(Candidate.split == split).subquery()

        # generate all documents for this candidate set
        doc_ids = get_doc_ids_by_split(self.session, self.candidate_class, cid_query)
        documents = self.session.query(Document).filter(Document.id.in_(doc_ids)).all()

        # create collection on disk
        os.makedirs(collection_path)

        for doc in documents:
            text = doc_to_text(doc)
            outfpath = "{}/{}".format(collection_path, doc.name)
            with codecs.open(outfpath + ".txt","w", self.encoding, errors='ignore') as fp:
                fp.write(text)
            with codecs.open(outfpath + ".ann","w", self.encoding, errors='ignore') as fp:
                fp.write("")

        # add minimal annotation.config based on candidate_subclass info
        self._init_annotation_config(self.candidate_class, doc_root)

    def view(self, doc_root, document=None, new_window=True):
        """
        Launch web interface for Snorkel. The default mode launches a new window.
        This is preferred as we have limited control of default widget sizes,
        which can cause display issues when rendering embedded in a Jupyter notebook cell.

        If no document is provided, we create a browser link to the file view mode of BRAT.
        Otherwise we create a link directly to the provided document

        :param document:
        :param new_window:
        :return:

        :param doc_root:
        :param document:
        :param new_window:
        :return:
        """
        # http://localhost:8001/index.xhtml#/pain/train/
        doc_name = document.name if document else ""
        url = "http://{}:{}/index.xhtml#/{}/{}".format(self.address, self.port, doc_root, doc_name)

        if new_window:
            # NOTE: if we use javascript, we need pop-ups enabled for a given browser
            #html = "<script>window.open('{}','_blank');</script>".format(url)
            html = "<a href='{}' target='_blank'>Launch BRAT</a>".format(url)
            display(HTML(html))

        else:
            self.display(url)

    def display(self, url, width='100%', height=700):
        """
        Create embedded iframe view of BRAT

        :param width:
        :param height:
        :return:
        """
        display(HTML("<style>.container { width:100% !important; }</style>"))
        display(IFrame(url, width=width, height=height))

    def _close(self):
        '''
        Kill the process group linked with this server.
        :return:
        '''
        print("Killing BRAT server [{}]...".format(self.process_group.pid))
        if self.process_group is not None:
            try:
                os.kill(self.process_group.pid, signal.SIGTERM)
            except Exception as e:
                sys.stderr.write('Could not kill BRAT server [{}] {}\n'.format(self.process_group.pid, e))

    def _start_server(self):
        """
        Launch BRAT server

        :return:
        """
        cwd = os.getcwd()
        os.chdir("{}/{}/".format(self.path, self.brat_root))
        cmd = ["python", "standalone.py", "{}".format(self.port)]
        self.process_group = subprocess.Popen(cmd, cwd=os.getcwd(), env=os.environ, shell=False )
        os.chdir(cwd)
        url = "http://{}:{}".format(self.address, self.port)
        print("Launching BRAT server at {} [pid={}]...".format(url, self.process_group.pid))

    def __del__(self):
        '''
        Clean-up this object by forcing the server process to shut-down
        :return:
        '''
        self._close()

    def _download(self):
        """
        Download and install latest version of BRAT
        :return:
        """
        fname = "{}/{}".format(self.path, 'brat-v1.3_Crunchy_Frog.tar.gz')
        if os.path.exists("{}/{}/".format(self.path,self.brat_root)):
            return

        url = "http://weaver.nlplab.org/~brat/releases/brat-v1.3_Crunchy_Frog.tar.gz"
        print("Downloading BRAT [{}]...".format(url))
        download(url, fname)

        # install brat
        cwd = os.getcwd()
        os.chdir(self.path)
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()

        print("Installing BRAT...")
        # setup default username and passwords
        shutil.copyfile("install.sh", "{}/install.sh".format(self.brat_root))
        os.chdir("{}/{}".format(self.path, self.brat_root))
        subprocess.call(["./install.sh"])

        # cleanup
        os.chdir(cwd)
        os.remove(fname)

    def _login(self):
        """
        BRAT requires a user login in order to edit annotations. We do this
        automatically behind the scenes.

        TODO: Not yet! User must login manually.
        :return:
        """
        # TODO -- this requires some jquery/cookie magic to automatically handle logins.
        pass

    def _init_annotation_config(self, candidate_class, doc_root):
        """

        :param candidate_class:
        :return:
        """
        collection_path = "{}/{}".format(self.data_root, doc_root)
        # create config file
        config = self._create_config([candidate_class])
        config_path = "{}/annotation.conf".format(collection_path)
        with codecs.open(config_path, 'w', self.encoding) as fp:
            fp.write(config)

    def _create_config(self, candidate_types):
        """
        Export a minimal BRAT configuration schema defining
        a binary relation and two argument types.

        :param candidate_type:
        :return:
        """
        entity_defs, rela_defs = [], []
        for stype in candidate_types:
            rel_type = str(stype.type).rstrip(".type")
            arg_types = [key.rstrip("_id") for key in stype.__dict__ if "_id" in key]
            arg_types = [name[0].upper()+name[1:] for name in arg_types]

            # HACK: Assume all args that differ by just a number are
            # of the same type, e.g., person1, person2
            arg_types = [re.sub("\d+$", "", name) for name in arg_types]

            entity_defs.extend(set(arg_types))
            if len(arg_types) > 1:
                rela_name = [str(stype.type).replace(".type","")] + arg_types
                rela_defs.append("{}\tArg1:{}, Arg2:{}".format(*rela_name))

        entity_defs = set(entity_defs)
        rela_defs = set(rela_defs)
        return self.config_tmpl.format("\n".join(entity_defs), "\n".join(rela_defs), "", "")

def get_doc_ids_by_split(session, candidate_class, cid_subquery):
    '''
    Given a candidate set split, return all corresponding parent document ids
    '''
    # TODO: better way to fetch argument id?
    arg = [arg for arg in candidate_class.__dict__.keys() if "_id" in arg][0]

    q1 = session.query(candidate_class.__dict__[arg]).filter(Candidate.id.in_(cid_subquery)).subquery()
    q2 = session.query(Span.sentence_id).filter(Span.id.in_(q1)).subquery()
    return session.query(Sentence.document_id).filter(Sentence.id.in_(q2)).distinct()


def doc_to_text(doc, sent_delim='\n'):
    """
    Convert document object to original text represention.
    Assumes parser offsets map to original document offsets
    :param doc:
    :param sent_delim:
    :return:
    """
    text = []
    for sent in doc.sentences:
        offsets = map(int, sent.stable_id.split(":")[-2:])
        char_start, char_end = offsets
        text.append({"text": sent.text, "char_start": char_start, "char_end": char_end})

    s = ""
    for i in range(len(text) - 1):
        gap = text[i + 1]['char_start'] - text[i]['char_end']
        s += text[i]['text'] + (sent_delim * gap)

    return s