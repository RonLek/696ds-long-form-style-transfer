import zero_shot
import few_shot
import self_discover

def perform_zero_shot(source_doc, reference_doc, publication_name):
    return zero_shot.run_zero_shot(source_doc, reference_doc, publication_name)

def perform_few_shot(source_doc, reference_docs=None, publication_name=None, paired_docs=None):
    return few_shot.run_few_shot(source_doc, reference_docs=reference_docs, publication_name=publication_name, paired_docs=paired_docs)

def perform_self_discover(source_doc, reference_doc, publication_name, reasoning_modules):
    return self_discover.run_self_discover(source_doc, reference_doc, publication_name, reasoning_modules)
