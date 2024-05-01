import zero_shot
import few_shot
import self_discover

def perform_zero_shot(source_doc, reference_doc, publication_name, use_publication_name=True, use_reference_doc=True):
    return zero_shot.run_zero_shot(source_doc, reference_doc, publication_name, use_publication_name=True, use_reference_doc=True)

def perform_few_shot(source_doc, reference_docs, publication_name):
    return few_shot.run_few_shot(source_doc, reference_docs, publication_name)

def perform_self_discover(source_doc, reference_docs, publication_name, reasoning_modules):
    return self_discover.run_self_discover(source_doc, reference_docs, publication_name, reasoning_modules)