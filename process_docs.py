import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from zensols.calamr import ApplicationFactory, FlowGraphResult

import config
from pyg_export import PyTorchGeometricExporter as PyGExport


def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_labels_as_map(dataset: str, subset: str) -> dict[str, int]:
    with open(f"calamr_input/{dataset}/{subset}/final_decisions.json") as f:
        final_decisions = json.load(f)
    return {decision["id"]: decision["final_decision"] for decision in final_decisions}


def _get_corpus(dataset: str, subset: str) -> tuple[dict[str, Any], dict[str, int]]:
    with open(f"calamr_input/{dataset}/{subset}/calamr_data.json") as f:
        corpus = json.load(f)

    labels = _load_labels_as_map(dataset=dataset, subset=subset)

    return corpus, labels


def process_docs(dataset: str = "medhallu", subset: str = "labeled", seed: int = 42):
    _set_seed(seed=seed)

    resources = ApplicationFactory.get_resources()
    pyg_export = PyGExport()

    corpus, labels = _get_corpus(dataset=dataset, subset=subset)

    # make output paths
    output_path = Path(f"{config.RESULTS_DIR}/{dataset}/{subset}")
    output_path.mkdir(parents=True, exist_ok=True)

    # make pyg output paths
    pyg_output_path = output_path / "pyg"
    pyg_output_path.mkdir(parents=True, exist_ok=True)

    # access an ad hoc corpus as defined above with the list of dictionaries above
    with resources.adhoc(corpus) as r:
        # list the keys in the corpus, each of which is available as a document or
        # alignment as flow metrics/data
        docs = list(r.documents.keys())

        for doc_id in docs:
            try:
                flow: FlowGraphResult = r.alignments[doc_id]
                if flow.is_failure():
                    print(f"Error processing document {doc_id}")
                    continue

                pyg_data = pyg_export.export_alignment_graph(flow)
                label = labels[doc_id]
                pyg_data.y = torch.tensor([label], dtype=torch.long)

                pyg_path = Path(output_path) / "pyg" / f"{doc_id}.pt"
                torch.save(pyg_data, pyg_path)
            except Exception:
                print(f"Error processing document {doc_id}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    argparser = ArgumentParser()
    argparser.add_argument("--dataset", type=str, default="psiloqa")
    argparser.add_argument("--seed", type=int, default=42)
    args = argparser.parse_args()

    if args.dataset == "psiloqa":
        process_docs(dataset="psiloqa", subset="test")
        process_docs(dataset="psiloqa", subset="train")
        process_docs(dataset="psiloqa", subset="validation")
    elif args.dataset == "medhallu":
        process_docs(dataset="medhallu", subset="labeled")
        process_docs(dataset="medhallu", subset="artificial")
